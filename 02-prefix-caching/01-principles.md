# 前缀缓存原理

> 本节深入解析 Prefix Caching 的核心原理：为什么相同的 prompt 前缀可以安全地共享 KV Cache？Cache 的粒度、匹配机制和驱逐策略如何设计？以及 Cached token 的经济学模型。

## 1. 为什么前缀可以共享？—— Causal Attention 的因果性保证

### 1.1 因果性（Causality）回顾

在 decoder-only Transformer（GPT、LLaMA、Claude 等）中，Self-Attention 使用 **causal mask（因果掩码）**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中 $M$ 是一个下三角矩阵，$M_{ij} = 0$ 当 $i \geq j$，$M_{ij} = -\infty$ 当 $i < j$。这意味着：

- **位置 $i$ 的 token 只能 attend to 位置 $0, 1, \ldots, i$ 的 token**
- 位置 $i$ 的 KV 计算结果完全由 $\text{token}_0, \text{token}_1, \ldots, \text{token}_i$ 决定
- 后续追加的 token 不会改变已有位置的 KV 值

### 1.2 形式化证明：前缀不变性

设两个请求的 token 序列为：

- 请求 A：$[t_0, t_1, \ldots, t_{n-1}, t_n^A, t_{n+1}^A, \ldots]$
- 请求 B：$[t_0, t_1, \ldots, t_{n-1}, t_n^B, t_{n+1}^B, \ldots]$

它们共享前缀 $[t_0, t_1, \ldots, t_{n-1}]$。由于 causal mask 的约束，对于任意 $i \leq n-1$：

$$
K_i^A = W_K \cdot x_i^A = W_K \cdot \text{Embed}(t_i) = W_K \cdot \text{Embed}(t_i) = K_i^B
$$

$$
V_i^A = W_V \cdot x_i^A = W_V \cdot \text{Embed}(t_i) = W_V \cdot \text{Embed}(t_i) = V_i^B
$$

注意这里有一个微妙之处：在多层 Transformer 中，第 $l$ 层的输入 $x_i^{(l)}$ 取决于第 $l-1$ 层的输出。但因为 causal mask 在每一层都生效，位置 $i$ 在第 $l$ 层的 hidden state 仍然只依赖位置 $0$ 到 $i$ 的 token。因此，只要前缀相同，**每一层的每个位置的 KV 值都完全相同**。

这就是 Prefix Caching 的理论基础：**因果注意力保证了前缀的 KV Cache 是完全确定性的，与后续的 token 无关。**

### 1.3 Encoder-Decoder 架构的特殊情况

对于 encoder-decoder 模型（如 T5、BART），encoder 使用双向注意力（bidirectional attention），每个位置都能 attend to 所有位置。这意味着更改输入的任何部分都会影响所有位置的表示。因此，encoder-decoder 模型的 encoder 侧 **不支持简单的前缀缓存**——除非整个 encoder 输入完全相同。

Decoder 侧因为仍使用 causal attention + cross-attention，理论上 decoder 侧的 self-attention KV cache 可以复用，但前提是 encoder 输出完全一致，实际意义不大。

## 2. Cache 粒度：token 级 vs block 级 vs 请求级

### 2.1 Token 级缓存

最细粒度的方案：每个 token 的 KV 向量单独缓存和查找。

```
Cache Key: hash(token_0, token_1, ..., token_i)  →  KV[i]
```

**优点：**
- 精确匹配，浪费最少
- 灵活性最高

**缺点：**
- Hash table 条目数 = 所有缓存请求的 token 数之和，内存开销巨大
- 每个 token 都需要一次 hash lookup，查找延迟高
- 与 PagedAttention 的 block 管理机制不兼容

### 2.2 Block 级缓存（主流方案）

将 token 序列按固定大小（通常 16 或 32 个 token）分成 block，以 block 为粒度缓存。

```
Block 0: [t_0, t_1, ..., t_15]     →  KV[0:16]
Block 1: [t_16, t_17, ..., t_31]   →  KV[16:32]
Block 2: [t_32, t_33, ..., t_47]   →  KV[32:48]
```

**优点：**
- 与 PagedAttention 的物理 block 天然对齐
- Hash table 条目数减少 16-32 倍
- 查找只需匹配 block 数（而非 token 数）
- GPU 内存管理更高效（减少碎片）

**缺点：**
- 最后一个 block 可能未填满，浪费 partial block 的匹配机会
- 前缀匹配必须以 block 为边界对齐

vLLM 和大多数生产系统采用这种方案，`block_size` 默认值为 16。

### 2.3 请求级缓存

最粗粒度：缓存整个请求的完整 KV Cache。

```
Cache Key: hash(entire_prompt)  →  KV[0:len(prompt)]
```

**优点：** 实现最简单
**缺点：** 只有完全相同的 prompt 才能命中，hit rate 极低

API 提供商的 Prompt Caching 本质上介于 block 级和请求级之间——它要求前缀以一定的最小长度匹配（如 1024 token），但不需要整个 prompt 完全相同。

### 2.4 粒度对比总结

| 粒度 | Hash 条目数 | 查找开销 | 与 PagedAttention 兼容 | 命中率 |
|------|-------------|----------|------------------------|--------|
| Token 级 | O(N) | 高 | 否 | 最高 |
| Block 级 | O(N/B) | 低 | 是 | 高 |
| 请求级 | O(R) | 最低 | 视实现 | 最低 |

其中 N 是总 token 数，B 是 block size，R 是请求数。

## 3. Hash 匹配机制：如何快速判断两个前缀是否相同

### 3.1 Content Hash 方案

将 block 内的 token ID 序列做 hash：

```python
def compute_block_hash(token_ids: tuple[int, ...], prefix_hash: int) -> int:
    """
    Block hash = hash(前面所有 block 的 hash, 当前 block 的 token IDs)
    这种链式 hash 保证了只有完整前缀匹配才能命中。
    """
    return hash((prefix_hash, token_ids))
```

关键设计：**链式 hash（chained hash）**。每个 block 的 hash 不仅依赖自身的 token 内容，还依赖前面所有 block 的 hash。这确保了：

1. Block 3 的 hash 隐含了 Block 0-2 的内容信息
2. 如果 Block 1 不同，Block 3 的 hash 也必然不同
3. 只需比较最后一个 block 的 hash，就能验证整个前缀是否匹配

```
Block 0 hash = hash(INITIAL, [t_0...t_15])
Block 1 hash = hash(Block_0_hash, [t_16...t_31])
Block 2 hash = hash(Block_1_hash, [t_32...t_47])
```

### 3.2 为什么不用位置信息？

你可能会问：为什么不直接比较 `(position, token_ids)` 而要用链式 hash？原因是：

1. **RoPE 等位置编码已经隐含在 KV 计算中**：KV cache 本身就包含了位置信息，不需要在 hash 中重复编码
2. **链式 hash 天然保证前缀连续性**：不存在 Block 2 匹配但 Block 1 不匹配的情况
3. **不同模型的位置编码方式不同**（absolute vs RoPE vs ALiBi），content hash 方案更通用

### 3.3 Hash 冲突处理

在实践中，hash 冲突的概率极低（使用 64-bit hash 时，birthday problem 告诉我们需要约 $2^{32} \approx 43$ 亿个不同的 block 才有 50% 的冲突概率）。但为了安全起见，部分实现会：

- 使用 128-bit hash 进一步降低冲突
- 在 cache hit 时做一次 token ID 序列的精确比较（verification）
- 通过 hash + 长度双重校验

## 4. Cache 命中的条件：完全前缀匹配

### 4.1 Exact Prefix Match

Prefix Caching 要求 **严格的前缀匹配（exact prefix match）**，即：

- 新请求的 token 序列从位置 0 开始，逐 block 与缓存比较
- 匹配在第一个不同的 block 处停止
- 只有匹配到的前缀部分可以复用

```
缓存:  [A B C D E F G H]  (8 blocks)
新请求: [A B C D X Y Z]   (7 blocks)
匹配:  [A B C D]          (前 4 blocks 匹配)
需计算: [X Y Z]            (后 3 blocks 需要从头计算)
```

### 4.2 为什么不支持子串匹配？

理论上，如果缓存中有 `[A B C]` 的 KV，而新请求是 `[X A B C]`，虽然 `A B C` 的 token 相同，但它们的 KV 值完全不同——因为在 `[X A B C]` 中，`A` 在位置 1 而非位置 0，且受到 `X` 的影响（虽然是 causal attention，但位置编码不同）。

更精确地说：
- 在 `[A B C]` 中，$K_A$ 对应位置 0 的 RoPE 旋转
- 在 `[X A B C]` 中，$K_A$ 对应位置 1 的 RoPE 旋转
- 两者数值不同，不可互换

这是前缀缓存只支持"前缀"而非"子串"的根本原因。

### 4.3 Partial Block 匹配

当前缀长度不是 block size 的整数倍时，最后一个不完整的 block 通常不会被缓存。例如 block size = 16，前缀长度 = 50 token：

- Block 0-2（48 tokens）可以缓存和复用
- 剩余 2 个 token 需要重新计算

这导致了一个实际问题：**前缀长度越短，block 对齐浪费越大**。1024 token 的前缀只浪费至多 15/1024 ≈ 1.5% 的计算，而 20 token 的前缀可能浪费 15/20 = 75%。

## 5. TTL 与驱逐策略

KV Cache 占用的 GPU 显存是有限的。当显存不足时，必须驱逐（evict）部分缓存。

### 5.1 LRU（Least Recently Used）

最常见的策略。维护一个访问时间戳，驱逐最久未被访问的缓存 block。

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()  # key -> (kv_data, last_access_time)

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # 标记为最近使用
            return self.cache[key]
        return None

    def evict(self):
        # 驱逐最久未使用的条目
        self.cache.popitem(last=False)
```

**优点：** 实现简单，适合大多数工作负载
**缺点：** 无法区分"被频繁使用"和"刚好最近被使用一次"

### 5.2 LFU（Least Frequently Used）

跟踪每个 block 的访问频率，驱逐最不常用的。

**优点：** 高频使用的 system prompt 不会被偶发的大请求冲走
**缺点：** 新条目可能因频率低而立即被驱逐（cache pollution），需要 aging 机制

### 5.3 ARC（Adaptive Replacement Cache）

IBM 提出的自适应策略，结合了 LRU 和 LFU 的优点：

- 维护两个 LRU 列表：$L_1$（最近使用一次）和 $L_2$（最近使用多次）
- 动态调整两个列表的大小比例
- 如果 $L_1$ 的 ghost list 频繁被访问，增大 $L_1$ 的比例（偏向 recency）
- 如果 $L_2$ 的 ghost list 频繁被访问，增大 $L_2$ 的比例（偏向 frequency）

ARC 在 LLM Serving 中的实际应用还较少，但它是理论上最优雅的方案之一。

### 5.4 TTL（Time-To-Live）

API 提供商通常给缓存设置 TTL：

| 提供商 | TTL |
|--------|-----|
| Anthropic | 5 分钟（自动刷新） |
| OpenAI | 5-10 分钟（短期）/ 最长约 1 小时 |
| Google Gemini（显式） | 用户自定义，最长 48 小时 |

TTL 的设置是一个 trade-off：
- TTL 过短：cache hit rate 低，用户频繁 cold start
- TTL 过长：占用大量显存/内存，可能缓存了不再需要的数据

### 5.5 驱逐中的引用计数

在 vLLM 中，block 只有在 **引用计数为 0** 时才能被驱逐。如果一个 block 仍被某个正在运行的请求引用，即使它是 LRU 中最老的条目，也不能被驱逐。

```
Block #42:
  ref_count = 2    # 被 Request A 和 Request B 同时引用
  last_access = 10s ago
  → 不可驱逐（ref_count > 0）

Block #17:
  ref_count = 0    # 所有引用它的请求都已完成
  last_access = 5s ago
  → 可以驱逐
```

## 6. Cached Token 的成本结构

### 6.1 Anthropic 的定价模型

以 Claude Sonnet 4 为例（截至 2025 年中的定价）：

| 类型 | 价格（每百万 token） | 相对基础价格 |
|------|---------------------|-------------|
| 基础 input | $3.00 | 1.0x |
| Cache write | $3.75 | 1.25x |
| Cache read | $0.30 | 0.1x |
| Output | $15.00 | 5.0x |

Cache write 比基础 input 贵 25%，因为需要额外的存储和索引成本。但 cache read 只需 10% 的成本，因为跳过了 prefill 计算。

### 6.2 盈亏平衡分析

假设一个 system prompt 有 $N$ 个 token，被 $k$ 次请求复用：

**不使用 cache 的总成本：**
$$C_{\text{no\_cache}} = k \cdot N \cdot P_{\text{input}}$$

**使用 cache 的总成本：**
$$C_{\text{cache}} = N \cdot P_{\text{write}} + (k-1) \cdot N \cdot P_{\text{read}}$$

（第一次请求是 cache write，后续 $k-1$ 次是 cache read）

盈亏平衡点：$C_{\text{cache}} = C_{\text{no\_cache}}$

$$N \cdot 1.25P + (k-1) \cdot N \cdot 0.1P = k \cdot N \cdot P$$

$$1.25 + 0.1(k-1) = k$$

$$1.25 + 0.1k - 0.1 = k$$

$$1.15 = 0.9k$$

$$k = 1.28$$

**结论：Anthropic 的定价下，同一前缀只要被使用超过 1.28 次（即 2 次），使用 cache 就已经省钱了。**

### 6.3 节省比例的计算

当复用 $k$ 次时，节省的比例为：

$$\text{Savings} = 1 - \frac{C_{\text{cache}}}{C_{\text{no\_cache}}} = 1 - \frac{1.25 + 0.1(k-1)}{k}$$

| 复用次数 $k$ | 节省比例 |
|-------------|---------|
| 2 | 27.5% |
| 5 | 67.0% |
| 10 | 78.5% |
| 50 | 87.7% |
| 100 | 88.9% |
| $\infty$ | 90.0% |

当 $k \to \infty$ 时，节省趋近于 $1 - 0.1 = 90\%$，即 cache read 的折扣率。

### 6.4 OpenAI 的不同模型

OpenAI 的 Prompt Caching 在定价上更简单：

- **Cache write：无额外费用**（与普通 input 同价）
- **Cache read：50% off**（半价）
- **自动触发**：不需要手动标记 cache breakpoint

这意味着 OpenAI 的盈亏平衡点是 $k = 1$，即 **任何复用都是纯赚**。但折扣力度（50% off）不如 Anthropic（90% off）。

### 6.5 深入经济学模型：何时使用 cache 不划算？

虽然 cache 看起来总是有利的，但有几种情况需要注意：

**1. 前缀太短，无法触发 cache**
- Anthropic 要求至少 1024 token（Claude Sonnet/Opus），Haiku 要求 2048 token
- 如果 system prompt 只有 500 token，无法使用 prompt caching

**2. 前缀变化频繁**
- 如果每次请求的前缀都不同，每次都是 cache write（1.25x），反而比不 cache 更贵
- 例如：在 system prompt 中包含当前时间戳、随机 session ID 等

**3. Cache 在 TTL 内未被再次命中**
- 写入了 cache 但在 5 分钟 TTL 内没有第二次请求
- 白白多付了 25% 的 write 成本

**4. 请求模式分析**

定义 cache hit rate $\alpha = \frac{\text{cache hits}}{\text{total requests}}$，则：

$$C_{\text{avg}} = (1-\alpha) \cdot N \cdot P_{\text{write}} + \alpha \cdot N \cdot P_{\text{read}}$$

$$= N \cdot P \cdot [(1-\alpha) \cdot 1.25 + \alpha \cdot 0.1]$$

$$= N \cdot P \cdot [1.25 - 1.15\alpha]$$

当 $\alpha > \frac{0.25}{1.15} \approx 21.7\%$ 时，cache 开始比不 cache 便宜。

## 7. 设计启示

### 7.1 Prompt 结构设计原则

基于以上分析，最大化 cache 效益的原则是：

```
[System Prompt - 静态，数千 token] ← 放最前面，最大化匹配长度
[Few-shot Examples - 相对稳定]     ← 次前面
[工具定义 - 相对稳定]              ← 次前面
[对话历史 - 增量变化]              ← 中间
[当前用户消息 - 每次不同]          ← 放最后面
```

### 7.2 Block Size 的选择

block size 影响两个方面：

1. **匹配精度**：越小越精确，越大浪费越多
2. **管理开销**：越小 hash table 越大，内存和查找开销越高

在实践中，16 是一个经过验证的平衡点。vLLM 默认 `block_size=16`，SGLang 也使用类似的粒度。

### 7.3 多租户场景下的 Cache 共享

API 提供商需要决定 cache 的共享范围：

- **同一请求内的多轮对话**：天然共享（同一前缀）
- **同一用户的不同请求**：如果使用相同的 system prompt，可以共享
- **不同用户**：如果使用相同的 system prompt + tools，理论上可以共享
- **安全边界**：Anthropic 限制在同一 workspace 内共享，OpenAI 限制在同一 organization 内

跨用户共享带来了隐私和安全的考量——如果不同用户的 system prompt 相同，共享 KV cache 不会泄露信息（因为 cache 的内容就是这些公共 prompt 的 KV 值）。但实现上需要确保不同用户的请求被路由到同一个 cache 分区。

---

**下一节：** [vLLM APC 源码分析](02-vllm-apc.md) — 深入 vLLM 的 Automatic Prefix Caching 实现，理解 hash 计算、block 匹配和回收机制。
