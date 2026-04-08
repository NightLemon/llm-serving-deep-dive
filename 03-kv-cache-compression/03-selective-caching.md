# 选择性缓存与上下文压缩

> 不是所有 token 的 KV 都同等重要——通过识别关键 token，用固定大小的 KV Cache 实现近似无损的推理。

## 1. 核心思路

### 1.1 全量缓存的浪费

在标准的自回归解码中，我们为序列中的每一个 token 都保留完整的 KV Cache。但大量研究表明，attention 分数的分布是**高度稀疏**的：

```
典型的 attention 分布（某一层某一个 head）：
Token position:  0    1    2    3    4    ...  995  996  997  998  999
Attention score: 0.15 0.01 0.00 0.00 0.02 ... 0.01 0.03 0.08 0.25 0.35
                 ↑                                            ↑    ↑
            attention sink                              最近的 tokens
                                                       获得最多注意力
```

通常超过 90% 的 attention weight 集中在不到 5% 的 token 上。这意味着大部分 token 的 KV Cache 对输出的贡献微乎其微——它们占用了显存但几乎不影响结果。

### 1.2 选择性缓存的基本范式

```
全量缓存：  [KV_0, KV_1, KV_2, ..., KV_t-2, KV_t-1, KV_t]
                                                    总共 t+1 个 KV

选择性缓存：[KV_0, KV_5, KV_23, ..., KV_t-2, KV_t-1, KV_t]
                ↑      ↑     ↑           ↑ 最近的 window
            重要 token（heavy hitters）
                                     总共 k 个 KV (k << t)
```

核心问题变为：**如何高效地识别"重要" token？**

## 2. H2O: Heavy-Hitter Oracle

### 2.1 论文概述

[H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048)（2023）首次系统性地提出了基于 attention 分数的 KV Cache 驱逐策略。

**核心发现**：在 LLM 的 attention 中，存在少量 **heavy-hitter token**——它们在大部分 attention head 中都获得较高的注意力分数。这些 token 承载了关键信息，丢弃它们会严重影响生成质量。

### 2.2 Heavy-Hitter 的识别

H2O 使用**累积 attention 分数**来识别 heavy-hitter：

```python
# H2O 的核心逻辑
class H2OCache:
    def __init__(self, budget_size, recent_window):
        """
        budget_size: KV Cache 的总预算（固定大小）
        recent_window: 最近 token 的保留窗口
        """
        self.budget = budget_size
        self.recent = recent_window
        self.heavy_budget = budget_size - recent_window
        self.accumulated_scores = {}  # token_idx → 累积 attention score
    
    def update_and_evict(self, attention_scores, new_token_idx):
        """
        attention_scores: [seq_len] 当前步骤的 attention 分数
        """
        # 1. 累积 attention 分数
        for idx, score in enumerate(attention_scores):
            self.accumulated_scores[idx] = (
                self.accumulated_scores.get(idx, 0) + score
            )
        
        # 2. 如果超出预算，驱逐 attention 最低的 token
        if len(self.accumulated_scores) > self.budget:
            # 保护最近的 token 不被驱逐
            eviction_candidates = {
                idx: score 
                for idx, score in self.accumulated_scores.items()
                if idx < new_token_idx - self.recent
            }
            
            # 驱逐累积分数最低的 token
            num_to_evict = len(self.accumulated_scores) - self.budget
            evict_indices = sorted(
                eviction_candidates, 
                key=eviction_candidates.get
            )[:num_to_evict]
            
            for idx in evict_indices:
                del self.accumulated_scores[idx]
                # 同时从 KV Cache 中移除对应的 KV 向量
                evict_from_kv_cache(idx)
```

### 2.3 Cache 组成

H2O 维护的 KV Cache 由两部分组成：

```
H2O Cache = [Heavy-Hitter Tokens] + [Recent Window]
             ↑                        ↑
        累积 attention 最高的         最近 w 个 token
        h 个 token (动态更新)         (滑动窗口)
        
总预算 B = h + w

示例（B=256, w=128）：
  [HH_0, HH_1, ..., HH_127] [Recent_t-127, Recent_t-126, ..., Recent_t]
   128 个 heavy-hitters        128 个最近 token
```

### 2.4 驱逐策略的变体

H2O 论文中探讨了多种驱逐策略：

| 策略 | 描述 | 效果 |
|------|------|------|
| Local (recent only) | 只保留最近 B 个 token | 丢失全局信息 |
| Random eviction | 随机驱逐 | 基线，效果差 |
| H2O (greedy) | 每步驱逐累积分数最低的 | 最佳效果 |
| H2O (lazy) | 每 k 步批量驱逐 | 计算开销更低，效果接近 greedy |

### 2.5 实验结果

H2O 在仅使用 20% 的 KV Cache 时，能保持与全量 cache 接近的生成质量：

```
模型: OPT-6.7B / LLaMA-7B
任务: 文本生成、摘要、问答

                     全量 Cache   H2O (20%)   Local (20%)   Random (20%)
WikiText-2 PPL:       10.86        11.02        12.45          15.23
XSUM ROUGE-L:         22.1         21.8         18.3           16.5
```

关键发现：
- 20% 的预算下，H2O 的 PPL 增加不到 2%
- 纯 Local（只保留最近 token）在需要全局信息的任务上严重退化
- Heavy-hitter 的识别在不同 decoding step 之间相对稳定

## 3. StreamingLLM / Attention Sink

### 3.1 Attention Sink 效应

[StreamingLLM](https://arxiv.org/abs/2309.17453)（2023）发现了一个令人意外的现象：**序列最开头的几个 token 总是获得异常高的 attention 分数，即使它们的语义内容并不重要。**

```
一个典型的 attention 分布：
Position:  [BOS] [The] [cat] [sat] [on]  ... [the] [mat] [.]
Score:      0.20  0.08  0.01  0.01  0.01 ...  0.05  0.15  0.25
            ↑                                             ↑
         Attention Sink                              Recent tokens
         (与语义无关的高分)                        (真正的上下文依赖)
```

### 3.2 为什么存在 Attention Sink？

这不是因为初始 token 语义重要，而是 **softmax 的数学性质**导致的：

```
Softmax 要求所有 attention weight 加和为 1：
  Σ_i softmax(score_i) = 1

当模型"不需要关注任何特定 token"时（即所有 token 都不太相关），
它需要一个"垃圾桶"来放置多余的 attention weight。

初始 token（尤其是 BOS）因为在所有训练样本中都出现，
被模型学习为默认的 attention sink——一个安全的"倾倒"位置。

如果移除初始 token 的 KV：
  - softmax 被迫将多余的 weight 分配给其他 token
  - 导致 attention 分布严重扭曲
  - 生成质量急剧下降
```

### 3.3 StreamingLLM 的方案

基于 attention sink 的发现，StreamingLLM 提出了一个极简方案：

```
StreamingLLM Cache = [Sink Tokens] + [Sliding Window]
                      前 s 个 token    最近 w 个 token

典型配置：s=4, w=1020 → 总预算 1024 tokens

示例：
  [BOS] [token_1] [token_2] [token_3] ... [token_t-1019] ... [token_t]
  ←--- sink tokens (4个) ---→          ←--- sliding window (1020个) ---→
  
  中间的 token 全部丢弃！
```

### 3.4 实现细节

```python
class StreamingLLMCache:
    def __init__(self, sink_size=4, window_size=1020):
        self.sink_size = sink_size
        self.window_size = window_size
        self.total_budget = sink_size + window_size
        
        self.sink_kv = None    # [sink_size, num_heads, head_dim]
        self.window_kv = None  # [window_size, num_heads, head_dim]
    
    def update(self, new_key, new_value, step):
        if step < self.sink_size:
            # 前 s 步：填充 sink cache
            self.sink_kv[step] = (new_key, new_value)
        else:
            # 之后：滑动窗口，FIFO 策略
            self.window_kv = roll_and_append(
                self.window_kv, (new_key, new_value)
            )
    
    def get_kv_cache(self):
        # 返回 [sink_tokens | window_tokens] 的 KV
        return concat(self.sink_kv, self.window_kv)
```

### 3.5 位置编码的处理

StreamingLLM 丢弃了中间 token，但保留了 sink token 的原始位置编码。这对使用 RoPE 的模型会产生问题：

```
原始序列位置：  [0, 1, 2, 3, ..., 500, 501, ..., 1023]
丢弃后的位置：  [0, 1, 2, 3,     504, 505, ..., 1023]
                sink tokens      window tokens
                
位置 gap = 500 → RoPE 会认为 sink 和 window 之间距离很远
```

StreamingLLM 的解决方案：**重编位置**

```
重编后的位置：  [0, 1, 2, 3, 4, 5, ..., 1023]
                sink tokens  window tokens (位置紧凑排列)
```

这样做虽然改变了绝对位置关系，但由于 RoPE 主要编码相对位置，且 sink token 的主要作用是作为 attention 的"垃圾桶"而非语义信息源，实际效果良好。

### 3.6 局限性

StreamingLLM 的局限性非常明显：

1. **中间上下文完全丢失**：
   ```
   用户输入一个 5000 token 的文档，然后问关于文档中间部分的问题
   → StreamingLLM 可能已经丢弃了相关的 KV Cache
   → 无法回答
   ```

2. **不适合 RAG 和长文档理解**：任何需要全局信息的任务都会退化

3. **与 prefix caching 冲突**：
   ```
   Prefix caching 假设完整保留 prefix 的 KV
   StreamingLLM 丢弃中间 prefix 的 KV
   → 两者的 cache 假设矛盾
   ```

4. **仅适合流式对话**：对于连续对话（chatbot），用户通常只关心最近的上下文，StreamingLLM 可以工作。但对于需要引用早期上下文的场景则不行。

## 4. SnapKV：基于观察窗口的智能压缩

### 4.1 核心思想

[SnapKV](https://arxiv.org/abs/2404.14469)（2024）提出了一种更精细的选择性缓存方案。核心洞察：**可以用序列末尾的一个"观察窗口"中的 attention pattern 来识别整个序列中的重要 token。**

```
输入序列：[t_0, t_1, t_2, ..., t_n-w, ..., t_n]
                                  ↑
                          观察窗口 (最后 w 个 token)

Step 1: 正常计算观察窗口的 attention
Step 2: 分析观察窗口中每个 token 对前文的 attention pattern
Step 3: 基于 attention pattern 选择要保留的 prefix token
Step 4: 只保留选中 token 的 KV Cache
```

### 4.2 选择算法

```python
class SnapKVSelector:
    def __init__(self, budget, observation_window=64, kernel_size=5):
        self.budget = budget
        self.obs_window = observation_window
        self.kernel_size = kernel_size  # pooling kernel for smoothing
    
    def select_important_tokens(self, attention_weights):
        """
        attention_weights: [n_heads, obs_window, seq_len]
        观察窗口中每个 token 对全序列的 attention 分布
        """
        # 1. 对观察窗口内的 attention 求和/平均
        #    得到每个位置的"重要性分数"
        importance = attention_weights.sum(dim=1)  # [n_heads, seq_len]
        
        # 2. 使用 1D average pooling 平滑（聚集相邻 token 的重要性）
        importance_smoothed = avg_pool1d(
            importance, kernel_size=self.kernel_size
        )
        
        # 3. 跨 head 聚合（投票机制）
        importance_aggregated = importance_smoothed.mean(dim=0)  # [seq_len]
        
        # 4. 选择 top-k 重要 token
        prefix_len = seq_len - self.obs_window
        top_k = min(self.budget - self.obs_window, prefix_len)
        selected_indices = importance_aggregated[:prefix_len].topk(top_k).indices
        
        # 5. 保留观察窗口 + 选中的 prefix tokens
        keep_indices = sorted(selected_indices.tolist()) + list(
            range(prefix_len, seq_len)
        )
        
        return keep_indices
```

### 4.3 SnapKV vs H2O

| 维度 | H2O | SnapKV |
|------|-----|--------|
| 选择时机 | 每个 decoding step 动态更新 | Prefill 阶段一次性选择 |
| 计算开销 | 每步都需要更新累积分数 | 只在 prefill 末尾计算一次 |
| 选择粒度 | Token 级别 | Token 级别 + 局部平滑 |
| 适用场景 | 长序列 decode | 长 prompt 场景（如 RAG） |
| 与 PagedAttention 兼容性 | 需要修改驱逐逻辑 | Prefill 后直接减少 cache 大小 |

### 4.4 效果

SnapKV 在长上下文任务中表现出色：

```
LongBench 评测（LLaMA-2-7B-32K）：

                  全量 Cache   SnapKV (1024)   SnapKV (2048)   H2O (1024)
Single-Doc QA:      45.2         43.8            44.9           40.1
Multi-Doc QA:       38.6         37.1            38.2           33.5
Summarization:      26.8         26.2            26.7           24.3
Few-shot:           67.3         66.1            67.0           62.8
```

## 5. PyramidInfer：层级递减的 KV Cache

### 5.1 核心观察

[PyramidInfer](https://arxiv.org/abs/2405.12532)（2024）基于一个重要观察：**不同层的 attention 稀疏度不同，浅层通常更稀疏**。

```
Attention 稀疏度随层数变化：
Layer 0:  ████░░░░░░  ~40% 的 token 获得显著 attention
Layer 8:  ███░░░░░░░  ~30%
Layer 16: ██░░░░░░░░  ~20%
Layer 24: ██░░░░░░░░  ~15%
Layer 31: █░░░░░░░░░  ~10%

→ 浅层可以保留更多 KV，深层只需保留少量 KV
→ 形成"金字塔"结构
```

### 5.2 层级预算分配

```python
class PyramidCache:
    def __init__(self, n_layers, total_budget, strategy="linear"):
        """
        按层分配不同的 KV Cache 预算
        """
        if strategy == "linear":
            # 线性递减：浅层预算大，深层预算小
            weights = [n_layers - i for i in range(n_layers)]
            total_weight = sum(weights)
            self.budgets = [
                int(total_budget * w / total_weight) 
                for w in weights
            ]
        elif strategy == "exponential":
            # 指数递减
            weights = [2 ** (n_layers - i - 1) for i in range(n_layers)]
            total_weight = sum(weights)
            self.budgets = [
                int(total_budget * w / total_weight) 
                for w in weights
            ]
```

```
示例（32 层模型，总预算 8192 tokens）：
Layer  0: budget = 512 tokens  ████████████████
Layer  8: budget = 384 tokens  ████████████
Layer 16: budget = 256 tokens  ████████
Layer 24: budget = 192 tokens  ██████
Layer 31: budget = 128 tokens  ████

→ 总 KV Cache = Σ budget_i × kv_size_per_token
  比均匀分配（每层 256）更有效利用显存
```

### 5.3 层间信息传递

PyramidInfer 的另一个创新是**利用上一层的 attention 信息指导下一层的 token 选择**：

```
Layer L 的 attention pattern → 识别 Layer L 的 important tokens
                             → 用于指导 Layer L+1 的 KV 保留策略

这避免了每层独立计算重要性分数的开销
```

## 6. 选择性缓存 vs 全量缓存：适用场景

### 6.1 决策框架

```
                        上下文长度
                    短 (<4K)        长 (>32K)
                 ┌───────────┬──────────────┐
需要全局信息？    │            │              │
    是           │ 全量 Cache  │  SnapKV /    │
                 │            │  PyramidInfer │
    否           │ 全量 Cache  │  StreamingLLM│
    (仅需最近)   │ (开销可控)  │  / H2O       │
                 └───────────┴──────────────┘
```

### 6.2 各方案对比总结

| 方案 | KV Cache 大小 | 全局信息保留 | 实现复杂度 | 延迟开销 | 最佳场景 |
|------|-------------|------------|----------|---------|---------|
| Full Cache | O(n) | 完整 | 最低 | 无 | 短序列 |
| StreamingLLM | O(1) | 极少 | 低 | 极低 | 流式对话 |
| H2O | O(1) | 部分 | 中 | 每步更新 | 长序列 decode |
| SnapKV | O(1) | 较好 | 中 | prefill 一次 | 长 prompt |
| PyramidInfer | O(1) per layer | 较好 | 高 | prefill 一次 | 极长序列 |

### 6.3 与量化的组合

选择性缓存可以与量化方案叠加：

```
组合 1: SnapKV + FP8 量化
  → 选择重要 token (减少 token 数) + 量化 (减少每个 token 的 bytes)
  → 压缩效果相乘：如 4x (选择) × 2x (FP8) = 8x 总压缩

组合 2: H2O + KIVI (2-bit)
  → 固定预算 + 极端量化
  → 适合显存极度受限的场景

组合 3: StreamingLLM + MLA
  → sink + window 的 KV 用 MLA latent 存储
  → 超长流式推理的极致方案
```

## 7. 与 Prefix Caching 的兼容性问题

### 7.1 冲突根源

Prefix Caching 和选择性缓存在根本假设上存在矛盾：

```
Prefix Caching 假设：
  相同的 prefix → 相同的 KV Cache → 可以复用
  
选择性缓存假设：
  不同的 query 对 prefix 中 token 的"重要性"判断不同
  → 同一个 prefix 可能产生不同的 cache 子集
  → 无法简单复用
```

### 7.2 具体示例

```
Prompt A: "阅读以下文章：[长文档]。请总结第一段的主要内容。"
Prompt B: "阅读以下文章：[长文档]。请分析最后一段的论点。"

两个 prompt 共享同一个 prefix [长文档]，
但 SnapKV 会基于不同的 query 选择不同的 important tokens：
  A 选择第一段相关的 token
  B 选择最后一段相关的 token

→ Prefix Cache 无法命中（cache 内容不同）
```

### 7.3 可能的解决方案

1. **保守选择策略**：选择 token 时使用更大的预算，确保覆盖所有可能的 query 需求
   - 降低了压缩效果，但保持了 prefix cache 兼容性

2. **两级缓存**：
   ```
   Level 1: Full prefix KV Cache (共享，用于 prefix caching)
   Level 2: Selected KV subset (per-query，用于快速 decode)
   
   Prefill 时先查 Level 1 (prefix cache hit)
   然后计算 query-specific 的 Level 2 subset
   ```

3. **Query-independent 选择**：使用不依赖 query 的重要性指标（如 entropy-based），使得相同 prefix 总是产生相同的 cache subset
   ```
   重要性 = f(attention_entropy, token_frequency, position)
   → 确定性函数，相同 prefix → 相同选择 → 可以复用
   ```

## 8. 前沿发展

### 8.1 CacheBlend (2024)

CacheBlend 提出了一种在 prefix caching 和选择性缓存之间取得平衡的方案：

- 使用 prefix cache 作为"粗粒度"初始化
- 对关键层和关键 token 进行"精细化"重计算
- 实现了 prefix cache 的复用性和选择性缓存的精度

### 8.2 Dynamic Token Pruning

更新的研究方向是**动态修剪**：根据生成过程中 attention 的变化，实时调整保留的 token 集合：

```
Step t:   保留 [t_0, t_5, t_12, t_45, ..., window]
Step t+1: 保留 [t_0, t_5, t_12, t_67, ..., window]  ← t_45 被替换为 t_67
                                ↑ 动态调整

相比 SnapKV（一次性选择后固定），dynamic pruning 可以适应上下文的变化
但计算开销更大，工程实现更复杂
```

### 8.3 与 Speculative Decoding 的交互

选择性缓存与 speculative decoding 的交互也是一个活跃的研究方向：

```
Speculative Decoding:
  Draft model 生成 k 个候选 token
  Target model 验证这 k 个 token

如果 target model 使用选择性缓存：
  - Draft model 的 KV Cache 可以使用更激进的压缩（它本身就是近似的）
  - Target model 验证时需要足够精确的 KV Cache
  - 两者的 cache 管理策略需要协调
```

## 9. 工程实践建议

### 9.1 选择方案的决策树

```
应用场景是什么？
├── 流式对话 / chatbot
│   └── StreamingLLM (sink=4, window=1020)
│       简单高效，适合无限长对话
│
├── 长文档理解 / RAG
│   ├── 文档长度 < 模型 context length
│   │   └── 全量 Cache（可配合 FP8 量化）
│   └── 文档长度 >> context length
│       └── SnapKV / PyramidInfer
│           在 prefill 阶段压缩，decode 阶段用固定预算
│
├── 批量推理 / 高吞吐
│   └── H2O (固定预算)
│       减少 KV Cache → 增大 batch size → 提高吞吐
│
└── 超长上下文 (128K+)
    └── PyramidInfer + KIVI 量化
        层级预算 + 极端量化 = 最大压缩
```

### 9.2 监控与调试

部署选择性缓存后，需要监控以下指标：

```python
# 关键监控指标
metrics = {
    "cache_hit_rate": "prefix cache 命中率（是否因选择性缓存下降？）",
    "eviction_rate": "每步驱逐的 token 数",
    "important_token_overlap": "连续步骤之间 heavy-hitter 集合的重叠率",
    "generation_quality": "ROUGE / BLEU / 人工评估分数",
    "memory_usage": "实际 KV Cache 显存占用",
    "attention_entropy": "attention 分布的熵（越低越稀疏，越适合选择性缓存）",
}
```

---

> **下一节**：[GQA/MQA 深度分析](04-gqa-mqa.md) — 从 attention head 数量的角度压缩 KV Cache。
