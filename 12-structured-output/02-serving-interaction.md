# 与 Serving 系统的交互

> Constrained Decoding 不是一个独立的模块——它嵌入到推理服务的 decode 循环中，与调度器、batch 管理、KV cache、投机解码等组件产生深刻的交互。理解这些交互是在生产环境中部署结构化输出的前提。

## 1. 对吞吐量的影响

### 1.1 Logit Masking 的计算开销

每个 decode step 增加的操作：

```
1. 查询当前 DFA/PDA 状态 → O(1)
2. 获取合法 token 集合 → O(1)（查预计算 Index 表）
3. 构建 mask 向量 → O(|V|)
4. 将 mask 加到 logits 上 → O(|V|)
```

其中 $|V|$ 是词表大小（32K~128K）。相比模型的 forward pass（涉及数十亿参数的矩阵运算），mask 操作的计算量微乎其微。瓶颈不在计算，而在两方面：

**内存开销**：预计算的 Index 表需要存储在内存中。

$$
\text{Index 内存} = |S| \times |V| \times \text{sizeof(bool)} = |S| \times |V| \text{ bits}
$$

对于一个 1000 状态的 DFA 和 128K 词表，Index 表约为 $1000 \times 128K / 8 \approx 16 \text{ MB}$。使用稀疏表示可以大幅压缩。

**首 token 延迟**：如果 schema 未被缓存，首次请求需要编译 JSON Schema → Regex → DFA → Index，这个过程可能耗时几十到几百毫秒。

### 1.2 Benchmark 数据

以下是典型场景下 constrained decoding 的吞吐量影响（基于 vLLM + Outlines 后端的经验数据）：

| Schema 复杂度 | DFA 状态数 | 吞吐量下降 | 说明 |
|--------------|-----------|-----------|------|
| 简单（3-5 字段，扁平） | ~50 | 5-8% | 大部分开销来自 mask 构建 |
| 中等（10-20 字段，1 层嵌套） | ~200 | 8-12% | Index 表查找仍在 L2 cache 内 |
| 复杂（嵌套对象 + 数组 + 枚举） | ~1000+ | 12-20% | Index 表可能溢出 cache |
| CFG（任意嵌套） | 动态 | 15-30% | PDA 栈操作增加开销 |

**关键发现**：对于大多数生产场景（扁平或浅层嵌套的 JSON Schema），吞吐量下降控制在 **10% 以内**——这是完全可以接受的代价，因为它消除了 retry 和解析失败带来的隐性成本。

## 2. 与 Continuous Batching 的交互

> 参考 [Ch08: 调度与批处理](../08-scheduling-batching/README.md) 了解 continuous batching 的基本原理。

### 2.1 Per-Request FSM State

Continuous batching 的核心优势是在每个 iteration 灵活地增删 batch 中的请求。引入 constrained decoding 后，每个请求需要额外维护：

- **当前 DFA/PDA 状态**：标记解码进度在语法中的位置
- **Index 表引用**：指向该 schema 对应的预计算 mask

```
Batch at iteration t:
┌─────────┬────────────┬───────────┬─────────────┐
│ Request │ KV Cache   │ DFA State │ Schema Ref  │
├─────────┼────────────┼───────────┼─────────────┤
│ Req1    │ blocks_1   │ s=12      │ schema_A    │
│ Req2    │ blocks_2   │ s=5       │ schema_B    │
│ Req3    │ blocks_3   │ s=23      │ schema_A    │
│ Req4    │ blocks_4   │ (none)    │ (free gen)  │
└─────────┴────────────┴───────────┴─────────────┘
```

### 2.2 无法跨请求共享 FSM 状态

与 prefix caching（[Ch02](../02-prefix-caching/README.md)）不同，FSM 状态**无法在不同请求间共享**——即使两个请求使用同一个 schema，它们的 DFA 状态也会随着不同的生成内容而分化。

但 **Index 表**（DFA 状态 → 合法 token 集合的映射）是可以共享的：同一个 schema 编译出的 Index 表对所有使用该 schema 的请求都是相同的。

```
共享层级:
Schema A ─→ DFA_A ─→ Index_A  ← Req1, Req3 共享
Schema B ─→ DFA_B ─→ Index_B  ← Req2 独占
```

### 2.3 对 Batch 效率的影响

Continuous batching 中，不同请求可能处于不同的 DFA 状态，因此需要不同的 logit mask。这在 GPU 上的实现方式通常是：

1. **逐请求构建 mask**：为 batch 中每个请求独立查表、构建 mask
2. **批量应用**：将所有 mask 组成一个矩阵，与 logits 矩阵做 element-wise 加法

由于 mask 构建的计算量远小于模型 forward pass，对 batch 效率的影响是**最小的**——前提是 Index 表已经预计算完成。

!!! warning "延迟风险"
    如果 batch 中混入了一个使用**未缓存 schema** 的请求，该请求的 DFA 编译可能阻塞整个 batch 的 decode step（除非编译在异步线程中进行）。生产环境应**预编译常用 schema** 或使用异步编译。

## 3. 与 Speculative Decoding 的交互

> 参考 [Ch07: 投机解码](../07-speculative-decoding/README.md) 了解 speculative decoding 的数学基础。

这是 constrained decoding 与 serving 系统交互中**最复杂**的部分。

### 3.1 核心冲突

Speculative decoding 的流程是：

1. Draft model 生成 $\gamma$ 个 candidate tokens
2. Target model 并行验证

问题在于：**Draft model 通常不知道 grammar constraints。** Draft model 可能生成违反 schema 的 token 序列，导致验证阶段大量 rejection。

```
Schema 要求: {"name": "...", "age": <integer>}
当前位置: 刚输出 "age": 

Draft model 生成: "twenty-five"  (文本而非数字)
Target model 验证: 第一个 token "twenty" 就违反约束 → reject
```

### 3.2 接受率下降

在无约束场景下，speculative decoding 的期望接受长度为：

$$
E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

其中 $\alpha$ 是 draft 与 target 分布的匹配程度。加入 constrained decoding 后，有效的 $\alpha$ 降低——因为 draft model 可能将概率分配给不合法的 token，而这些概率在 target 端被强制归零。

**经验数据**：在典型 JSON 生成场景下，constrained decoding 导致 speculative decoding 的接受率下降 **20-40%**。对于约束密集的区域（如 JSON key 名称，基本被完全决定），接受率可能降至接近 0。

### 3.3 解决方案

**方案 A：在 Draft Model 上也施加约束**

```
Draft: 生成 token 时同样查 DFA → 只生成合法 token
Target: 验证时再次检查 DFA（双重保证）
```

- **优点**：接受率恢复到接近无约束的水平
- **缺点**：Draft model 速度变慢（需要在每步做 logit masking），部分抵消了 speculative decoding 的收益

**方案 B：Verification 阶段过滤**

```
Draft: 自由生成 γ 个 token（不施加约束）
Target: 验证时同时检查 DFA 状态，将违反约束的 token 视为 rejection
```

- **优点**：Draft model 速度不受影响
- **缺点**：接受率大幅降低，在约束密集区域几乎退化为逐 token 生成

**方案 C：混合策略（推荐）**

对 draft model 施加**轻量级约束**（只检查明显不合法的 token，如在需要数字的位置禁止字母），让详细的 grammar 检查留给 target model。

$$
\text{Draft mask} = \text{简化版 schema 约束（coarse）}
$$
$$
\text{Target mask} = \text{完整 schema 约束（fine-grained）}
$$

这在接受率和 draft 速度之间取得平衡。

### 3.4 特殊情况：Forced Tokens

当 DFA 状态只有一个合法的 next token 时（例如 JSON key 的固定字符 `"`），draft model 和 target model 都必须生成这个 token——此时 speculative decoding 的接受率为 100%。这些 **forced token** 区域反而是 speculative decoding 效果最好的地方。

## 4. 与 KV Cache 的交互

### 4.1 基本兼容性

Constrained decoding **不改变 KV cache 的行为**——它只修改 logits（模型输出的最后一层），不影响模型内部的 attention 计算和 KV cache 的读写。

```
Forward pass → hidden states → logits
                                  ↓
                           logit masking ← DFA state
                                  ↓
                              sampling
                                  ↓
                           token → KV cache update（正常流程）
```

### 4.2 Jump-Forward 优化

这是 constrained decoding 对 KV cache 层面的一个**正面影响**。

当 DFA 状态只有一个合法的 continuation 时（即当前位置的 token 是完全确定的），我们甚至不需要做 forward pass——直接输出该 token 即可。如果连续多个 token 都被确定（例如 JSON key `"name":`），可以一次性 "跳过" 这些 token。

```
Schema: {"name": "...", "age": ...}
当前已输出: {

接下来的 token 序列完全确定: "name":  → 可以直接 jump forward
```

**SGLang 的 jump-forward 优化**：

1. 检测当前 DFA 状态是否有唯一后继
2. 如果是，收集连续的确定性 token
3. 将这些 token 作为一个 "mini prefill" 批量处理
4. 跳过逐个 decode 的过程，显著减少 decode steps

$$
\text{节省的 decode steps} = \sum_{i} \mathbb{1}[|\mathcal{V}_{valid}(s_i)| = 1]
$$

在典型的 JSON 生成场景中，key 名称和标点符号（`{`, `}`, `:`, `,`, `"`）通常是完全确定的。根据 schema 的字段数量，这可以节省 **20-40% 的 decode steps**。

### 4.3 与 Prefix Caching 的交互

> 参考 [Ch02: 前缀缓存](../02-prefix-caching/README.md) 了解 prefix caching 的基本原理。

当使用 prefix caching 时，schema 信息通常包含在 system prompt 中：

```
System: 你是一个信息提取助手。请以如下 JSON 格式输出：
{"name": string, "age": integer, "skills": string[]}

User: 从以下文本中提取用户信息：...
```

**问题**：不同的 schema 导致不同的 system prompt → 不同的 prefix → 无法共享 prefix cache。

**解决方案**：将 schema 信息与 system prompt 解耦。

```
方案 1: Schema 放在 user message 末尾（而非 system prompt）
  System: "你是一个信息提取助手。"  ← 所有 schema 共享
  User: "从以下文本中提取用户信息：... [schema 附在此处]"

方案 2: 使用 serving 层面的 guided decoding（schema 不进 prompt）
  prompt 只包含自然语言指令
  schema 通过 API 参数（response_format）传递给 serving 引擎
  → prefix cache 完全不受影响
```

方案 2 是生产环境的推荐做法——**schema 走 API 参数，不污染 prompt**。这样同一类任务的不同 schema 可以共享 prefix cache，最大化缓存命中率。

## 5. 小结

| Serving 组件 | 交互关系 | 影响程度 | 建议 |
|-------------|---------|---------|------|
| Continuous Batching | 每请求独立 FSM 状态 | 低 | 预编译常用 schema |
| Speculative Decoding | Draft 可能违反约束 | 高（接受率下降） | 混合约束策略 |
| KV Cache | 无直接影响 | 正面（jump-forward） | 利用确定性 token 加速 |
| Prefix Caching | Schema 影响 cache key | 中 | Schema 走 API 参数 |

核心结论：Constrained decoding 对 serving pipeline 的影响是**可控的**，但需要在架构设计时有意识地处理——特别是与 speculative decoding 的兼容性问题。下一节将讨论生产环境中的具体实践模式。
