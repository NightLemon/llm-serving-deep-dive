# GQA/MQA 深度分析

> 从 attention head 的数量出发，理解 MHA → MQA → GQA 的演化路径及其对推理性能的影响。

## 1. 从 MHA 到 MQA 到 GQA

### 1.1 Multi-Head Attention (MHA) 回顾

标准 MHA（Vaswani et al., 2017）为每个 attention head 维护独立的 Q、K、V 投影：

```
输入: x ∈ R^{d_model}

对于 head i (i = 0, 1, ..., n_h - 1):
  Q_i = x × W_Q^i    ∈ R^{d_h}
  K_i = x × W_K^i    ∈ R^{d_h}    ← 每个 head 独立的 K 投影
  V_i = x × W_V^i    ∈ R^{d_h}    ← 每个 head 独立的 V 投影

  head_i = softmax(Q_i × K_i^T / √d_h) × V_i

output = Concat(head_0, head_1, ..., head_{n_h-1}) × W_O
```

**KV Cache 大小**：每个 head 都有独立的 K 和 V，需要全部缓存。

```
KV Cache per token per layer = 2 × n_h × d_h × sizeof(dtype)
```

### 1.2 Multi-Query Attention (MQA)

[MQA](https://arxiv.org/abs/1911.02150)（Shazeer, 2019）的核心思想极其简单：**所有 Query heads 共享同一组 K 和 V。**

```
对于所有 head:
  Q_i = x × W_Q^i    ∈ R^{d_h}    ← 每个 head 独立的 Q
  K   = x × W_K      ∈ R^{d_h}    ← 所有 head 共享同一个 K
  V   = x × W_V      ∈ R^{d_h}    ← 所有 head 共享同一个 V

  head_i = softmax(Q_i × K^T / √d_h) × V
```

**KV Cache 大小**：只有 1 组 K 和 V。

```
KV Cache per token per layer = 2 × 1 × d_h × sizeof(dtype)
                              = 2 × d_h × sizeof(dtype)
```

**压缩比**：

```
MHA KV Cache / MQA KV Cache = (2 × n_h × d_h) / (2 × d_h) = n_h

以 n_h = 32 的模型为例：MQA 实现 32x KV Cache 压缩
以 n_h = 64 的模型为例：MQA 实现 64x KV Cache 压缩
```

**MQA 的问题**：共享一组 KV 对所有 head 来说过于激进，会导致模型质量下降，尤其在需要多样化注意力模式的任务上。

### 1.3 Grouped-Query Attention (GQA)

[GQA](https://arxiv.org/abs/2305.13245)（Ainslie et al., 2023）是 MHA 和 MQA 的折中：**将 Query heads 分组，每组共享一组 K 和 V。**

```
将 n_h 个 Query heads 分为 n_g 组（每组 n_h/n_g 个 Q heads）

对于组 g (g = 0, 1, ..., n_g - 1):
  对于组内的每个 head i:
    Q_i = x × W_Q^i      ∈ R^{d_h}    ← 独立的 Q
  K_g = x × W_K^g        ∈ R^{d_h}    ← 组内共享的 K
  V_g = x × W_V^g        ∈ R^{d_h}    ← 组内共享的 V

  head_i = softmax(Q_i × K_g^T / √d_h) × V_g
```

```
特殊情况：
  n_g = n_h  → GQA 退化为 MHA（每个 Q head 有自己的 KV）
  n_g = 1    → GQA 退化为 MQA（所有 Q heads 共享一组 KV）
  1 < n_g < n_h → GQA 的一般形式
```

## 2. KV Cache 大小计算与对比

### 2.1 统一公式

```
KV Cache per token per layer = 2 × n_kv_heads × d_h × sizeof(dtype)

其中 n_kv_heads:
  MHA: n_kv_heads = n_h        (e.g., 32)
  GQA: n_kv_heads = n_g        (e.g., 8)
  MQA: n_kv_heads = 1
```

### 2.2 具体模型对比

以下是主流模型的 KV Cache 大小对比（FP16，per token，所有层合计）：

| 模型 | Attention | n_h | n_kv | d_h | layers | KV/token/layer | KV/token (all layers) |
|------|-----------|-----|------|-----|--------|----------------|----------------------|
| GPT-3 175B | MHA | 96 | 96 | 128 | 96 | 49,152 B | 4,608 KB |
| LLaMA-2-70B | GQA-8 | 64 | 8 | 128 | 80 | 4,096 B | 320 KB |
| LLaMA-3-70B | GQA-8 | 64 | 8 | 128 | 80 | 4,096 B | 320 KB |
| LLaMA-3-8B | GQA-8 | 32 | 8 | 128 | 32 | 4,096 B | 128 KB |
| Qwen-2.5-72B | GQA-8 | 64 | 8 | 128 | 80 | 4,096 B | 320 KB |
| Falcon-40B | MQA | 64 | 1 | 64 | 60 | 256 B | 15 KB |
| DeepSeek-V3 | MLA | 128 | — | 128 | 61 | 1,152 B | 68.6 KB |

> 注：DeepSeek-V3 使用 MLA，KV Cache 维度是 d_c + d_rope = 512 + 64 = 576 元素，故 1,152 bytes。

### 2.3 KV Cache 总显存对比

假设 batch_size=64, seq_len=4096, FP16：

```
模型             KV/token (all layers)   总 KV Cache
─────────────────────────────────────────────────────
GPT-3 175B       4,608 KB               1,152 GB   ← 完全不可行（单机）
LLaMA-2-70B      320 KB                 80 GB      ← 一张 H100 刚好放下
LLaMA-3-8B       128 KB                 32 GB      ← 舒适
Falcon-40B       15 KB                  3.75 GB    ← 几乎可忽略
DeepSeek-V3      68.6 KB                17.2 GB    ← 非常舒适

计算公式：总 KV Cache = batch_size × seq_len × KV/token (all layers)
```

### 2.4 压缩比汇总

以相同 hidden_dim 的 MHA 为 baseline：

```
             n_kv_heads    压缩比 (vs MHA)
MHA (n_h=64):    64           1x
GQA-16:          16           4x
GQA-8:            8           8x
GQA-4:            4          16x
GQA-2:            2          32x
MQA:              1          64x
MLA:             ~4.5*        ~14x (对比同规模 MHA)

* MLA 的等效 KV heads 数取决于 d_c 和 d_h 的比值
```

## 3. GQA 分组数对推理性能的影响

### 3.1 不只是显存：带宽才是关键

在 decode 阶段（batch_size=1，每步生成 1 个 token），attention 的性能瓶颈不是计算量，而是**内存带宽**（memory bandwidth）：

```
Decode attention 的操作：
  Q: [1, n_h, d_h]           ← 只有 1 个 token 的 query
  K: [seq_len, n_kv, d_h]    ← 需要从 HBM 读取整个 KV Cache
  V: [seq_len, n_kv, d_h]    ← 需要从 HBM 读取整个 KV Cache

计算量: O(seq_len × n_h × d_h)              ← 少
内存读取量: O(seq_len × n_kv × d_h × 2)     ← 多（读 K 和 V）

Arithmetic Intensity = 计算量 / 内存读取量 ≈ n_h / (2 × n_kv)

MHA (n_h=n_kv): AI ≈ 0.5   ← 极度 memory-bound
GQA-8:          AI ≈ 4      ← 仍然 memory-bound，但好了 8x
MQA:            AI ≈ 32     ← 接近 compute-bound
```

### 3.2 吞吐量与延迟的影响

减少 n_kv_heads 对推理的影响有两个层面：

**1. 延迟降低（单请求）：**

```
Decode 延迟主要受 KV Cache 读取量控制：

读取量 = seq_len × n_kv × d_h × 2 × sizeof(dtype)

H100 HBM 带宽: ~3.35 TB/s

以 LLaMA-3-70B (GQA-8) 为例，seq_len=4096, 单层:
  读取量 = 4096 × 8 × 128 × 2 × 2 = 16 MB
  理论延迟 = 16 MB / 3.35 TB/s ≈ 4.8 μs/layer

如果是 MHA (n_kv=64):
  读取量 = 4096 × 64 × 128 × 2 × 2 = 128 MB
  理论延迟 = 128 MB / 3.35 TB/s ≈ 38 μs/layer

GQA-8 的单层 attention 延迟降低约 8x
```

**2. 吞吐提升（大 batch）：**

```
减少 KV Cache → 同等显存下可以放更大的 batch
→ batch 越大，GPU 利用率越高（从 memory-bound 走向 compute-bound）
→ 吞吐提升

示例（80GB H100，模型权重占 40GB，剩余 40GB 用于 KV Cache）：

                    KV/token (all layers)   最大 batch×seq   吞吐倍数
MHA (n_kv=64):      2,560 KB               ~16K tokens      1x
GQA-8 (n_kv=8):     320 KB                 ~128K tokens     ~8x
MQA (n_kv=1):       40 KB                  ~1M tokens       ~64x

吞吐倍数是理想情况，实际受 compute 限制会打折
```

### 3.3 分组数选择的权衡

```
n_g (groups)   KV Cache    模型质量    推理性能
───────────────────────────────────────────────
n_h (MHA)      最大         最好        最慢
n_h/2          1/2          几乎无损    2x 提速
n_h/4          1/4          轻微下降    4x 提速
n_h/8          1/8          可接受      8x 提速   ← 主流选择
n_h/16         1/16         有损        16x 提速
1 (MQA)        最小         明显下降    最快
```

**为什么 GQA-8 成为主流？**

1. 8x 的 KV Cache 压缩已经能够在实际部署中显著降低显存压力
2. 模型质量损失在 1% 以内（perplexity 和下游任务）
3. 进一步减少到 GQA-4 或 MQA 的质量收益递减，但质量损失加速
4. 8 这个数字也恰好适合 GPU 的 warp/wavefront 并行粒度

## 4. 主流模型的 Attention 方案选择

### 4.1 各系列模型的选择

| 模型系列 | 版本演进 | Attention 方案 | 备注 |
|---------|---------|---------------|------|
| GPT | GPT-3 | MHA | 2020 年的设计 |
| | GPT-4 | 未公开 (推测 GQA) | |
| LLaMA | LLaMA-1 | MHA | 2023 年初 |
| | LLaMA-2-7B | MHA | |
| | LLaMA-2-70B | GQA-8 | 仅 70B 使用 GQA |
| | LLaMA-3-all | GQA-8 | 所有尺寸统一用 GQA |
| Qwen | Qwen-1 | MHA | |
| | Qwen-2/2.5 | GQA | |
| Falcon | Falcon-7B | MHA | |
| | Falcon-40B | MQA | 较早采用 MQA |
| | Falcon-180B | GQA-8 | 转向 GQA |
| Mistral | Mistral-7B | GQA-8 | 从第一个版本就用 GQA |
| | Mixtral-8x7B | GQA-8 | |
| DeepSeek | DeepSeek-V1 | MHA | |
| | DeepSeek-V2 | MLA | 创新架构 |
| | DeepSeek-V3 | MLA | 延续 MLA |
| Gemma | Gemma-1/2 | MHA / MQA | 7B 用 MHA，2B 用 MQA |

### 4.2 趋势分析

```
2020-2022: MHA 主导（GPT-3, LLaMA-1, etc.）
2023 H1:   MQA 开始出现（Falcon-40B, PaLM）
2023 H2:   GQA 成为主流（LLaMA-2-70B, Mistral-7B）
2024:      GQA 统一 + MLA 创新（LLaMA-3, DeepSeek-V2/V3）
2025:      GQA 仍是默认选择，MLA 被更多模型采用

未来方向：
- GQA 将继续作为"安全的默认选择"
- MLA 在追求极致 KV 效率的大规模模型中扩展
- 可能出现新的 attention 变体（如结合 MLA 和 GQA 的思想）
```

## 5. 从 MQA 到 GQA 的训练方法变化

### 5.1 Uptrain：从 MHA 到 GQA 的转换

GQA 论文的一个重要贡献是提出了将已训练好的 MHA 模型转换为 GQA 模型的方法——**uptrain**：

```
Step 1: 将 MHA checkpoint 的 KV heads 合并为 GQA groups
  原始 MHA: n_h 个独立的 KV heads
  目标 GQA: n_g 个共享的 KV heads（n_g < n_h）
  
  合并方法：对同一组内的 KV heads 取平均
  W_K^g = mean(W_K^{i} for i in group_g)
  W_V^g = mean(W_V^{i} for i in group_g)

Step 2: 用少量数据继续训练（uptrain）
  通常只需要原始训练的 5-10% 数据量
  学习率设为较小值（原始的 1/10）
  
Step 3: 得到 GQA 模型
```

```python
# Uptrain 的 KV head 合并（概念代码）
def convert_mha_to_gqa(mha_model, num_groups):
    heads_per_group = mha_model.num_heads // num_groups
    
    for layer in mha_model.layers:
        # 原始 KV weights: [num_heads, head_dim, hidden_dim]
        k_weights = layer.k_proj.weight.view(
            mha_model.num_heads, head_dim, hidden_dim
        )
        v_weights = layer.v_proj.weight.view(
            mha_model.num_heads, head_dim, hidden_dim
        )
        
        # 对每组取平均
        new_k_weights = []
        new_v_weights = []
        for g in range(num_groups):
            start = g * heads_per_group
            end = (g + 1) * heads_per_group
            new_k_weights.append(k_weights[start:end].mean(dim=0))
            new_v_weights.append(v_weights[start:end].mean(dim=0))
        
        layer.k_proj.weight = nn.Parameter(
            torch.stack(new_k_weights).view(-1, hidden_dim)
        )
        layer.v_proj.weight = nn.Parameter(
            torch.stack(new_v_weights).view(-1, hidden_dim)
        )
    
    return mha_model  # 现在是 GQA 模型
```

### 5.2 直接训练 GQA

从 LLaMA-3 开始，主流做法是**从头训练时就使用 GQA**：

```
GQA 从头训练 vs Uptrain 对比：

直接训练 GQA:
  + 模型从一开始就学习适应共享 KV 的表示
  + 通常质量更好
  + 不需要 MHA 预训练阶段
  - 需要全量训练计算

Uptrain (MHA → GQA):
  + 可以复用已有的 MHA checkpoint
  + 只需少量计算（5-10% 原始训练量）
  - 最终质量略低于直接训练
  - 需要选择合并策略（mean, random selection, etc.）
```

### 5.3 MHA → MQA 的 Uptrain

将 MHA 转换为 MQA 更激进——所有 KV heads 合并为 1 个：

```
MHA → MQA: 所有 head 的 KV weight 取平均（或选择某个 head）
  W_K_mqa = mean(W_K^0, W_K^1, ..., W_K^{n_h-1})
  W_V_mqa = mean(W_V^0, W_V^1, ..., W_V^{n_h-1})

这种合并丢失了大量信息，通常需要更多 uptrain 数据
```

## 6. GQA 和 MQA 对推理吞吐量的实际影响

### 6.1 端到端 Benchmark

以下数据来自 GQA 论文和后续实验报告的综合：

```
模型: T5-XXL (11B), 4096 tokens, batch_size=1, 单 GPU

                Time to First Token    Tokens/sec (decode)
MHA:            45 ms                  32 tokens/s
GQA-8:          43 ms                  58 tokens/s     (+81%)
MQA:            42 ms                  71 tokens/s     (+122%)

注意：
- TTFT 差异不大（prefill 是 compute-bound，KV heads 数量影响小）
- Decode 速度差异显著（decode 是 memory-bound，KV 读取量直接影响速度）
```

### 6.2 大 Batch 场景

```
模型: LLaMA-2-70B equivalent, 2048 tokens, H100 80GB

                     最大 batch_size   吞吐 (tokens/s)
MHA (n_kv=64):       8                 256
GQA-8 (n_kv=8):     64                1,280           (+5x)
MQA (n_kv=1):        256              2,560           (+10x)

GQA-8 的吞吐接近 MQA 的一半——在大部分场景下这是一个很好的质量-性能平衡点。
```

### 6.3 Prefill vs Decode 的不同影响

```
Prefill 阶段（处理长 prompt）：
  - 操作是 compute-bound（大矩阵乘法）
  - KV heads 数量的影响：
    计算量减少（生成 KV 的矩阵乘法变小）
    但相对于 Q 的计算，KV 的计算占比小
    总体 prefill 加速 < 10%

Decode 阶段（逐 token 生成）：
  - 操作是 memory-bound（从 HBM 读 KV Cache）
  - KV heads 数量的影响：
    内存读取量线性减少
    Decode 延迟近似线性降低
    总体 decode 加速接近 n_h/n_kv 倍
```

### 6.4 模型质量的影响

GQA 论文中给出了详细的质量对比：

```
T5-XXL (11B) 在多个 NLP benchmark 上的表现：

                 SuperGLUE   SQuAD     CNN/DM
MHA (baseline):  90.7        93.2      21.0
GQA-8 (uptrain): 89.9        92.8      20.8     ← 损失 < 1%
GQA-4 (uptrain): 89.5        92.5      20.6
GQA-2 (uptrain): 88.8        91.9      20.3
MQA (uptrain):   88.2        91.3      20.0     ← 损失 ~2.5%

GQA-8 (从头训练): 90.3       93.0      20.9    ← 损失 < 0.5%
MQA (从头训练):   89.4        92.4      20.5    ← 损失 ~1.5%
```

关键结论：
1. GQA-8 的质量损失非常小（< 1%），无论是 uptrain 还是从头训练
2. 从头训练比 uptrain 效果更好
3. MQA 的质量损失更明显，但在某些延迟敏感场景仍然可接受

## 7. GQA 的实现细节

### 7.1 KV Head 广播

在 GQA 的 attention 计算中，需要将 n_g 个 KV heads "广播" 给 n_h 个 Q heads：

```python
# GQA attention 的核心实现
def gqa_attention(query, key, value, num_kv_heads, num_q_heads):
    """
    query: [batch, seq_q, num_q_heads, head_dim]
    key:   [batch, seq_kv, num_kv_heads, head_dim]
    value: [batch, seq_kv, num_kv_heads, head_dim]
    """
    heads_per_group = num_q_heads // num_kv_heads
    
    # 方法 1: 显式 repeat (简单但可能增加内存)
    key = key.repeat_interleave(heads_per_group, dim=2)
    value = value.repeat_interleave(heads_per_group, dim=2)
    # 现在 key, value shape: [batch, seq_kv, num_q_heads, head_dim]
    
    # 标准 attention 计算
    scores = einsum('bqhd,bkhd->bhqk', query, key) / sqrt(head_dim)
    attn = softmax(scores, dim=-1)
    output = einsum('bhqk,bkhd->bqhd', attn, value)
    
    return output

    # 方法 2: reshape + broadcast (更高效)
    query = query.view(batch, seq_q, num_kv_heads, heads_per_group, head_dim)
    key = key.unsqueeze(3)    # [batch, seq_kv, num_kv_heads, 1, head_dim]
    value = value.unsqueeze(3)
    
    scores = einsum('bqghd,bkghd->bghqk', query, key) / sqrt(head_dim)
    attn = softmax(scores, dim=-1)
    output = einsum('bghqk,bkghd->bqghd', attn, value)
    output = output.view(batch, seq_q, num_q_heads, head_dim)
    
    return output
```

### 7.2 FlashAttention 中的 GQA 支持

FlashAttention-2 及以后版本原生支持 GQA，无需显式广播：

```python
# FlashAttention GQA 调用
from flash_attn import flash_attn_func

# FlashAttention 自动处理 GQA 的 head 数量不匹配
output = flash_attn_func(
    q,   # [batch, seq_q, num_q_heads, head_dim]
    k,   # [batch, seq_kv, num_kv_heads, head_dim]  ← num_kv_heads < num_q_heads
    v,   # [batch, seq_kv, num_kv_heads, head_dim]
    causal=True
)
# FlashAttention 内部处理 head 广播，不需要显式 repeat
```

### 7.3 vLLM 中的 GQA 处理

vLLM 在模型加载时自动检测 GQA 配置：

```python
# vLLM 的 GQA 配置处理 (简化)
class ModelConfig:
    def __init__(self, hf_config):
        self.num_attention_heads = hf_config.num_attention_heads
        self.num_key_value_heads = getattr(
            hf_config, 'num_key_value_heads', 
            hf_config.num_attention_heads  # 默认为 MHA
        )
        
        # 计算 KV Cache 大小
        self.kv_cache_size_per_token = (
            2 * self.num_key_value_heads * self.head_dim * 
            self.num_hidden_layers * self.dtype_size
        )
```

## 8. 设计哲学对比：GQA vs MLA vs 量化

```
三种 KV Cache 压缩策略的正交性：

               减少什么？            怎么减？
────────────────────────────────────────────────────
GQA/MQA:       KV heads 数量         结构化共享
MLA:           每个 token 的 KV 维度  低秩投影
量化:          每个元素的 bit 数      数值压缩
选择性缓存:    缓存的 token 数量      重要性筛选

组合效果（理论最大压缩）：
  GQA-8 × FP8 × SnapKV-4x = 8 × 2 × 4 = 64x 压缩
  MLA × FP8 × SnapKV-4x = 57 × 2 × 4 = 456x 压缩

这些方法是正交的，可以自由组合使用
```

## 9. 总结与选择建议

### 9.1 快速选择指南

```
新模型训练：
  ├── 通用场景 → GQA-8（安全且高效）
  ├── 超大规模（>200B）→ MLA（极致 KV 效率）
  └── 延迟敏感、小模型 → MQA（最快 decode）

已有 MHA 模型优化：
  ├── 有计算预算 → Uptrain 为 GQA-8
  ├── 无计算预算 → KV Cache 量化（FP8）
  └── 两者结合 → Uptrain + FP8 量化

部署优化（模型不变）：
  ├── 显存充足 → 全精度 MHA/GQA
  ├── 显存紧张 → FP8 KV Cache
  └── 显存极度紧张 → INT4 量化 + 选择性缓存
```

### 9.2 性能-质量 Pareto 图

```
   推理吞吐 (相对值)
   10x ┤                              MQA ★
       │
   8x  ┤
       │                      GQA-4 ★
   6x  ┤
       │                GQA-8 ★         ← 最佳平衡点
   4x  ┤
       │
   2x  ┤        GQA-16 ★
       │  MHA ★
   1x  ┤
       └──────┬──────┬──────┬──────┬──
             0     0.5    1.0    1.5   2.0  质量损失 (%)
```

GQA-8 始终位于 Pareto 前沿的"甜蜜点"，这解释了为什么它成为了行业标准。

---

> **下一节**：[动手练习](exercises.md) — 通过实践加深对 KV Cache 压缩技术的理解。
