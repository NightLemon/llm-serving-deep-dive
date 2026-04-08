# KV Cache 量化

> 用更少的比特存储 KV Cache，在显存节省和精度损失之间寻找最优平衡。

## 1. 为什么 KV Cache 可以量化？

### 1.1 KV Cache 的显存瓶颈

在自回归解码阶段，KV Cache 是显存的主要消耗者之一。以 LLaMA-3-70B 为例：

```
每个 token 的 KV Cache 大小 = 2 × n_kv_heads × d_head × n_layers × bytes_per_element
                             = 2 × 8 × 128 × 80 × 2 (FP16)
                             = 327,680 bytes ≈ 320 KB/token
```

当 batch_size=64、sequence_length=4096 时，KV Cache 总量达到 **~80 GB**，已经超过单张 H100 (80GB) 的全部显存。模型权重本身还需要 ~140 GB (FP16)。显然，KV Cache 的显存压力是推理系统扩展的核心瓶颈。

### 1.2 K/V 向量的数值分布特征

量化的可行性取决于数据的分布特征。研究表明 K 和 V 向量有不同的数值分布：

**Key 向量**：
- 分布较为集中，通常近似高斯分布
- 不同 channel（即 head dimension 的不同位置）之间方差差异较大
- 存在明显的 **outlier channel**：某些 channel 的数值范围可能是其他 channel 的 10-100 倍
- 这些 outlier 在不同 token 之间高度一致（即同一个 channel 在所有 token 上都是 outlier）

**Value 向量**：
- 分布相对均匀，outlier 现象不如 Key 严重
- 不同 token 之间的数值范围差异较大（per-token 方差变化大）
- 对量化更友好，通常可以用更低精度量化

这种分布差异决定了最优的量化策略：**Key 更适合 per-channel 量化，Value 更适合 per-token 量化**。

```
Key 分布示意（某一层的多个 channel）：
Channel 0:  [-0.2, 0.3, -0.1, 0.2, ...]    范围 ≈ [-0.5, 0.5]
Channel 1:  [-0.1, 0.05, -0.08, 0.1, ...]   范围 ≈ [-0.2, 0.2]
Channel 63: [-5.2, 8.3, -6.1, 7.2, ...]     范围 ≈ [-10, 10]   ← outlier channel
                                              ↑ 量化范围被 outlier 主导

Value 分布示意（某一层的多个 token）：
Token 0:    [-0.3, 0.2, -0.1, 0.5, ...]     范围 ≈ [-0.5, 0.5]
Token 1:    [-1.2, 0.8, -0.5, 1.1, ...]     范围 ≈ [-1.5, 1.5]
Token 2:    [-0.1, 0.05, -0.08, 0.1, ...]   范围 ≈ [-0.2, 0.2]
                                              ↑ token 之间范围差异大
```

## 2. 量化粒度：从粗到细

量化粒度（granularity）是影响量化质量的关键设计选择。粒度越细，量化误差越小，但需要存储更多的 scale/zero-point 元数据。

### 2.1 Per-tensor 量化

整个 KV Cache tensor 共享一组 scale 和 zero-point：

```python
# Per-tensor quantization
scale = (max(tensor) - min(tensor)) / (2^bits - 1)
zero_point = round(-min(tensor) / scale)
quantized = round(tensor / scale) + zero_point
```

- **优点**：元数据开销最小，kernel 实现简单
- **缺点**：outlier 会严重拉大量化范围，导致大量精度损失
- **适用场景**：FP8 量化（动态范围已经足够大）

### 2.2 Per-token 量化

每个 token 的 KV 向量独立量化，拥有自己的 scale/zero-point：

```python
# Per-token quantization
for token_idx in range(seq_len):
    kv_vec = kv_cache[token_idx]  # shape: [num_heads * head_dim]
    scale[token_idx] = (max(kv_vec) - min(kv_vec)) / (2^bits - 1)
    quantized[token_idx] = round(kv_vec / scale[token_idx])
```

- **优点**：适应不同 token 之间的数值范围差异（对 Value 特别有效）
- **缺点**：同一 token 内的 outlier channel 仍然影响精度
- **元数据开销**：每个 token 额外存储 1 个 scale（+ 1 个 zero_point）

### 2.3 Per-channel 量化

沿 head_dim 维度为每个 channel 计算独立的 scale/zero-point：

```python
# Per-channel quantization
for ch in range(head_dim):
    channel_vals = kv_cache[:, ch]  # 所有 token 的第 ch 个 channel
    scale[ch] = (max(channel_vals) - min(channel_vals)) / (2^bits - 1)
    quantized[:, ch] = round(channel_vals / scale[ch])
```

- **优点**：完美处理 outlier channel 问题（对 Key 特别有效）
- **缺点**：新增 token 时需要更新所有 channel 的统计量（或使用 calibration）
- **元数据开销**：每个 channel 额外存储 1 个 scale

### 2.4 Per-group 量化

将 channel 分组，每组共享 scale/zero-point，是 per-tensor 和 per-channel 的折中：

```python
# Per-group quantization (group_size = 32/64/128)
for group_start in range(0, head_dim, group_size):
    group = kv_cache[:, group_start:group_start+group_size]
    scale[group_start // group_size] = (max(group) - min(group)) / (2^bits - 1)
```

- 常见 group_size：32、64、128
- GPTQ、AWQ 等权重量化方法广泛使用此粒度

## 3. FP8 量化：精度损失最小的方案

### 3.1 FP8 格式

FP8 有两种主要格式，由 IEEE 提议并被 NVIDIA Hopper 架构（H100）原生支持：

| 格式 | 符号位 | 指数位 | 尾数位 | 动态范围 | 精度 | 适用场景 |
|------|-------|--------|--------|---------|------|---------|
| E4M3 | 1 | 4 | 3 | ±240 | 较高 | KV Cache、权重 |
| E5M2 | 1 | 5 | 2 | ±57344 | 较低 | 梯度、动态范围大的场景 |

对于 KV Cache 量化，**E4M3 是首选格式**，因为：
1. KV Cache 的数值通常在 [-10, 10] 范围内，E4M3 的动态范围（±240）完全足够
2. 多一位尾数（3 vs 2）意味着更高的精度
3. 从 FP16 到 FP8 E4M3 的量化误差通常小于 1%

### 3.2 FP8 量化的工作方式

```python
# FP8 E4M3 量化（simplified）
def quantize_to_fp8_e4m3(tensor: torch.Tensor):
    # 计算 per-tensor scale（使 tensor 范围适配 E4M3 的表示范围）
    amax = tensor.abs().max()
    scale = (240.0 / amax).clamp(max=1.0)  # E4M3 max = 240
    
    # 缩放并转换为 FP8
    scaled_tensor = tensor * scale
    fp8_tensor = scaled_tensor.to(torch.float8_e4m3fn)
    
    return fp8_tensor, scale  # scale 用于反量化

def dequantize_from_fp8(fp8_tensor, scale):
    return fp8_tensor.to(torch.float16) / scale
```

### 3.3 FP8 KV Cache 的显存节省

```
FP16 KV Cache per token (LLaMA-3-70B): 320 KB
FP8  KV Cache per token (LLaMA-3-70B): 160 KB
                                        ↓
                              节省 50% 显存
```

FP8 量化的核心优势是**几乎无损**。在多项研究中，FP8 KV Cache 对 perplexity 的影响通常小于 0.1，对下游任务的准确率影响不超过 0.5%。

## 4. INT8 / INT4 量化：更激进的压缩

### 4.1 INT8 量化

INT8 使用定点整数表示，需要显式的 scale 和 zero-point：

```python
# INT8 对称量化
scale = amax / 127.0
quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)

# INT8 非对称量化
scale = (max_val - min_val) / 255.0
zero_point = round(-min_val / scale)
quantized = torch.clamp(torch.round(tensor / scale) + zero_point, 0, 255).to(torch.uint8)
```

INT8 的精度损失比 FP8 略大，主要因为：
- 整数量化在数值密集区域的分辨率不如浮点格式
- 需要额外处理 outlier（通常通过 per-channel 或 per-group 量化缓解）

### 4.2 INT4 量化

INT4 将 KV Cache 压缩到原始大小的 25%，但精度挑战显著增大：

```
FP16: 16 bits → INT4: 4 bits = 75% 显存节省
但只有 2^4 = 16 个离散表示值（对称量化有效值 15 个）
```

直接的 INT4 per-tensor 量化通常不可接受，必须结合：
- Fine-grained 量化粒度（per-channel 或 per-group）
- 非对称量化（asymmetric quantization）
- Calibration-based 的 scale 选择
- Outlier 特殊处理

## 5. 量化误差在多层注意力中的累积

KV Cache 量化的一个关键挑战是**误差累积效应**：

```
Layer 0: Q × K_quant^T → attention_scores（引入误差 ε₀）
         → softmax → × V_quant（引入误差 δ₀）
         → output₀ = f(ε₀, δ₀)

Layer 1: 基于 output₀ 计算新的 Q, K, V
         → Q₁ × K₁_quant^T → attention_scores（引入误差 ε₁ + 从 ε₀ 传播的误差）
         ...

Layer L: 累积误差 = g(ε₀, δ₀, ε₁, δ₁, ..., ε_L, δ_L)
```

这种累积有两个维度：
1. **层间累积**：模型通常有 32-80 层，每层的量化误差都会传递到下一层
2. **序列累积**：在长序列推理中，早期 token 的量化 KV 会被反复使用，误差影响更大

研究发现：
- 前几层和最后几层对量化更敏感，中间层相对鲁棒
- **混合精度策略**：对关键层使用 FP16，其余层使用 INT8/INT4，可以有效控制累积误差
- Key 的量化误差比 Value 的影响更大（因为 Key 参与 softmax 计算，误差被非线性放大）

## 6. vLLM 中的量化 KV Cache 配置

### 6.1 启用 FP8 KV Cache

```bash
# 方式 1：启动 vLLM 服务时指定 KV Cache 数据类型
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 4

# 方式 2：使用 fp8_e4m3（显式指定格式）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --kv-cache-dtype fp8_e4m3

# 方式 3：结合量化模型配置
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8
```

### 6.2 关键实现路径

vLLM 中 KV Cache 量化的核心逻辑分布在以下文件：

```
vllm/
├── attention/
│   ├── backends/
│   │   ├── flash_attn.py     # FlashAttention backend，支持 FP8 KV
│   │   └── utils.py          # KV Cache 量化/反量化工具
├── model_executor/
│   └── layers/
│       └── quantization/
│           └── fp8.py         # FP8 量化逻辑
└── worker/
    └── cache_engine.py        # KV Cache 内存分配（按 dtype 分配空间）
```

### 6.3 量化流程

```python
# 简化的 vLLM KV Cache 量化流程
class Attention:
    def forward(self, query, key, value, kv_cache):
        # 1. 如果启用了 FP8，在写入 cache 前量化
        if self.kv_cache_dtype == "fp8":
            key, key_scale = quantize_kv(key, dtype=torch.float8_e4m3fn)
            value, val_scale = quantize_kv(value, dtype=torch.float8_e4m3fn)
        
        # 2. 写入 paged KV cache（存储的是量化后的值）
        write_to_paged_cache(key, value, kv_cache, slot_mapping)
        
        # 3. 在 attention 计算时，由 kernel 内部处理反量化
        #    FlashAttention FP8 kernel 直接接受 FP8 KV
        output = flash_attention_with_fp8_kv(
            query,           # FP16/BF16
            kv_cache_key,    # FP8
            kv_cache_value,  # FP8
            key_scale,
            val_scale
        )
        return output
```

## 7. 量化感知的 Attention Kernel

### 7.1 FlashAttention 对 FP8 KV 的支持

FlashAttention-3（以及后续版本）原生支持 FP8 KV Cache：

```
标准 FlashAttention:  Q(FP16) × K^T(FP16) → Scores → Softmax → × V(FP16)
FP8-aware FlashAttention: Q(FP16) × K^T(FP8) → Scores → Softmax → × V(FP8)
                                       ↑                           ↑
                               kernel 内部反量化            kernel 内部反量化
```

关键设计点：
- **反量化发生在 kernel 内部**：FP8→FP16 的转换在 SRAM 中完成，不需要额外的全局内存读写
- **计算仍在 FP16/FP32 精度**：只有存储使用 FP8，矩阵乘法和累加使用高精度
- **带宽节省是核心收益**：FP8 KV 意味着从 HBM 读取 KV Cache 的带宽减半

### 7.2 性能提升

FP8 KV Cache 的性能收益主要来自两方面：

```
1. 显存节省 → 支持更大 batch size → 提高吞吐
   FP16 KV: 能放 batch_size=32 的 KV Cache
   FP8  KV: 能放 batch_size=64 的 KV Cache  → 吞吐翻倍（在 memory-bound 阶段）

2. 带宽节省 → 降低 memory-bound attention 的延迟
   Decode 阶段的 attention 是 memory-bound（Q 只有 1 个 token）
   读取 KV Cache 的数据量减半 → attention 延迟降低 ~40-50%
```

## 8. 不同量化精度的质量对比

### 8.1 Perplexity 对比

以下数据来源于多篇论文和实验报告的综合：

| 模型 | KV 精度 | WikiText-2 PPL | 相对变化 | 显存节省 |
|------|---------|---------------|---------|---------|
| LLaMA-2-7B | FP16 (baseline) | 5.47 | — | 0% |
| LLaMA-2-7B | FP8 E4M3 | 5.48 | +0.2% | 50% |
| LLaMA-2-7B | INT8 (per-token) | 5.52 | +0.9% | 50% |
| LLaMA-2-7B | INT4 (per-channel) | 5.78 | +5.7% | 75% |
| LLaMA-2-7B | INT4 (per-group-64) | 5.61 | +2.6% | ~72% |
| LLaMA-2-7B | INT2 (KIVI) | 5.94 | +8.6% | ~87% |

### 8.2 Pareto 分析

```
   显存节省 (%)
   100 ┤
    90 ┤                              ★ INT2 (KIVI)
    80 ┤
    75 ┤              ★ INT4-g64      ★ INT4-pc
    70 ┤
    60 ┤
    50 ┤  ★ FP8          ★ INT8
    40 ┤
    30 ┤
    20 ┤
    10 ┤
     0 ┤★ FP16
       └──────┬──────┬──────┬──────┬──────┬──
              0     2      5      8     10    PPL 增加 (%)

Pareto 前沿：FP8 → INT4-g64 → INT2(KIVI)
```

**实际工程建议**：
- **首选 FP8**：几乎零精度损失，50% 显存节省，工程复杂度最低
- **需要更多显存时用 INT4 per-group**：精度损失可控，75% 显存节省
- **极端场景用 INT2（KIVI）**：适合超长上下文，需要仔细评估精度影响

## 9. KIVI：2-bit 非对称 KV 量化

### 9.1 核心思想

[KIVI](https://arxiv.org/abs/2402.02750)（2024）提出了一种 tuning-free 的 2-bit KV Cache 量化方案，核心洞察：

1. **Key 使用 per-channel 量化**：因为 Key 的 outlier 沿 channel 维度分布一致
2. **Value 使用 per-token 量化**：因为 Value 的数值范围主要沿 token 维度变化
3. **非对称量化**：使用不同的 min/max 做 scale，比对称量化更精确

```python
# KIVI 的量化策略
def kivi_quantize(key_cache, value_cache, bits=2):
    # Key: per-channel 非对称量化
    for ch in range(head_dim):
        k_channel = key_cache[:, :, ch]
        k_min, k_max = k_channel.min(), k_channel.max()
        k_scale = (k_max - k_min) / (2**bits - 1)
        key_quantized[:, :, ch] = round((k_channel - k_min) / k_scale)
    
    # Value: per-token 非对称量化
    for tok in range(seq_len):
        v_token = value_cache[:, tok, :]
        v_min, v_max = v_token.min(), v_token.max()
        v_scale = (v_max - v_min) / (2**bits - 1)
        value_quantized[:, tok, :] = round((v_token - v_min) / v_scale)
    
    return key_quantized, value_quantized
```

### 9.2 Residual 保护

KIVI 对最近的 token（例如最后 128 个）保留 FP16 精度，只对之前的 token 做 2-bit 量化：

```
[FP16 sink tokens][INT2 量化区域 ··················][FP16 最近 128 tokens]
     ↑ 前 4 个                                            ↑ sliding window
```

这种设计基于两个观察：
- 最近的 token 对当前生成的影响最大
- 最近 token 的 KV 在后续步骤中可能还需要更新（如 beam search）

### 9.3 显存压缩效果

```
2-bit KIVI vs FP16 baseline:
- 显存压缩比：~8x（2/16 + scale 开销）
- LLaMA-2-7B on 128K context: 
  FP16 KV Cache: ~40 GB
  KIVI 2-bit:    ~5.5 GB（含 residual FP16 window）
```

## 10. KVQuant：超长上下文 KV 量化

### 10.1 动机

[KVQuant](https://arxiv.org/abs/2401.18079)（2024）面向超长上下文（1M+ tokens）场景，目标是实现 10M context length 的 LLM 推理。在这种极端场景下，KV Cache 是绝对的显存瓶颈：

```
LLaMA-2-70B @ 1M tokens (FP16 KV):
  KV Cache = 2 × 8 × 128 × 80 × 1,000,000 × 2 bytes
           = 327,680,000,000 bytes ≈ 305 GB
```

### 10.2 核心技术

KVQuant 综合使用了多种技术：

1. **Per-channel Key 量化 + Per-token Value 量化**（与 KIVI 类似的洞察）
2. **Non-uniform quantization (NUQ)**：使用 k-means 聚类找到最优量化 codebook，而非均匀分割
3. **Dense-and-Sparse 量化**：将 outlier 单独用 sparse 格式存储，其余用 dense 低比特量化
4. **Q-Norm**：对 Query 做归一化，减少 Key 量化误差对 attention score 的影响

```python
# KVQuant 的 Dense-and-Sparse 策略（概念示意）
def kvquant_quantize(tensor, bits=4, outlier_threshold=6.0):
    # 1. 识别 outlier
    mean, std = tensor.mean(), tensor.std()
    outlier_mask = (tensor - mean).abs() > outlier_threshold * std
    
    # 2. outlier 用 FP16 sparse 格式保存
    outlier_indices = outlier_mask.nonzero()
    outlier_values = tensor[outlier_mask].to(torch.float16)
    
    # 3. 非 outlier 部分用 NUQ 量化
    normal_values = tensor[~outlier_mask]
    codebook = kmeans(normal_values, n_clusters=2**bits)
    quantized_indices = vq(normal_values, codebook)
    
    return quantized_indices, codebook, outlier_indices, outlier_values
```

### 10.3 效果

KVQuant 在 4-bit 量化下实现了与 FP16 基本持平的精度，3-bit 量化也保持了可接受的精度：

| 量化方案 | bits | LLaMA-2-7B PPL | 压缩比 |
|---------|------|-----------------|-------|
| FP16 | 16 | 5.47 | 1x |
| KVQuant (uniform) | 4 | 5.58 | 4x |
| KVQuant (NUQ) | 4 | 5.51 | 4x |
| KVQuant (NUQ+sparse) | 3 | 5.56 | ~5x |
| KVQuant (NUQ+sparse) | 2 | 5.89 | ~7x |

## 11. 工程实践总结

### 11.1 量化方案选择指南

```
                    精度要求高？
                   /           \
                  是             否
                 /               \
            FP8 E4M3          上下文长吗？
          (50% 节省)          /         \
                           短(<8K)      长(>32K)
                            /              \
                      INT8 per-token    INT4/INT2
                      (50% 节省)     (75-87% 节省)
                                          |
                                    KIVI / KVQuant
```

### 11.2 与其他优化的组合

KV Cache 量化可以与其他技术叠加：

| 组合方案 | 效果 | 注意事项 |
|---------|------|---------|
| FP8 KV + PagedAttention | 每个 page block 存储 FP8 数据 | vLLM 原生支持 |
| FP8 KV + Prefix Caching | 缓存的 KV 以 FP8 存储 | 需确保 scale 一致性 |
| INT4 KV + GQA | 压缩效果叠加 | GQA 已减少 KV heads，量化进一步压缩每个 head |
| 量化 KV + Tensor Parallelism | 每个 GPU 存储量化后的分片 | 通信量也相应减少 |

### 11.3 注意事项

1. **Calibration 数据选择**：量化的 scale 应基于有代表性的输入数据校准
2. **动态 vs 静态 scale**：动态 scale 更准确但有计算开销，静态 scale 更快但可能不适应所有输入
3. **首 token 保护**：许多方案对序列开头的 token 保留高精度（attention sink 效应）
4. **评估要全面**：不要只看 perplexity，还要测试具体下游任务（摘要、问答、代码生成等）
5. **硬件兼容性**：FP8 需要 Hopper (H100) 或更新架构；INT8/INT4 的高效 kernel 需要特定 GPU 支持

---

> **下一节**：[MLA: Multi-head Latent Attention 深度解读](02-mla.md) — 从架构层面压缩 KV Cache。
