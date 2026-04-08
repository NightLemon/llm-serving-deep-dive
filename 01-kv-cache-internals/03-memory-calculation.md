# 显存占用精确计算

> 本节推导 KV Cache 显存占用的精确公式，并以主流模型为例进行实际计算。掌握这些计算方法，是容量规划和性能调优的基础。

## 1. 基础公式推导

### 1.1 单请求、单层 KV Cache

对于一个请求，单层 Transformer 的 KV Cache 大小为：

$$
\text{KV}_\text{layer} = 2 \times n_{kv} \times d_h \times s \times b_\text{dtype}
$$

其中：
- $2$ — K 和 V 两个张量
- $n_{kv}$ — KV head 数量（MHA: $n_{kv} = n_h$，GQA: $n_{kv} < n_h$）
- $d_h$ — 每个 head 的维度
- $s$ — 序列长度（已生成的 token 数）
- $b_\text{dtype}$ — 数据类型的字节数（FP16=2, FP8=1）

### 1.2 全模型、单请求 KV Cache

叠加所有层：

$$
\boxed{\text{KV}_\text{total} = 2 \times L \times n_{kv} \times d_h \times s \times b_\text{dtype}}
$$

其中 $L$ 为 Transformer 层数。

### 1.3 每 token 的 KV Cache 大小

实际工程中，常用**每 token 的 KV Cache 大小**作为基本单位：

$$
\text{KV}_\text{per\_token} = 2 \times L \times n_{kv} \times d_h \times b_\text{dtype}
$$

这个值是模型的**固有属性**，与序列长度和 batch 无关。

### 1.4 MLA 模型的 KV Cache 公式

对于使用 MLA 的模型（如 DeepSeek-V2/V3），公式不同：

$$
\text{KV}_\text{per\_token}^{MLA} = L \times (d_c + d_h^{rope}) \times b_\text{dtype}
$$

其中：
- $d_c$ — latent 压缩维度（DeepSeek-V2 中为 512）
- $d_h^{rope}$ — RoPE key 的额外维度（DeepSeek-V2 中为 64）
- 注意没有 $\times 2$，因为 latent 同时编码了 K 和 V

---

## 2. 主流模型的架构参数

### 2.1 参数速查表

| 模型 | 参数量 | 层数 $L$ | 隐藏维度 $d$ | Q heads $n_h$ | KV heads $n_{kv}$ | Head dim $d_h$ | Attention 类型 |
|------|-------|---------|-------------|--------------|-------------------|---------------|--------------|
| LLaMA-3-8B | 8B | 32 | 4096 | 32 | 8 | 128 | GQA |
| LLaMA-3-70B | 70B | 80 | 8192 | 64 | 8 | 128 | GQA |
| LLaMA-3.1-405B | 405B | 126 | 16384 | 128 | 8 | 128 | GQA |
| Qwen-2.5-7B | 7B | 28 | 3584 | 28 | 4 | 128 | GQA |
| Qwen-2.5-72B | 72B | 80 | 8192 | 64 | 8 | 128 | GQA |
| Mistral-7B | 7B | 32 | 4096 | 32 | 8 | 128 | GQA |
| DeepSeek-V2 | 236B (MoE) | 60 | 5120 | 128 | MLA | 128 | MLA ($d_c$=512, $d_h^{rope}$=64) |
| DeepSeek-V3 | 671B (MoE) | 61 | 7168 | 128 | MLA | 128 | MLA ($d_c$=512, $d_h^{rope}$=64) |

### 2.2 每 token KV Cache 大小

使用基础公式计算各模型的 `KV_per_token`（FP16）：

```
LLaMA-3-8B:
  KV_per_token = 2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB

LLaMA-3-70B:
  KV_per_token = 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB

LLaMA-3.1-405B:
  KV_per_token = 2 × 126 × 8 × 128 × 2 = 516,096 bytes ≈ 504 KB

Qwen-2.5-7B:
  KV_per_token = 2 × 28 × 4 × 128 × 2 = 57,344 bytes ≈ 56 KB

Qwen-2.5-72B:
  KV_per_token = 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB

DeepSeek-V3 (MLA):
  KV_per_token = 61 × (512 + 64) × 2 = 70,272 bytes ≈ 68.6 KB
```

### 2.3 每 token KV Cache 对比

| 模型 | KV_per_token (FP16) | 相对 LLaMA-3-70B |
|------|-------------------|-----------------|
| LLaMA-3-8B | 128 KB | 0.40x |
| LLaMA-3-70B | 320 KB | 1.00x (基准) |
| LLaMA-3.1-405B | 504 KB | 1.58x |
| Qwen-2.5-7B | 56 KB | 0.18x |
| Qwen-2.5-72B | 320 KB | 1.00x |
| **DeepSeek-V3 (MLA)** | **68.6 KB** | **0.21x** |

**关键发现**：
- DeepSeek-V3 虽然有 671B 参数，但得益于 MLA，其 KV Cache 比 LLaMA-3-8B 还小！
- GQA 的 KV Cache 主要由 `num_kv_heads` 决定，与模型总参数量关系不大
- 同为 8 KV heads 的 LLaMA-3-70B 和 Qwen-2.5-72B，KV_per_token 相同

---

## 3. 实例计算：给定序列长度的显存占用

### 3.1 LLaMA-3-70B

```
配置: L=80, n_kv=8, d_h=128, FP16

场景 1: 单请求, seq_len=4096
  KV_cache = 2 × 80 × 8 × 128 × 4096 × 2
           = 1,342,177,280 bytes
           = 1.25 GB

场景 2: 单请求, seq_len=128K
  KV_cache = 2 × 80 × 8 × 128 × 131072 × 2
           = 42,949,672,960 bytes
           = 40 GB    ← 超过单张 H100 的一半显存！

场景 3: batch=32, avg_seq_len=4096
  KV_cache = 32 × 1.25 GB = 40 GB

场景 4: batch=32, avg_seq_len=4096, FP8
  KV_cache = 32 × 0.625 GB = 20 GB   ← FP8 节省一半
```

### 3.2 DeepSeek-V3

```
配置: L=61, d_c=512, d_h_rope=64, FP16

场景 1: 单请求, seq_len=4096
  KV_cache = 61 × (512 + 64) × 4096 × 2
           = 287,834,112 bytes
           ≈ 0.268 GB

场景 2: 单请求, seq_len=128K
  KV_cache = 61 × 576 × 131072 × 2
           = 9,210,691,584 bytes
           ≈ 8.58 GB    ← 同场景下 LLaMA-3-70B 要 40 GB！

场景 3: batch=32, avg_seq_len=4096
  KV_cache = 32 × 0.268 GB = 8.57 GB
```

### 3.3 Qwen-2.5-72B

```
配置: L=80, n_kv=8, d_h=128, FP16

与 LLaMA-3-70B 的 KV Cache 参数完全相同:
  KV_per_token = 320 KB
  seq_len=4096: 1.25 GB / 请求
  seq_len=32768: 10 GB / 请求
```

---

## 4. KV Cache vs 模型权重的显存占比分析

### 4.1 模型权重的显存占用

模型权重大小的粗略估算：

$$
\text{Weights} \approx P \times b_\text{dtype}
$$

其中 $P$ 为参数量。更精确地，包含 Embedding、Attention 和 FFN 参数：

| 模型 | 参数量 | FP16 权重大小 | BF16 权重大小 | INT8 权重大小 | INT4 权重大小 |
|------|-------|-------------|-------------|-------------|-------------|
| LLaMA-3-8B | 8B | 16 GB | 16 GB | 8 GB | 4 GB |
| LLaMA-3-70B | 70B | 140 GB | 140 GB | 70 GB | 35 GB |
| Qwen-2.5-72B | 72B | 144 GB | 144 GB | 72 GB | 36 GB |
| DeepSeek-V3 | 671B | 1342 GB* | 1342 GB* | 671 GB* | 336 GB* |

*DeepSeek-V3 是 MoE 模型，实际需加载全部 expert 权重。

### 4.2 KV Cache 占比公式

$$
\text{KV ratio} = \frac{\text{KV Cache Size}}{\text{KV Cache Size} + \text{Model Weights}}
$$

### 4.3 不同场景下的占比

以 LLaMA-3-70B (BF16) 为例，模型权重 140 GB：

| 并发 × 序列长度 | KV Cache (FP16) | 总显存 | KV 占比 |
|---------------|----------------|-------|---------|
| 1 × 2K | 0.625 GB | 140.6 GB | 0.4% |
| 1 × 128K | 40 GB | 180 GB | 22% |
| 16 × 4K | 20 GB | 160 GB | 12.5% |
| 64 × 4K | 80 GB | 220 GB | 36% |
| 64 × 32K | 640 GB | 780 GB | **82%** |
| 128 × 8K | 320 GB | 460 GB | **70%** |

```
显存占比随 batch × seq_len 增长:

KV Cache 占比
  100% ┤
       │                                    ╭──── batch=128
   80% ┤                              ╭────╯
       │                         ╭───╯
   60% ┤                    ╭───╯    ╭──── batch=64
       │               ╭───╯   ╭───╯
   40% ┤          ╭───╯   ╭───╯
       │     ╭───╯   ╭───╯       ╭──── batch=16
   20% ┤╭───╯   ╭───╯       ╭───╯
       ││  ╭───╯        ╭───╯
    0% ┤├──┴────────────┴────────────────
       └┼────┼────┼────┼────┼────┼────┼──
        1K   4K   8K  16K  32K  64K  128K
                    序列长度
```

**关键结论**：
1. 对于**短上下文、小 batch** 场景，模型权重是显存主体
2. 对于**长上下文或大 batch** 场景，KV Cache 快速成为主要瓶颈
3. 当 `batch × seq_len` 超过某个阈值后，KV Cache 可占 **70-90%** 的总显存

---

## 5. `gpu_memory_utilization` 参数

### 5.1 定义与作用

vLLM 的 `gpu_memory_utilization` 参数控制**可用于模型+KV Cache 的 GPU 显存比例**：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --gpu-memory-utilization 0.9   # 默认值
```

### 5.2 显存分配流程

```python
# vLLM 显存分配逻辑 (简化)

total_gpu_memory = get_total_gpu_memory()        # 如 80 GB (H100)
usable_memory = total_gpu_memory * gpu_memory_utilization  # 0.9 × 80 = 72 GB

# 1. 加载模型权重
model_memory = load_model_weights()               # 如 35 GB (TP=4)

# 2. 运行 profile 获取激活值峰值
activation_memory = profile_activations()          # 如 2 GB

# 3. 剩余显存用于 KV Cache
kv_cache_memory = usable_memory - model_memory - activation_memory
#                = 72 - 35 - 2 = 35 GB

# 4. 计算可分配的 Block 数量
kv_per_block = (2 * num_layers_per_gpu * num_kv_heads * head_dim
                * block_size * dtype_bytes)
# = 2 × 20 × 8 × 128 × 16 × 2 = 1,310,720 bytes ≈ 1.25 MB (per block, per GPU)

num_blocks = kv_cache_memory // kv_per_block
# = 35 GB / 1.25 MB ≈ 28,000 blocks

# 5. 可缓存的最大 token 数
max_tokens = num_blocks * block_size
# = 28,000 × 16 = 448,000 tokens
```

### 5.3 不同 `gpu_memory_utilization` 的影响

以 LLaMA-3-70B on 4×H100 (TP=4) 为例：

| `gpu_memory_utilization` | 可用显存/GPU | KV Cache 空间/GPU | 可缓存 token 数 | 最大并发 (4K) |
|------------------------|-------------|-----------------|---------------|-------------|
| 0.7 | 56 GB | 19 GB | ~237K tokens | ~58 |
| 0.8 | 64 GB | 27 GB | ~337K tokens | ~82 |
| **0.9 (默认)** | **72 GB** | **35 GB** | **~437K tokens** | **~107** |
| 0.95 | 76 GB | 39 GB | ~487K tokens | ~119 |
| 0.99 | 79.2 GB | 42.2 GB | ~527K tokens | ~129 |

### 5.4 调优建议

```
gpu_memory_utilization 选择策略:

0.9 (默认) ─── 生产环境安全值
  ├── 适合: 稳定的负载，可预测的请求模式
  ├── 预留 10% 给 CUDA context、临时分配、碎片
  └── 大部分场景推荐

0.95 ─── 积极优化
  ├── 适合: 专用推理卡，无其他 GPU 进程
  ├── 注意: 偶尔可能触发 OOM
  └── 建议搭配 --swap-space 使用

0.8 或更低 ─── 保守策略
  ├── 适合: 与其他 GPU 任务共享显卡
  ├── 或: batch 大小波动大的场景
  └── 牺牲吞吐换稳定性
```

---

## 6. 最大可服务并发数计算

### 6.1 通用公式

给定硬件配置和模型参数，最大可服务的并发请求数为：

$$
\text{max\_concurrency} = \frac{\text{KV Cache 可用显存}}{\text{KV}_\text{per\_request}}
$$

其中：

$$
\text{KV}_\text{per\_request} = 2 \times L_\text{per\_gpu} \times n_{kv} \times d_h \times s_\text{max} \times b_\text{dtype}
$$

注意：使用 Tensor Parallelism (TP) 时，$L_\text{per\_gpu} = L / \text{TP}$（近似，实际上 TP 分的是 head 而不是 layer），更准确的是 KV heads 在 TP 之间分配：$n_{kv}^{per\_gpu} = n_{kv} / \text{TP}$。

修正公式：

$$
\text{KV}_\text{per\_request}^{per\_gpu} = 2 \times L \times \frac{n_{kv}}{\text{TP}} \times d_h \times s_\text{max} \times b_\text{dtype}
$$

### 6.2 计算实例

**实例 1：LLaMA-3-70B on 4×H100-80GB (TP=4)**

```
已知:
  总显存: 4 × 80 GB = 320 GB
  gpu_memory_utilization: 0.9
  模型权重 (BF16): 140 GB (每卡 35 GB)
  激活值: ~2 GB/GPU
  L=80, n_kv=8, d_h=128, TP=4

Step 1: KV Cache 可用显存/GPU
  可用 = 80 × 0.9 - 35 - 2 = 35 GB

Step 2: 每请求 KV Cache/GPU (max_seq_len=4096)
  KV/req/gpu = 2 × 80 × (8/4) × 128 × 4096 × 2
             = 2 × 80 × 2 × 128 × 4096 × 2
             = 335,544,320 bytes ≈ 320 MB

Step 3: 最大并发
  max_concurrency = 35 GB / 320 MB ≈ 112 请求
```

**实例 2：LLaMA-3-8B on 1×A100-80GB**

```
已知:
  总显存: 80 GB
  gpu_memory_utilization: 0.9
  模型权重 (BF16): 16 GB
  激活值: ~1 GB
  L=32, n_kv=8, d_h=128, TP=1

Step 1: KV Cache 可用显存
  可用 = 80 × 0.9 - 16 - 1 = 55 GB

Step 2: 每请求 KV Cache (max_seq_len=8192)
  KV/req = 2 × 32 × 8 × 128 × 8192 × 2
         = 1,073,741,824 bytes = 1 GB

Step 3: 最大并发
  max_concurrency = 55 GB / 1 GB = 55 请求
```

**实例 3：DeepSeek-V3 on 8×H100-80GB (TP=8)**

```
已知:
  总显存: 8 × 80 GB = 640 GB
  gpu_memory_utilization: 0.9
  模型权重 (BF16, 部分 expert 按需加载): ~200 GB/GPU (MoE active params)
  注: 实际上 DeepSeek-V3 需要更多 GPU, 此处简化说明
  激活值: ~4 GB/GPU
  L=61, d_c=512, d_h_rope=64

Step 1: KV Cache 可用显存/GPU (假设权重可放下)
  可用 = 80 × 0.9 - 模型权重/GPU - 4 GB

Step 2: 每请求 KV Cache/GPU (max_seq_len=32768)
  KV/req/gpu = 61 × (512 + 64) × 32768 × 2 / 8 (TP=8)
  注: MLA 的 latent 在 TP 时的分配策略不同于 GQA
  ≈ 287 MB (远小于 GQA 模型)
```

### 6.3 快速估算公式

为了日常快速估算，可以记住这些近似值：

| 模型类型 | KV_per_token/GPU (FP16) | 快速记忆 |
|---------|----------------------|---------|
| 8B GQA-8 (TP=1) | 128 KB | **~0.125 MB/tok** |
| 70B GQA-8 (TP=4) | 80 KB | **~0.08 MB/tok** |
| 405B GQA-8 (TP=8) | 63 KB | **~0.06 MB/tok** |

```python
def quick_estimate_max_concurrency(
    total_gpu_memory_gb: float,
    num_gpus: int,
    model_weights_gb: float,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    tp_size: int = 1,
    dtype_bytes: int = 2,
    gpu_memory_utilization: float = 0.9,
    activation_overhead_gb: float = 2.0,
):
    """快速估算最大并发数。"""
    
    # 每 GPU 可用 KV Cache 显存
    available_per_gpu = (
        total_gpu_memory_gb * gpu_memory_utilization
        - model_weights_gb / num_gpus
        - activation_overhead_gb
    )
    
    # 每请求 KV Cache 每 GPU
    kv_per_req_per_gpu = (
        2 * num_layers * (num_kv_heads / tp_size)
        * head_dim * max_seq_len * dtype_bytes
    ) / (1024**3)  # 转为 GB
    
    max_concurrency = int(available_per_gpu / kv_per_req_per_gpu)
    
    print(f"KV Cache 可用/GPU: {available_per_gpu:.1f} GB")
    print(f"KV Cache/请求/GPU: {kv_per_req_per_gpu:.3f} GB")
    print(f"最大并发: {max_concurrency} 请求")
    
    return max_concurrency

# LLaMA-3-70B on 4×H100
quick_estimate_max_concurrency(
    total_gpu_memory_gb=80,
    num_gpus=4,
    model_weights_gb=140,
    num_layers=80,
    num_kv_heads=8,
    head_dim=128,
    max_seq_len=4096,
    tp_size=4,
)
# 输出:
# KV Cache 可用/GPU: 35.0 GB
# KV Cache/请求/GPU: 0.312 GB
# 最大并发: 112 请求
```

---

## 7. `max_model_len` 对吞吐量的影响

### 7.1 `max_model_len` 的含义

vLLM 的 `max_model_len` 定义了**单个请求的最大序列长度**（包括 prompt + generation）：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --max-model-len 4096    # 限制最大 4096 tokens
```

### 7.2 影响链路

```
max_model_len ↑
    │
    ├──→ 每请求 KV Cache ↑ → 最大并发 ↓ → 吞吐量受限
    │
    ├──→ 允许更长上下文 → 适用场景扩大
    │
    └──→ Block Table 行数 ↑ → 微小的 metadata 开销
```

### 7.3 定量分析

LLaMA-3-70B on 4×H100 (TP=4)，KV Cache 可用空间 35 GB/GPU：

| `max_model_len` | KV/请求/GPU | 最大并发 | 峰值吞吐量 (估算) |
|----------------|-----------|---------|----------------|
| 2048 | 160 MB | 224 | ~3500 tok/s |
| 4096 | 320 MB | 112 | ~2800 tok/s |
| 8192 | 640 MB | 56 | ~2000 tok/s |
| 16384 | 1.25 GB | 28 | ~1200 tok/s |
| 32768 | 2.5 GB | 14 | ~700 tok/s |
| 131072 | 10 GB | 3 | ~200 tok/s |

```
吞吐量 vs max_model_len:

吞吐量 (tok/s)
 4000 ┤
      │ ●
 3000 ┤   ●
      │      ●
 2000 ┤         ●
      │
 1000 ┤              ●
      │                    ●
    0 ┤────────────────────────●──
      └─┼────┼────┼────┼────┼──
       2K   4K   8K  16K  32K  128K
              max_model_len
```

### 7.4 实际调优策略

```
场景 1: 聊天应用
  ├── 大部分对话 < 4K tokens
  ├── 偶尔长对话 < 16K tokens
  └── 建议: max_model_len = 8192-16384
            平衡并发和长对话支持

场景 2: RAG / 文档问答
  ├── 长 context (文档) 5K-50K tokens
  ├── 短生成 < 1K tokens
  └── 建议: max_model_len = 32768-65536
            允许长文档输入

场景 3: 代码生成
  ├── 中等 context (代码文件) 2K-8K tokens
  ├── 较长生成 1K-4K tokens
  └── 建议: max_model_len = 8192-16384

场景 4: 批量处理
  ├── 短 prompt < 1K tokens
  ├── 短生成 < 500 tokens
  └── 建议: max_model_len = 2048
            最大化并发和吞吐
```

### 7.5 动态序列长度 vs 静态分配

需要注意的是，vLLM 使用 PagedAttention **不会**为每个请求预分配 `max_model_len` 的完整空间。KV Cache Block 是**按需分配**的——请求实际使用多少 token，就分配多少 Block。

`max_model_len` 的主要作用是：
1. **验证限制**：拒绝超长请求
2. **Block Table 大小**：确定 Block Table 的最大行数
3. **理论上限**：影响 `num_blocks` 的计算（间接影响）

因此，即使设置了较大的 `max_model_len`，如果实际请求较短，vLLM 仍能维持较高的并发。但 `max_model_len` 设得太大会导致一些元数据分配变大，并且如果真有请求用到最大长度，会占用大量 KV Cache。

---

## 8. Tensor Parallelism 下的 KV Cache 分布

### 8.1 TP 下的 KV Head 分配

当使用 Tensor Parallelism 时，KV heads 被均匀分配到各 GPU：

```
LLaMA-3-70B, n_kv=8, TP=4:

GPU 0: KV heads [0, 1]    ← 2 KV heads
GPU 1: KV heads [2, 3]    ← 2 KV heads
GPU 2: KV heads [4, 5]    ← 2 KV heads
GPU 3: KV heads [6, 7]    ← 2 KV heads

每个 GPU 的 KV_per_token = 2 × 80 × 2 × 128 × 2 = 80 KB
```

### 8.2 TP 对 KV Cache 总量的影响

TP 不改变总 KV Cache 大小，只影响分布：

$$
\text{KV}_\text{total} = \text{TP} \times \text{KV}_\text{per\_gpu}
$$

但由于每个 GPU 可用显存独立，TP 实际上**线性扩展**了可缓存的总 token 数：

```
TP=1 (1×H100): 可用 KV 空间 55 GB, n_kv=8
  KV/token = 320 KB → 可缓存 ~180K tokens

TP=4 (4×H100): 每 GPU 可用 KV 空间 35 GB, n_kv_per_gpu=2
  KV/token/GPU = 80 KB → 每 GPU 可缓存 ~460K tokens
  总可缓存 token 不变（每请求占用分散到各 GPU）
  但并发数 = 460K / max_seq_len ← 由单 GPU 决定
```

### 8.3 TP 与 GQA 的约束

当 $n_{kv} < \text{TP}$ 时，需要特殊处理（例如 KV heads 复制）：

```
LLaMA-3-70B, n_kv=8, TP=16 (理论场景):
  8 / 16 = 0.5 KV heads per GPU → 不整除！
  解决方案: 每 2 个 GPU 共享一组 KV heads (需要通信)

实际约束: TP size 通常要求能整除 n_kv
  LLaMA-3-70B (n_kv=8): TP ∈ {1, 2, 4, 8}
  Qwen-2.5-7B (n_kv=4):  TP ∈ {1, 2, 4}
```

---

## 9. 综合计算表

### 9.1 完整场景对比

以下为几种典型部署场景的完整显存计算：

| 配置 | 模型权重 | KV Cache | 激活值 | 总计 | 利用率 |
|------|---------|----------|-------|------|--------|
| LLaMA-3-8B, 1×A100-80GB, TP=1, batch=32, seq=4K, BF16 | 16 GB | 32 GB | ~1 GB | 49 GB | 61% |
| LLaMA-3-70B, 4×H100-80GB, TP=4, batch=64, seq=4K, BF16 | 35 GB/GPU | 20 GB/GPU | ~2 GB | 57 GB/GPU | 71% |
| LLaMA-3-70B, 4×H100-80GB, TP=4, batch=16, seq=32K, BF16 | 35 GB/GPU | 40 GB/GPU | ~2 GB | 77 GB/GPU | 96% |
| Qwen-2.5-72B, 4×H100-80GB, TP=4, batch=64, seq=4K, FP8 KV | 36 GB/GPU | 10 GB/GPU | ~2 GB | 48 GB/GPU | 60% |
| DeepSeek-V3, 8×H100-80GB, TP=8, batch=64, seq=8K, BF16 | ~84 GB/GPU* | 4.3 GB/GPU | ~4 GB | 92 GB/GPU* | - |

*DeepSeek-V3 为 MoE 模型，权重分布较为特殊，实际部署通常需要更多 GPU。

### 9.2 速算心法

记住这几个数字，就能快速估算：

```
黄金法则:
  1 KB ≈ 1 token 的 KV Cache (GQA-8, 8B 模型, TP=1, FP16, 仅限量级估算)

更精确的速算:
  LLaMA-3-8B:   ~0.125 MB/token (FP16, TP=1)
  LLaMA-3-70B:  ~0.08 MB/token/GPU (FP16, TP=4)
  DeepSeek-V3:  ~0.008 MB/token/GPU (FP16, TP=8, MLA)

1 GB KV Cache 可存:
  LLaMA-3-8B:   ~8,000 tokens
  LLaMA-3-70B:  ~12,500 tokens (per GPU, TP=4)
  DeepSeek-V3:  ~125,000 tokens (per GPU, TP=8)
```

---

## 10. 总结

| 要点 | 内容 |
|------|------|
| **基础公式** | $\text{KV} = 2 \times L \times n_{kv} \times d_h \times s \times b_\text{dtype}$ |
| **MLA 公式** | $\text{KV} = L \times (d_c + d_h^{rope}) \times s \times b_\text{dtype}$ |
| **GQA 效果** | $n_{kv}$ 从 64 降到 8 → 8x 节省 |
| **MLA 效果** | 576 维 vs 2048 维 (GQA-8) → ~3.6x 额外节省 |
| **FP8 效果** | 比 FP16 节省 2x |
| **KV 占比趋势** | 短上下文 <10%，长上下文 40-80% |
| **gpu_memory_utilization** | 默认 0.9，影响可用 KV Cache 空间 |
| **max_model_len** | 影响最大并发数，但 vLLM 按需分配 Block |
| **TP 约束** | TP size 需整除 num_kv_heads |

---

## 参考资料

- [vLLM 显存管理文档](https://docs.vllm.ai/en/latest/)
- [LLaMA-3 模型配置](https://github.com/meta-llama/llama3)
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)
- [Qwen-2.5 技术报告](https://arxiv.org/abs/2412.15115)
- [GQA 论文](https://arxiv.org/abs/2305.13245)
