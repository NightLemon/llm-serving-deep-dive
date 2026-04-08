# KV Cache 内存布局

> 本节从物理存储的视角拆解 KV Cache，理解不同 Attention 变体（MHA / GQA / MLA）在 GPU 显存中的真实布局。

## 1. Attention 计算中的 K、V 张量

### 1.1 Self-Attention 回顾

在标准 Transformer 的 Self-Attention 中，输入 $X \in \mathbb{R}^{s \times d}$ 通过三个线性投影生成 Q、K、V：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$，$s$ 为序列长度，$d$ 为隐藏维度。

Attention 的核心计算为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right) V
$$

**关键洞察**：在自回归生成中，每生成一个新 token，都需要用到**所有历史** token 的 K 和 V。如果不缓存这些张量，每步生成都要重新计算前面所有 token 的 KV，时间复杂度为 $O(s^2)$。KV Cache 的本质就是"**用空间换时间**"。

### 1.2 单层 KV Cache 的形状

对于 Multi-Head Attention (MHA)，每一层的 KV Cache 张量形状为：

| 张量 | 形状 | 说明 |
|------|------|------|
| K cache | `[batch, num_heads, seq_len, head_dim]` | 所有 head 的 Key |
| V cache | `[batch, num_heads, seq_len, head_dim]` | 所有 head 的 Value |

其中：
- `batch`：当前 batch 中的请求数
- `num_heads`：注意力头的数量（如 LLaMA-3-70B 为 64）
- `seq_len`：已生成的序列长度（动态增长）
- `head_dim`：每个注意力头的维度（通常为 128）

### 1.3 全模型 KV Cache 的逻辑视图

整个模型的 KV Cache 可以理解为一个 6D 张量：

```
KV Cache 全局形状:
[num_layers, 2, batch, num_kv_heads, seq_len, head_dim]
     │       │    │        │           │         │
     │       │    │        │           │         └── 每个 head 的维度 (128)
     │       │    │        │           └── 序列位置 (动态增长)
     │       │    │        └── KV head 数量 (MHA=64, GQA=8, MQA=1)
     │       │    └── batch 中的请求数
     │       └── K 和 V 两个张量
     └── Transformer 层数
```

---

## 2. 连续内存 vs 分页内存

### 2.1 连续内存分配（传统实现）

在 HuggingFace Transformers 等早期实现中，每个请求的 KV Cache 被分配为**一块连续的 GPU 显存**：

```
请求 A (prompt=500, max_gen=1500):
┌─────────────────────────────────────────────────────┐
│ █ █ █ █ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ ░ │  已分配 2000 tokens
│ ← 500 已用 →│← ─── 1500 预留但未使用 ───────────→ │
└─────────────────────────────────────────────────────┘

请求 B (prompt=200, max_gen=100):
┌──────────────┐
│ █ █ ░ ░ ░ ░ │  已分配 300 tokens
│200│← 100 →│
└──────────────┘
```

**连续分配的三大问题**：

1. **内部碎片（Internal Fragmentation）**：必须按 `max_seq_len` 预分配，但大多数请求实际生成长度远小于最大值。实测中浪费比例通常为 **60-80%**。

2. **外部碎片（External Fragmentation）**：请求完成后释放的显存块大小不一，难以被新请求复用：

```
GPU 显存:
┌──────┐┌────────────────┐┌──────┐┌────────────────────────┐
│ 在用 ││     空闲       ││ 在用 ││        空闲            │
│ 200  ││     800        ││ 300  ││        1200            │
└──────┘└────────────────┘└──────┘└────────────────────────┘
                                    ↑ 新请求需要 1500，但没有连续的 1500 空间
```

3. **无法共享（No Sharing）**：即使多个请求共享相同的 system prompt，每个请求都必须独立存储一份 KV Cache 副本。

### 2.2 分页内存分配（PagedAttention）

vLLM 引入的 PagedAttention 借鉴了操作系统的**虚拟内存分页**思想：

- KV Cache 被切分为固定大小的 **Block**（默认 16 tokens/block）
- 每个请求通过一个 **Block Table** 记录其使用的 Block 编号
- Block 在物理显存上**不需要连续**

```
Block Pool (物理层):
┌────────┬────────┬────────┬────────┬────────┬────────┐
│Block 0 │Block 1 │Block 2 │Block 3 │Block 4 │Block 5 │ ...
│ Req A  │ Req B  │ Req A  │ 空闲   │ Req A  │ Req B  │
└────────┴────────┴────────┴────────┴────────┴────────┘

Block Table (逻辑层):
Req A → [0, 2, 4]     # 物理 Block 0, 2, 4
Req B → [1, 5]         # 物理 Block 1, 5
```

**分页的优势**：
- **接近零浪费**：只有最后一个 Block 可能有内部碎片（平均浪费 block_size/2 个 token）
- **消除外部碎片**：所有空闲 Block 等价，可被任意请求使用
- **支持共享**：相同 prefix 的 Block 可通过引用计数共享（Prefix Caching 的基础）

---

## 3. MHA 布局详解

### 3.1 MHA 的 KV Cache 形状

Multi-Head Attention 中，Q、K、V 的 head 数量相同：

```
MHA 配置 (以 GPT-3 175B 为例):
- num_heads = 96
- head_dim = 128
- num_kv_heads = 96  (与 Q heads 相同)
```

每一层的单个请求 KV Cache 大小：

$$
\text{KV}_\text{per\_layer} = 2 \times n_h \times d_h \times s \times \text{sizeof(dtype)}
$$

其中 $n_h = 96, d_h = 128$。

### 3.2 在 vLLM 中的 Block 布局

vLLM 中每个 KV Cache Block 的张量形状为：

```python
# vLLM block shape for MHA
key_block_shape = (num_heads, block_size, head_dim)
value_block_shape = (num_heads, block_size, head_dim)

# 例：LLaMA-3-70B MHA 配置
# key_block: [64, 16, 128] = 131,072 elements
# 每个 block 的 K cache: 131,072 × 2 bytes (FP16) = 256 KB
# K + V 合计: 512 KB / block / layer
```

---

## 4. GQA 布局：存储优化的关键

### 4.1 GQA 的核心思想

Grouped-Query Attention (GQA) 将多个 Q head **分组共享**同一对 KV head：

```
MHA (num_heads=8, num_kv_heads=8):
Q: h0 h1 h2 h3 h4 h5 h6 h7
K: h0 h1 h2 h3 h4 h5 h6 h7   ← 8 对 KV heads
V: h0 h1 h2 h3 h4 h5 h6 h7

GQA (num_heads=8, num_kv_heads=2):
Q: h0 h1 h2 h3 | h4 h5 h6 h7
K:    kv0       |     kv1        ← 只需 2 对 KV heads
V:    kv0       |     kv1

每 4 个 Q head 共享 1 对 KV head
```

### 4.2 GQA KV Cache 显存节省

| 模型 | num_heads (Q) | num_kv_heads | GQA Ratio | KV Cache 缩减 |
|------|-------------|-------------|-----------|--------------|
| LLaMA-2-70B | 64 | 8 | 8:1 | **8x** |
| LLaMA-3-8B | 32 | 8 | 4:1 | **4x** |
| LLaMA-3-70B | 64 | 8 | 8:1 | **8x** |
| Qwen-2.5-72B | 64 | 8 | 8:1 | **8x** |
| Mistral-7B | 32 | 8 | 4:1 | **4x** |

GQA 下，KV Cache 布局中 `num_kv_heads` 维度显著缩小：

```python
# GQA KV Cache block shape (LLaMA-3-70B)
key_block_shape = (num_kv_heads, block_size, head_dim)
                = (8, 16, 128)   # 仅 16,384 elements
# 对比 MHA: (64, 16, 128) = 131,072 elements
# 8 倍节省！
```

### 4.3 GQA 中的 Q-K 映射

在 Attention 计算中，需要将 Q 的 head index 映射到对应的 KV head index：

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV heads 扩展以匹配 Q heads 的数量。
    
    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: num_heads // num_kv_heads (GQA ratio)
    Returns:
        [batch, num_heads, seq_len, head_dim]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

注意：`expand` 不会分配新显存（zero-copy），所以 GQA 的计算效率很高。

---

## 5. MLA 布局：DeepSeek-V2 的革新

### 5.1 MLA 的动机

Multi-head Latent Attention (MLA) 由 DeepSeek-V2 提出，其核心思想是：**不缓存原始的 K 和 V，而是缓存一个低维的 latent 向量**。

在标准 Attention 中，KV Cache 的大小与 `num_kv_heads × head_dim` 成正比。MLA 通过低秩投影将其压缩到一个维度远小于 `num_kv_heads × head_dim` 的 latent space。

### 5.2 MLA 的数学形式

**下投影（缓存时）**：

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}
$$

其中 $d_c \ll n_h \times d_h$。只需缓存 $c_t^{KV}$，而不是完整的 K 和 V。

**上投影（计算时）**：

$$
K_t = W^{UK} c_t^{KV}, \quad V_t = W^{UV} c_t^{KV}
$$

### 5.3 MLA vs GQA 的 KV Cache 大小

| 配置 | 缓存内容 | 每 token 缓存维度 | 示例 |
|------|---------|----------------|------|
| MHA (64 heads) | K + V | $2 \times 64 \times 128 = 16384$ | GPT-3 |
| GQA (8 KV heads) | K + V | $2 \times 8 \times 128 = 2048$ | LLaMA-3-70B |
| MLA | $c^{KV}$ + RoPE key | $d_c + d_h^{rope} = 512 + 64 = 576$ | DeepSeek-V2 |

DeepSeek-V2 的配置：
- $d_c = 512$（latent 压缩维度）
- $d_h^{rope} = 64$（需要额外缓存的 RoPE 部分，因为 RoPE 与位置相关，无法被压缩吸收）

**MLA 的 KV Cache 比 GQA 再缩小 ~3.6x**（2048 / 576）。

### 5.4 MLA 的 Cache 布局

MLA 的 Cache 布局与 MHA/GQA 完全不同：

```python
# MHA/GQA Cache 布局:
# K: [batch, num_kv_heads, seq_len, head_dim]
# V: [batch, num_kv_heads, seq_len, head_dim]

# MLA Cache 布局:
# latent: [batch, seq_len, d_c + d_h_rope]
#                          └─ 512 + 64 = 576 维

# 注意：
# 1. 没有 "2" (K/V) 维度 —— latent 同时编码了 K 和 V
# 2. 没有 num_heads 维度 —— latent 是所有 head 的共享压缩表示
# 3. 计算时需要上投影恢复 K/V，这是额外的计算开销
```

### 5.5 MLA 的工程权衡

| 维度 | GQA | MLA |
|------|-----|-----|
| KV Cache 显存 | 基准 1x | ~0.28x |
| 计算开销 | 标准 Attention | 额外的上投影矩阵乘法 |
| 适合场景 | 通用 | 超长上下文 / 大 batch |
| 实现复杂度 | 低 | 高 |
| 推理框架支持 | 完善 | vLLM/SGLang 已支持 |

---

## 6. vLLM KVCache 张量分配源码分析

### 6.1 V1 架构中的 KV Cache 分配

在 vLLM V1 架构中，KV Cache 的分配逻辑位于 `vllm/v1/worker/gpu_model_runner.py`：

```python
# 简化的 KV Cache 初始化流程
class GPUModelRunner:
    def _initialize_kv_caches(self, kv_cache_config):
        """为所有层分配 KV Cache 张量。"""
        
        # 1. 计算每个 block 的形状
        #    vLLM v1 使用 [num_blocks, block_size, num_kv_heads, head_dim] 的布局
        kv_cache_shape = self._get_kv_cache_shape()
        
        # 2. 预分配整个 KV Cache pool
        #    所有 block 一次性分配，之后通过 block manager 动态分配给请求
        self.gpu_cache = []
        for layer_idx in range(self.num_layers):
            # 每层分配 K 和 V 两个张量
            key_cache = torch.zeros(
                kv_cache_shape,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
            value_cache = torch.zeros(
                kv_cache_shape,
                dtype=self.kv_cache_dtype,
                device=self.device,
            )
            self.gpu_cache.append((key_cache, value_cache))
```

### 6.2 KV Cache 的核心形状

vLLM 中 KV Cache Block 的实际张量形状为：

```python
# vLLM v1 KV Cache Block 形状
kv_cache_shape = (num_blocks, block_size, num_kv_heads, head_dim)

# 例：LLaMA-3-70B, block_size=16, num_blocks=2000
# key_cache shape: [2000, 16, 8, 128]
# 每层 K cache 大小: 2000 × 16 × 8 × 128 × 2 = 64 MB (FP16)
# 80 层 K+V 总计: 80 × 64 × 2 = 10,240 MB ≈ 10 GB
```

### 6.3 num_blocks 的确定

可用的 Block 数量由以下因素决定：

```python
def determine_num_available_blocks(self):
    """确定可分配多少个 KV Cache blocks。"""
    
    # 1. 获取总 GPU 显存
    total_gpu_memory = torch.cuda.get_device_properties(
        self.device).total_memory
    
    # 2. 计算模型权重和激活占用
    #    通过一次 dummy forward 测量
    peak_memory = self._profile_run()
    
    # 3. 可用于 KV Cache 的显存
    #    gpu_memory_utilization 默认 0.9，预留 10% 给其他用途
    usable_memory = total_gpu_memory * gpu_memory_utilization
    kv_cache_memory = usable_memory - peak_memory
    
    # 4. 计算 block 数量
    block_size_bytes = (
        block_size * num_kv_heads * head_dim * dtype_size * 2  # K + V
        * num_layers
    )
    num_blocks = kv_cache_memory // block_size_bytes
    return num_blocks
```

---

## 7. HuggingFace Transformers：DynamicCache vs StaticCache

### 7.1 DynamicCache（默认）

`DynamicCache` 是 HuggingFace Transformers `>= 4.36` 的默认 Cache 实现：

```python
class DynamicCache(Cache):
    """动态增长的 KV Cache，每次 forward 追加新的 KV。"""
    
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []   # 每层一个 tensor
        self.value_cache: List[torch.Tensor] = []
    
    def update(self, key_states, value_states, layer_idx):
        """追加新的 key/value states 到 cache 中。"""
        if layer_idx == len(self.key_cache):
            # 新层：直接存储
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 已有层：在 seq_len 维度上 cat
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

**DynamicCache 的问题**：
- 每步 `torch.cat` 导致**频繁内存分配和拷贝**
- 旧 tensor 需要被 GC 回收，增加显存碎片
- 不可预测的内存增长模式

### 7.2 StaticCache

`StaticCache` 通过预分配固定大小的缓冲区解决上述问题：

```python
class StaticCache(Cache):
    """预分配的固定大小 KV Cache，适合 torch.compile。"""
    
    def __init__(self, config, max_batch_size, max_cache_len, dtype=None):
        self.max_cache_len = max_cache_len
        
        # 预分配完整缓冲区
        cache_shape = (
            max_batch_size,
            config.num_key_value_heads,
            max_cache_len,
            config.head_dim,
        )
        self.key_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(config.num_hidden_layers)
        ]
        self.value_cache = [
            torch.zeros(cache_shape, dtype=dtype, device=device)
            for _ in range(config.num_hidden_layers)
        ]
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        """通过 scatter 写入新的 KV，无需内存分配。"""
        # 直接写入预分配的位置，零分配
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        
        # 使用 index_copy_ 或 scatter_ 写入指定位置
        cache_position = cache_kwargs.get("cache_position")
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        
        return k_out, v_out
```

### 7.3 对比总结

| 特性 | DynamicCache | StaticCache |
|------|-------------|-------------|
| 内存分配 | 每步动态分配 (torch.cat) | 一次性预分配 |
| 显存效率 | 低（碎片化） | 中（可能浪费，但无碎片） |
| torch.compile | 不兼容（动态 shape） | 兼容 |
| 适用场景 | 研究、原型 | 生产推理 |
| 最大序列长度 | 仅受显存限制 | 需要预设 max_cache_len |

---

## 8. Block Table 的定义与索引方式

### 8.1 Block Table 数据结构

Block Table 是 PagedAttention 的核心数据结构，它维护了**逻辑 Block 到物理 Block 的映射**：

```python
# Block Table 的逻辑结构
# block_tables[req_id] = [phys_block_0, phys_block_1, ..., phys_block_n]
#
# 例：请求 A 有 50 个 token，block_size=16
# 需要 ceil(50/16) = 4 个 block
# block_tables["A"] = [7, 23, 1, 45]
#                       │   │   │   │
#                       │   │   │   └── token 48-49 (部分填充)
#                       │   │   └── token 32-47
#                       │   └── token 16-31
#                       └── token 0-15
```

### 8.2 Token 到 Block 的索引

给定一个 token 的**逻辑位置** `pos`，计算其在物理显存中的位置：

```python
def get_physical_location(pos, block_table, block_size):
    """将逻辑 token 位置映射到物理 block + offset。"""
    logical_block_idx = pos // block_size      # 逻辑 block 编号
    block_offset = pos % block_size            # block 内偏移
    
    physical_block_idx = block_table[logical_block_idx]  # 查表
    
    # 在 KV Cache 张量中的索引：
    # kv_cache[physical_block_idx, block_offset, :, :]
    return physical_block_idx, block_offset

# 例：pos=35, block_table=[7, 23, 1, 45], block_size=16
# logical_block_idx = 35 // 16 = 2
# block_offset = 35 % 16 = 3
# physical_block_idx = block_table[2] = 1
# → kv_cache[1, 3, :, :] 即为 token 35 的 KV 值
```

### 8.3 Block Table 在 GPU 上的表示

为了让 GPU kernel 能高效访问 Block Table，vLLM 将其打包为一个 2D 张量：

```python
# GPU 上的 Block Table 张量
# shape: [max_num_seqs, max_num_blocks_per_seq]
# dtype: torch.int32

block_tables_tensor = torch.tensor([
    [7, 23,  1, 45, -1, -1],   # 请求 A：4 个 block
    [3, 12, 56,  8, 19, -1],   # 请求 B：5 个 block
    [0, 11, -1, -1, -1, -1],   # 请求 C：2 个 block
], dtype=torch.int32, device="cuda")
# -1 表示未分配的 slot
```

---

## 9. 不同精度下的显存占用对比

### 9.1 数据类型与字节数

| 数据类型 | 每元素字节数 | 精度 | 常见用途 |
|---------|-----------|------|---------|
| FP32 | 4 bytes | 高 | 训练（少用于推理 KV Cache） |
| FP16 | 2 bytes | 中 | 推理默认 |
| BF16 | 2 bytes | 中 | 推理默认（更好的数值范围） |
| FP8 (E4M3) | 1 byte | 低 | 新一代推理优化 |
| INT8 | 1 byte | 低 | 量化推理 |
| INT4 | 0.5 bytes | 极低 | 激进量化 |

### 9.2 LLaMA-3-70B 各精度下的 KV Cache 大小

以 LLaMA-3-70B 为例（80 层，GQA 8 KV heads，head_dim=128），单个请求 4096 tokens：

$$
\text{KV}_\text{per\_request} = 2 \times L \times n_{kv} \times d_h \times s \times \text{bytes}
$$

| 精度 | bytes | 计算 | 每请求 KV Cache |
|------|-------|------|---------------|
| FP16 | 2 | $2 \times 80 \times 8 \times 128 \times 4096 \times 2$ | **1.25 GB** |
| BF16 | 2 | 同上 | **1.25 GB** |
| FP8 | 1 | $2 \times 80 \times 8 \times 128 \times 4096 \times 1$ | **0.625 GB** |
| INT4 | 0.5 | $2 \times 80 \times 8 \times 128 \times 4096 \times 0.5$ | **0.3125 GB** |

### 9.3 FP8 KV Cache 的实际应用

vLLM 支持 FP8 KV Cache 量化（`--kv-cache-dtype fp8`），在 Hopper (H100) 和 Ada (L40S) 架构上可用：

```bash
# 启用 FP8 KV Cache
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --kv-cache-dtype fp8 \
    --dtype bfloat16 \
    --tensor-parallel-size 4
```

FP8 KV Cache 的注意事项：
- **精度损失**：大多数场景下质量损失可忽略（<0.5% perplexity 退化）
- **显存收益**：2x KV Cache 容量，意味着 2x 并发或 2x 上下文长度
- **计算兼容**：Attention 计算仍在 FP16/BF16 下进行，仅存储和读取时使用 FP8

---

## 10. KV Cache 占模型总显存的比例

### 10.1 显存的四大组成

```
GPU 显存分配:
┌──────────────────────────────────────────────────┐
│                  Total GPU Memory                │
├──────────────────┬───────────────────────────────┤
│  Model Weights   │       KV Cache Pool          │
│   (固定)         │    (动态分配给请求)            │
├──────────────────┼───────────────────────────────┤
│  Activations     │       Other                   │
│  (临时)          │  (CUDA context, fragmentation)│
└──────────────────┴───────────────────────────────┘
```

### 10.2 典型场景分析

以 **LLaMA-3-70B on 4×H100-80GB**（TP=4）为例：

```
总显存: 4 × 80 GB = 320 GB (可用约 288 GB @ gpu_memory_utilization=0.9)

模型权重 (BF16): 70B × 2 bytes = 140 GB
  → 每卡: 140 / 4 = 35 GB

激活值 (峰值): ~2-4 GB / GPU

可用 KV Cache: 288 - 140 - 16 (激活+其他) ≈ 132 GB

KV Cache 每 token (所有层, 每卡):
  = 2 × (80/4) × 8 × 128 × 2 bytes (TP=4, 每卡 20 层)
  = 2 × 20 × 8 × 128 × 2 = 81,920 bytes ≈ 80 KB/token

最大 token 容量: 132 GB / 80 KB ≈ 1,650,000 tokens
  → 若 max_seq_len=4096: 约 400 并发请求
  → 若 max_seq_len=32768: 约 50 并发请求
```

### 10.3 KV Cache 显存占比随场景变化

| 场景 | 模型权重 | KV Cache | KV 占比 |
|------|---------|----------|---------|
| 小模型 + 短上下文 (7B, 2K) | ~14 GB | ~1 GB | ~7% |
| 大模型 + 短上下文 (70B, 4K) | ~140 GB | ~10 GB | ~7% |
| 大模型 + 长上下文 (70B, 128K) | ~140 GB | ~100+ GB | **~42%+** |
| 小模型 + 长上下文 (7B, 128K) | ~14 GB | ~50+ GB | **~78%+** |

**关键结论**：长上下文场景下，KV Cache 才是显存的主要瓶颈，远超模型权重本身。

---

## 11. 总结

| 概念 | 要点 |
|------|------|
| **连续 vs 分页** | 分页消除碎片，提升显存利用率从 ~20% 到 ~95% |
| **MHA 布局** | `[batch, num_heads, seq_len, head_dim]`，KV heads = Q heads |
| **GQA 布局** | `num_kv_heads << num_heads`，KV Cache 缩小数倍 |
| **MLA 布局** | 缓存低维 latent，不存原始 KV，进一步缩小数倍 |
| **Block Table** | 逻辑→物理映射，支持非连续存储和共享 |
| **DynamicCache** | torch.cat 追加，灵活但低效 |
| **StaticCache** | 预分配固定缓冲，适合 compile 和生产 |
| **精度** | FP8 比 FP16 节省 2x 显存，精度损失可控 |
| **显存占比** | 长上下文场景下 KV Cache 可占总显存 40-80% |

---

## 参考资料

- [Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [GQA: Training Generalized Multi-Query Attention (ICML 2023)](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM](https://arxiv.org/abs/2405.04434)
- [vLLM 源码 - gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)
- [HuggingFace Transformers - Cache 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
