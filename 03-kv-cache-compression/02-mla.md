# MLA: Multi-head Latent Attention 深度解读

> 从架构层面重新设计 attention，将 KV Cache 压缩到传统 MHA 的 ~1/9。

## 1. 设计动机

### 1.1 KV Cache 的根本问题

在标准 Multi-Head Attention (MHA) 中，每个 token 需要缓存完整的 K 和 V 向量：

```
每个 token 每层的 KV Cache = 2 × n_h × d_h × sizeof(dtype)

以 LLaMA-65B 为例：
  n_h = 64 (attention heads)
  d_h = 128 (head dimension)
  n_layers = 80
  dtype = FP16 (2 bytes)
  
  每 token KV Cache = 2 × 64 × 128 × 2 = 32,768 bytes = 32 KB/token/layer
  所有层合计 = 32 × 80 = 2,560 KB ≈ 2.5 MB/token
```

当服务大量并发用户时（每个用户有数千 token 的上下文），KV Cache 的显存消耗会迅速超过模型权重本身。

### 1.2 已有的压缩方案及其局限

在 MLA 之前，主流的 KV Cache 缩减方案是 **GQA（Grouped-Query Attention）**：

```
MHA:  n_kv_heads = n_h           → 64 个 KV heads
GQA:  n_kv_heads = n_h / g       → 8 个 KV heads (g=8, 如 LLaMA-3)
MQA:  n_kv_heads = 1             → 1 个 KV head
```

GQA 通过减少 KV heads 的数量来减少 KV Cache，但这是一种**有损**的方法——多个 Query heads 共享同一组 KV，表达能力有所下降。DeepSeek 团队思考了一个更本质的问题：**能不能在不损失表达能力的前提下，大幅压缩 KV Cache？**

### 1.3 低秩投影的直觉

MLA 的核心直觉来自一个观察：**K 和 V 矩阵本身包含大量冗余信息**。如果我们将高维的 K、V 投影到一个低维的 latent space，然后在推理时从 latent vector 重建 K 和 V，那么我们只需要缓存低维的 latent vector 即可。

```
传统 MHA：缓存 K 和 V 的完整表示
          cache = [K, V]  →  维度: 2 × n_h × d_h

MLA：缓存低维 latent vector
     cache = c_t^KV       →  维度: d_c  (d_c << 2 × n_h × d_h)
     推理时从 c_t^KV 重建 K 和 V
```

## 2. MLA 的数学表达

### 2.1 标准 MHA 回顾

在标准 MHA 中，对于输入 hidden state $h_t \in \mathbb{R}^d$：

```
Q = h_t × W_Q    ∈ R^{n_h × d_h}
K = h_t × W_K    ∈ R^{n_h × d_h}
V = h_t × W_V    ∈ R^{n_h × d_h}

KV Cache 存储: [K_1, K_2, ..., K_t], [V_1, V_2, ..., V_t]
每 token 存储量: 2 × n_h × d_h
```

### 2.2 MLA 的 KV 低秩压缩

MLA 引入一个 **down-projection** 将 K 和 V 联合压缩到低维 latent space：

```
Step 1: Down-projection（推理时执行，结果缓存）
  c_t^KV = h_t × W_DKV    ∈ R^{d_c}
  其中 d_c << 2 × n_h × d_h（压缩维度）

Step 2: Up-projection（推理时从缓存重建）
  K_t = c_t^KV × W_UK    ∈ R^{n_h × d_h}
  V_t = c_t^KV × W_UV    ∈ R^{n_h × d_h}
```

关键洞察：**只需要缓存 $c_t^{KV}$，不需要缓存完整的 K 和 V**。

```
KV Cache 大小对比：
  MHA:  2 × n_h × d_h = 2 × 128 × 128 = 32,768 per token per layer
  MLA:  d_c            = 512            per token per layer (DeepSeek-V2 配置)
  
  压缩比 = 32,768 / 512 = 64x （纯 latent 部分）
```

### 2.3 Query 也做低秩压缩

MLA 同样对 Query 做低秩压缩以减少计算量（虽然 Q 不需要缓存，但压缩 Q 可以减少上投影的计算）：

```
c_t^Q = h_t × W_DQ    ∈ R^{d_c'}    (d_c' 是 Q 的压缩维度)
Q_t = c_t^Q × W_UQ    ∈ R^{n_h × d_h}
```

### 2.4 Attention 计算中的吸收技巧

在实际计算中，MLA 利用**矩阵乘法结合律**避免显式重建完整的 K 和 V：

```
标准计算：
  score = Q_t × K_s^T = (c_t^Q × W_UQ) × (c_s^KV × W_UK)^T
        = c_t^Q × W_UQ × W_UK^T × (c_s^KV)^T

吸收技巧：
  令 W_absorbed = W_UQ × W_UK^T    (可以预计算)
  score = c_t^Q × W_absorbed × (c_s^KV)^T

  类似地，output 计算：
  output = softmax(scores) × V
         = softmax(scores) × c_s^KV × W_UV
  令 W_absorbed_v = W_UV    (直接在 attention 输出后乘)
```

这意味着 attention kernel **直接在 latent space 上操作**，不需要将 $c^{KV}$ 还原为完整的 K 和 V，进一步减少了计算量和内存带宽。

## 3. Decoupled RoPE：位置编码的特殊处理

### 3.1 问题：RoPE 与低秩压缩不兼容

Rotary Position Embedding (RoPE) 将位置信息通过旋转矩阵编码到 K 和 Q 中：

```
K_rotated = RoPE(K, position)
Q_rotated = RoPE(Q, position)

score = Q_rotated × K_rotated^T  → 包含相对位置信息
```

问题在于：RoPE 是一个 position-dependent 的变换。如果我们只缓存 $c^{KV}$（latent vector），那么在重建 K 时还需要知道该 token 的位置来应用 RoPE。但 **位置信息无法被压缩到 latent space 中**，因为不同位置的同一个 token 应该有不同的 K 值。

### 3.2 解决方案：Decoupled RoPE

MLA 的解决方案是将 K 分为两部分：

```
K_t = [K_t^C, K_t^R]

K_t^C = c_t^KV × W_UK    ← 内容相关，从 latent vector 重建（不带位置编码）
K_t^R = h_t × W_KR       ← 位置相关，单独计算并应用 RoPE

其中：
  K_t^C ∈ R^{n_h × d_h}     内容 Key
  K_t^R ∈ R^{n_h × d_h^R}   位置 Key (d_h^R 通常远小于 d_h)
```

同样，Query 也分为两部分：

```
Q_t = [Q_t^C, Q_t^R]

Q_t^C = c_t^Q × W_UQ     ← 内容 Query
Q_t^R = c_t^Q × W_QR     ← 位置 Query，应用 RoPE
```

Attention score 的计算变为：

```
score(t, s) = [Q_t^C, Q_t^R] × [K_s^C, K_s^R]^T
            = Q_t^C × (K_s^C)^T + Q_t^R × (K_s^R)^T
              ↑ 内容相关性            ↑ 位置相关性
```

### 3.3 KV Cache 的最终组成

因为 $K_t^R$ 需要包含位置信息且无法从 $c_t^{KV}$ 中恢复，所以 KV Cache 需要额外存储 RoPE Key：

```
MLA KV Cache per token per layer = d_c + d_h^R

DeepSeek-V2 的配置：
  d_c = 512        (latent vector)
  d_h^R = 64       (RoPE head dim, 所有 heads 共享，只存一份)
  
  总计 = 512 + 64 = 576 per token per layer
  对比 MHA = 2 × 128 × 128 = 32,768 per token per layer
  
  实际压缩比 ≈ 32,768 / 576 ≈ 57x
```

> 注意：DeepSeek-V2 的 RoPE Key 在所有 heads 之间共享（$K^R$ 是 multi-query 风格的），因此只存储一份 $d_h^R = 64$ 维的向量。

## 4. 具体数字：MHA vs MLA

### 4.1 DeepSeek-V2-236B 的 MLA 配置

```
模型参数：
  hidden_dim (d) = 5120
  n_heads (n_h) = 128
  head_dim (d_h) = 128
  n_layers = 60
  kv_lora_rank (d_c) = 512
  qk_rope_head_dim (d_h^R) = 64
  dtype = FP16 (2 bytes)
```

### 4.2 每 token 的 KV Cache 对比

```
假设同规模的 MHA 模型：
  KV Cache/token/layer = 2 × 128 × 128 × 2 = 65,536 bytes = 64 KB
  KV Cache/token (所有层) = 64 × 60 = 3,840 KB ≈ 3.75 MB

DeepSeek-V2 MLA：
  KV Cache/token/layer = (512 + 64) × 2 = 1,152 bytes ≈ 1.125 KB
  KV Cache/token (所有层) = 1.125 × 60 = 67.5 KB

压缩比 = 3,840 / 67.5 ≈ 57x
```

### 4.3 实际部署的显存影响

```
场景：batch_size=128, seq_len=4096

MHA (等效 236B 模型):
  KV Cache = 128 × 4096 × 3.75 MB ≈ 1,920 GB  ← 完全不可行（单机）

MLA (DeepSeek-V2-236B):
  KV Cache = 128 × 4096 × 67.5 KB ≈ 33.75 GB  ← 单个 8×H100 节点可承受

实际对比 GQA-8 (如 LLaMA-3-70B):
  KV Cache/token = 2 × 8 × 128 × 80 × 2 = 327,680 bytes ≈ 320 KB
  KV Cache = 128 × 4096 × 320 KB ≈ 160 GB
  
MLA 相对 GQA-8: 160 / 33.75 ≈ 4.7x 进一步压缩
```

### 4.4 论文中的数据对比

DeepSeek-V2 论文中给出了更精确的对比（以每 token 每层的 KV Cache 元素数为单位）：

| 模型 | Attention 类型 | KV Cache 元素数/token/layer | 相对 MHA |
|------|---------------|---------------------------|---------|
| 等效 MHA | MHA-128 heads | 32,768 | 1.0x |
| LLaMA-3-70B | GQA-8 | 2,048 | 16x 压缩 |
| DeepSeek-V2-236B | MLA | 576 | 57x 压缩 |

## 5. vLLM 中的 MLA 实现

### 5.1 实现路径

vLLM 对 MLA 的支持主要在以下文件中：

```
vllm/
├── model_executor/
│   └── layers/
│       └── mla.py              # MLA attention layer 实现
├── model_executor/
│   └── models/
│       └── deepseek_v2.py      # DeepSeek-V2 模型定义
├── attention/
│   └── backends/
│       └── flash_attn.py       # FlashAttention backend（需支持 MLA）
└── worker/
    └── cache_engine.py         # KV Cache 分配（按 MLA 的维度分配）
```

### 5.2 核心实现逻辑

```python
# 简化的 MLA forward 逻辑
class MLAAttention(nn.Module):
    def __init__(self, config):
        self.kv_lora_rank = config.kv_lora_rank        # d_c = 512
        self.qk_rope_head_dim = config.qk_rope_head_dim  # d_h^R = 64
        self.num_heads = config.num_attention_heads       # n_h = 128
        self.head_dim = config.head_dim                   # d_h = 128
        
        # Down-projection: h_t → c_t^KV
        self.kv_down_proj = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank,  # 5120 → 512
            bias=False
        )
        
        # Up-projection: c_t^KV → [K^C, V]
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.head_dim + self.head_dim),  # 512 → 128*(128+128)
            bias=False
        )
        
        # RoPE Key 的独立投影
        self.k_rope_proj = nn.Linear(
            config.hidden_size,
            self.qk_rope_head_dim,  # 5120 → 64
            bias=False
        )
    
    def forward(self, hidden_states, kv_cache, positions):
        # 1. 计算 latent vector（需要缓存的部分）
        c_kv = self.kv_down_proj(hidden_states)  # [batch, seq, 512]
        
        # 2. 计算 RoPE key（需要缓存的部分）
        k_rope = self.k_rope_proj(hidden_states)  # [batch, seq, 64]
        k_rope = apply_rope(k_rope, positions)
        
        # 3. 写入 KV Cache: 只存 [c_kv, k_rope]
        #    总维度 = 512 + 64 = 576（远小于 MHA 的 32768）
        cached = torch.cat([c_kv, k_rope], dim=-1)
        write_to_cache(cached, kv_cache)
        
        # 4. 从 cache 中读取所有历史 token 的 [c_kv, k_rope]
        all_c_kv, all_k_rope = read_from_cache(kv_cache)
        
        # 5. Up-project 重建完整的 K^C 和 V
        kv_full = self.kv_up_proj(all_c_kv)  # [batch, total_seq, n_h*(d_h+d_h)]
        k_content, v = kv_full.split(
            [self.num_heads * self.head_dim, self.num_heads * self.head_dim],
            dim=-1
        )
        
        # 6. 组合 K = [K^C, K^R]
        k = torch.cat([k_content, all_k_rope.expand(...)], dim=-1)
        
        # 7. 执行 attention
        output = flash_attention(q, k, v)
        return output
```

### 5.3 吸收优化在推理中的应用

在实际的高性能实现中，不会显式地执行 up-projection。取而代之的是使用**权重吸收 (weight absorption)** 技巧：

```python
# 推理优化：预计算吸收后的权重
# 在模型加载时执行一次
W_absorbed_QK = W_UQ @ W_UK.T  # [n_h*d_h, d_c] → 预计算
W_absorbed_VO = W_UV @ W_O     # [d_c, d] → 预计算

# 推理时的 attention 计算
# 不需要显式重建 K 和 V
def absorbed_attention(c_q, c_kv_cache, k_rope_cache, q_rope):
    # 内容 attention score: 直接在 latent space 计算
    # score_content = c_q @ W_absorbed_QK @ c_kv^T
    score_content = einsum('bhd,bhsd->bhs', 
                           c_q @ W_absorbed_QK, c_kv_cache)
    
    # 位置 attention score
    score_position = einsum('bhd,bhsd->bhs', q_rope, k_rope_cache)
    
    # 总 score
    scores = score_content + score_position
    attn_weights = softmax(scores / sqrt(d_h))
    
    # 输出：在 latent space 做加权求和，然后一次性投影
    latent_output = einsum('bhs,bhsd->bhd', attn_weights, c_kv_cache)
    output = latent_output @ W_absorbed_VO  # 一次矩阵乘法代替 up-proj + output_proj
    
    return output
```

这种优化使得 MLA 在 decode 阶段的计算效率甚至可以优于标准 MHA，因为：
1. 从 HBM 读取的 KV Cache 数据量大幅减少（memory bandwidth 节省）
2. 吸收后的计算量并不增加太多

## 6. FlashMLA：专用 Attention Kernel

### 6.1 为什么需要专用 kernel

标准的 FlashAttention kernel 假设 KV Cache 的 layout 是 `[batch, seq, num_kv_heads, head_dim]`。但 MLA 的 KV Cache 存储的是 latent vector `[batch, seq, d_c + d_h^R]`，layout 完全不同。

此外，吸收优化需要 kernel 内部执行额外的矩阵乘法（latent space 的投影），这不是标准 FlashAttention 支持的操作。

### 6.2 FlashMLA 的设计

FlashMLA 是 DeepSeek 团队开发的专用 attention kernel，关键特点：

```
标准 FlashAttention:
  输入: Q [B, H, 1, D], K [B, H, S, D], V [B, H, S, D]
  计算: softmax(Q @ K^T / sqrt(D)) @ V

FlashMLA:
  输入: Q_latent [B, H, 1, d_c'], Q_rope [B, H, 1, d_rope],
        KV_latent [B, 1, S, d_c], K_rope [B, 1, S, d_rope]
  计算: 
    score = Q_latent @ W_abs @ KV_latent^T + Q_rope @ K_rope^T
    output = softmax(score / sqrt(D)) @ KV_latent  (后续外部乘 W_abs_v)
```

FlashMLA 的核心优化：
- **Paged KV Cache 支持**：兼容 vLLM 的 PagedAttention 内存管理
- **异构 layout**：KV Cache 中 latent 部分和 RoPE 部分有不同维度
- **Fused 操作**：将吸收矩阵的乘法融入 attention kernel

## 7. MLA vs GQA：设计哲学差异

### 7.1 压缩策略对比

```
GQA: 减少 KV heads 的数量
     "fewer heads, each with full dimension"
     KV Cache = n_kv_heads × d_h
     
     多个 Q heads 共享同一组 KV → 跨 head 信息共享
     每个 KV head 保留完整的 d_h 维表示

MLA: 压缩每个 token 的 KV 表示维度
     "all heads from a shared low-rank latent"
     KV Cache = d_c (+ d_rope)
     
     所有 heads 的 KV 从同一个 latent vector 重建 → 更细粒度的信息共享
     latent vector 包含生成所有 heads KV 所需的信息
```

### 7.2 表达能力

| 维度 | GQA | MLA |
|------|-----|-----|
| K 的自由度 | n_kv_heads × d_h | d_c（通过 W_UK 映射到 n_h × d_h）|
| V 的自由度 | n_kv_heads × d_h | d_c（通过 W_UV 映射到 n_h × d_h）|
| 跨 head 关系 | 同组内强制相同 | 通过 latent space 隐式编码 |
| 训练灵活性 | 需要预定义分组 | 端到端学习最优表示 |

MLA 的优势在于：即使 latent dimension $d_c$ 很小，通过学习到的 $W_{UK}$ 和 $W_{UV}$，理论上可以为每个 head 生成不同的 K 和 V 表示。而 GQA 中同组的 heads 必须共享完全相同的 K 和 V。

### 7.3 工程实现复杂度

| 维度 | GQA | MLA |
|------|-----|-----|
| Kernel 支持 | FlashAttention 原生支持 | 需要专用 FlashMLA kernel |
| 框架兼容性 | 所有推理框架支持 | 需要框架适配（vLLM、SGLang） |
| Prefix Caching | 标准 KV hash 即可 | latent vector 的 hash 需要特殊处理 |
| 量化兼容 | 标准量化方法适用 | latent space 的量化特性待研究 |

## 8. MLA 对 Prefix Caching 的影响

### 8.1 Hash 兼容性

Prefix Caching 的核心是对相同 prefix 的 KV Cache 进行 hash 匹配。对于 MLA：

```
标准 MHA/GQA Prefix Caching:
  hash(token_ids, layer_idx) → KV Cache block
  KV block 格式: [K_block, V_block]

MLA Prefix Caching:
  hash(token_ids, layer_idx) → Latent Cache block
  Cache block 格式: [c_KV_block, K_rope_block]
```

MLA 的 prefix cache 存储 latent vector 而非完整 KV，这意味着：
1. **缓存块更小**：每个 block 的数据量大幅减少
2. **Hash 计算不变**：hash 只依赖 token_ids 和 layer_idx，与 cache 内容格式无关
3. **Cache hit 时的收益更大**：避免了 down-projection 的计算（虽然开销不大）

### 8.2 性能影响

```
Cache miss 时的额外开销：
  MHA:  计算 K, V（两个线性投影）→ 写入 cache
  MLA:  计算 c_KV, K_rope（down-projection + RoPE proj）→ 写入 cache
        然后 attention 时需要吸收或 up-project
  
Cache hit 时：
  MHA:  直接读取 K, V → attention
  MLA:  读取 c_KV, K_rope → 吸收优化下直接 attention
  
由于 MLA cache 更小，cache hit 率可能更高（同样的显存能缓存更多 prefix）
```

## 9. DeepSeek-V3 中 MLA 的进一步发展

### 9.1 架构延续

DeepSeek-V3 (2024.12) 延续了 MLA 架构，并在以下方面做了改进：

```
DeepSeek-V2 → DeepSeek-V3 的 MLA 变化：
  模型规模: 236B → 671B (MoE)
  n_heads: 128 → 128（保持）
  kv_lora_rank: 512 → 512（保持）
  qk_rope_head_dim: 64 → 64（保持）
  
  KV Cache/token/layer: 576 × 2 bytes = 1,152 bytes（与 V2 相同）
```

### 9.2 与 Multi-Token Prediction (MTP) 的配合

DeepSeek-V3 引入了 MTP（Multi-Token Prediction），在每一步预测多个后续 token。MLA 与 MTP 的配合优势：

- MTP 生成的多个候选 token 可以共享 prefix 的 MLA latent cache
- MLA 的小 cache footprint 使得 MTP 的额外 KV Cache 开销可控
- 验证阶段可以并行处理多个候选 token 的 attention

### 9.3 MLA 的局限性与未来方向

1. **训练效率**：MLA 的 down-projection 和 up-projection 增加了训练时的计算量和通信量
2. **Kernel 优化空间**：FlashMLA 的优化仍在进行中，目前效率低于成熟的 FlashAttention
3. **生态兼容性**：需要推理框架和硬件的专门支持
4. **量化交互**：latent space 的量化特性（是否更适合/更不适合量化）尚需更多研究

## 10. 总结

### 10.1 MLA 的核心价值

```
MLA 的本质：用计算换存储（和带宽）

传统方法（GQA）：减少冗余 KV heads     → 8-16x 压缩
MLA 方法：      低秩压缩 KV 表示       → 50-60x 压缩
                + Decoupled RoPE       → 保留位置编码能力
                + Weight Absorption    → 优化实际计算效率
```

### 10.2 适用场景

| 场景 | MLA 优势 | 注意事项 |
|------|---------|---------|
| 大 batch 推理 | KV Cache 小 → 支持更大 batch | 需要 FlashMLA kernel 支持 |
| 长上下文 | 同等显存下支持更长序列 | 仍需配合 prefix caching |
| 多并发用户 | 每用户 KV 占用极小 | 模型需原生支持 MLA |
| 低资源部署 | 显存需求大幅降低 | 模型选择受限（目前仅 DeepSeek 系列） |

---

> **下一节**：[选择性缓存与上下文压缩](03-selective-caching.md) — 通过识别重要 token 来减少 KV Cache。
