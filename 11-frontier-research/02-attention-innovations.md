# Attention 架构创新

> 从 FlashAttention 到 FlashInfer，从 MHA 到 MLA——Attention 机制的演进如何重塑 Serving 系统

## 1. FlashInfer：下一代 Attention Kernel 库

### 1.1 为什么需要 FlashInfer？

FlashAttention（Tri Dao, 2022-2024）通过 tiling 和 kernel fusion 显著加速了 attention 计算，但它有几个限制：

1. **KV 布局固定**：FlashAttention 主要针对连续内存布局，不原生支持 PagedAttention 的 block table 间接寻址
2. **灵活性不足**：不同 serving 场景（prefill vs decode、单 request vs batch、不同 head 配置）需要不同的 kernel 变体
3. **与 serving 系统耦合不够**：FlashAttention 是通用 attention kernel，不针对 serving 场景优化

FlashInfer（由 CMU 的 Zihao Ye 等人开发）正是为了解决这些问题而生。它是一个**专为 LLM serving 设计的 attention kernel 库**。

### 1.2 核心特性

#### 多种 KV 布局支持

```
FlashInfer 支持的 KV 布局:

1. Ragged Tensor（连续存储，variable-length）
   ┌───────────────────────────────────────┐
   │ Seq0 KV │ Seq1 KV │ Seq2 KV │ ...   │
   └───────────────────────────────────────┘
   + indptr: [0, 100, 250, 380, ...]
   适用: Prefill 阶段, 不需要 paging

2. Paged KV Cache（block table 间接寻址）
   ┌──────┐ ┌──────┐ ┌──────┐
   │Blk 0 │ │Blk 1 │ │Blk 2 │  ...
   └──────┘ └──────┘ └──────┘
   + block_table: [[0, 3, 7], [1, 5, 8], ...]
   适用: Decode 阶段, PagedAttention

3. Paged KV Cache with Ragged Query（混合模式）
   Query: Ragged Tensor (prefill, variable-length)
   KV: Paged (已有 cache)
   适用: Chunked prefill, prefix caching hit 后的增量 prefill
```

#### Plan-Run 两阶段 API

FlashInfer 的独特设计——将 attention 计算分为 **Plan** 和 **Run** 两个阶段：

```python
import flashinfer

# 阶段 1: Plan (CPU 端, 提前规划)
# 计算各 request 的元信息, 生成 workspace
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
wrapper.plan(
    qo_indptr=qo_indptr,           # query 的 ragged indptr
    paged_kv_indptr=kv_indptr,     # KV 的 page table indptr
    paged_kv_indices=kv_indices,   # KV block 索引
    paged_kv_last_page_len=last_page_len,
    num_qo_heads=num_q_heads,
    num_kv_heads=num_kv_heads,
    head_dim=head_dim,
    page_size=page_size,
)

# 阶段 2: Run (GPU 端, 执行计算)
output = wrapper.run(query, paged_kv_cache)
```

**为什么要分两阶段？**

1. **与 CUDA Graph 兼容**：Plan 阶段在 CPU 执行，可以处理动态 shape；Run 阶段固定计算图，可以被 CUDA Graph 捕获
2. **减少 GPU 端分支**：Plan 阶段预计算好所有分发逻辑，Run 阶段的 kernel 无条件执行
3. **复用 workspace**：多次 Run 可以共享同一个 Plan 的结果（如果 batch 组成不变）

#### 高效 Decode Attention

FlashInfer 的 decode attention（单 token query vs 长 KV）使用了专门的优化：

```
标准 FlashAttention decode:
  每个 head 一个 thread block
  → 无法充分利用 GPU 并行度 (batch_size × num_heads 可能不够多)

FlashInfer decode (cascading):
  多个 thread block 协作处理一个 head
  → 更好的 GPU 利用率, 尤其在长序列时
  
  Split-KV 策略:
  ┌─────────┬─────────┬─────────┐
  │ KV 0-1K │ KV 1K-2K│ KV 2K-3K│  ← 不同 thread block 处理不同 KV 段
  └────┬────┴────┬────┴────┬────┘
       └─────────┼─────────┘
            Reduce (合并 partial softmax)
```

### 1.3 FlashInfer 在 SGLang 和 vLLM 中的集成

SGLang 从早期就使用 FlashInfer 作为默认 attention backend，vLLM v1 也开始将 FlashInfer 作为主要的 attention 实现之一：

```python
# vLLM 中选择 attention backend
# 通过环境变量或配置选择
VLLM_ATTENTION_BACKEND=FLASHINFER  # 使用 FlashInfer
VLLM_ATTENTION_BACKEND=FLASH_ATTN  # 使用 FlashAttention

# FlashInfer 提供了 vLLM 所需的所有接口:
# - prefill attention (ragged query + ragged/paged KV)
# - decode attention (single-token query + paged KV)
# - append attention (chunked prefill 场景)
```

## 2. Multi-head Latent Attention (MLA) 的后续发展

### 2.1 MLA 回顾

DeepSeek-V2 (2024) 提出的 MLA 通过低秩压缩 KV Cache 实现了显著的显存节省：

```
标准 GQA (LLaMA-3 70B):
  KV Cache per token per layer = 2 × 8 × 128 × 2B = 4KB (FP16)
  60 层 × 4KB = 240KB per token

MLA (DeepSeek-V2):
  KV Cache per token per layer = 512 × 2B = 1KB (FP16, 压缩维度)
  60 层 × 1KB = 60KB per token
  → 4× 压缩比!
```

MLA 的核心思路：

```
标准 MHA:
  K = X · W_K,  V = X · W_V  →  缓存 (K, V)
  
MLA:
  C = X · W_C  (低维压缩表示, d_c << n_h × d_h)
  K = C · W_UK, V = C · W_UV  (解码时动态恢复)
  →  只缓存 C, 大幅缩小缓存
```

### 2.2 MLA 在 Serving 中的挑战

**挑战 1：吸收 W_UK 到 Q 的变换中**

为了避免在 decode 时做 `C · W_UK` 的矩阵乘法（会抵消显存节省的收益），实际实现中将 W_UK 吸收到 Q 的投影中：

```python
# 朴素实现 (慢, 因为需要恢复 K):
Q = X_q · W_Q                    # (batch, n_heads, d_head)
K = KV_cache_C · W_UK            # (batch, seq_len, n_heads, d_head) ← 需要恢复!
attn = Q @ K.T / sqrt(d)

# 优化实现 (吸收 W_UK 到 Q):
Q_absorbed = X_q · W_Q · W_UK.T  # 预计算, 变换到压缩空间
C = KV_cache_C                    # 直接使用压缩表示
attn = Q_absorbed @ C.T / sqrt(d) # 在压缩空间做 attention
# → 不需要恢复 K, 直接用压缩 cache!
```

**挑战 2：与 FlashAttention 的兼容性**

标准 FlashAttention 假设 Q, K, V 具有相同的 head_dim，但 MLA 的 "吸收" 方法改变了维度关系。解决方案：

1. **自定义 attention kernel**：DeepSeek 团队编写了专用 kernel
2. **FlashInfer 支持**：FlashInfer 2024 年底开始支持自定义 head_dim 的 attention
3. **Triton 实现**：社区提供了 Triton 版本的 MLA attention

### 2.3 MLA 的后续采用者

2025 年以来，MLA 思路被更多模型采用（部分直接使用，部分受启发改进）：

| 模型/方法 | KV 压缩策略 | 与 MLA 的关系 |
|-----------|-------------|---------------|
| DeepSeek-V3 (2024.12) | MLA + MTP | 原版 MLA |
| MiniMax-01 (2025) | 类 MLA 的低秩压缩 | 受 MLA 启发 |
| 学术研究: LoRC (2025) | 低秩 KV 压缩 (post-training) | 泛化 MLA 到已有模型 |
| 学术研究: GoldFinch (2024) | Linear Attention + 压缩 KV | 不同技术路线，相似目标 |

### 2.4 MLA 对 Serving 系统的影响

```python
# MLA 模型的 KV Cache 管理需要特殊处理:

# 1. Block 大小不同
#    标准 MHA: block 存储 (block_size, 2, num_kv_heads, head_dim)
#    MLA: block 存储 (block_size, compressed_dim)
#    → block 更小, 同样显存可以存更多 token

# 2. Prefix caching 仍然有效
#    C = X · W_C 是 token 的确定性函数
#    → 相同前缀 → 相同 C → 可以共享

# 3. 量化效果不同
#    压缩后的 C 的数值分布与原始 KV 不同
#    → 需要重新评估量化策略 (FP8/INT8 for compressed cache)
```

## 3. Linear Attention 与 State Space Models

### 3.1 从 Attention 到 Linear Attention

标准 Attention 的计算复杂度是 O(n²)（n 为序列长度），Linear Attention 试图将其降低到 O(n)：

```
标准 Attention:
  Attn(Q, K, V) = softmax(QK^T / √d) · V
  复杂度: O(n² · d)

Linear Attention (概念):
  Attn(Q, K, V) = φ(Q) · (φ(K)^T · V)  
  其中 φ 是某个特征映射
  复杂度: O(n · d²)  ← 当 d << n 时更优
  
  关键: 可以写成递推形式!
  S_t = S_{t-1} + φ(k_t) · v_t^T   (state update, O(d²))
  o_t = φ(q_t) · S_t                 (output, O(d²))
  → 不需要存储所有历史 KV, 只需要维护一个 d×d 的 state!
```

### 3.2 Mamba 和 Mamba-2 的 Serving 特性

Mamba (Gu & Dao, 2023) 和 Mamba-2 (Dao & Gu, 2024) 是目前最成功的 SSM 架构：

```python
# Mamba 的 "KV Cache" 等价物

# 标准 Transformer decode:
#   输入: 新 token embedding x_t
#   需要: 所有历史 token 的 KV Cache (O(t) 空间)
#   输出: 新的 hidden state + 追加 KV 到 cache

# Mamba decode:
#   输入: 新 token embedding x_t
#   需要: SSM state (O(1) 空间, 不随 t 增长!)
#   输出: 新的 hidden state + 更新 SSM state (in-place)

# 具体而言, Mamba 的 state:
class MambaState:
    conv_state: Tensor  # (d_inner, d_conv)  ← 短卷积的缓存
    ssm_state: Tensor   # (d_inner, d_state) ← SSM 递推状态
    
    # 大小: d_inner × (d_conv + d_state)
    # 例: 4096 × (4 + 16) = 81920 个 float
    # ≈ 320KB per layer (float32)
    # 不随序列长度增长!
```

### 3.3 Mamba-2 的关键改进

Mamba-2 (SSD, Structured State Space Duality) 引入了一个关键洞察：**SSM 和 Attention 之间存在数学对偶**。

```
Mamba-2 对偶关系:
  SSM (递推形式): 适合 decode (逐 token 生成)
  Attention (矩阵形式): 适合 prefill (并行处理所有 token)
  
  Mamba-2 可以在两种形式之间切换!
  Prefill 时: 用 "chunk-wise" attention 形式, 高效并行
  Decode 时: 用递推形式, O(1) state
```

这对 serving 意味着：

1. **Prefill 不再是瓶颈**：可以利用 GPU 的并行计算能力
2. **Decode 极其高效**：O(1) state 更新，不需要读取长 KV Cache
3. **长序列优势巨大**：100K+ token 时，Mamba-2 的 decode 速度远超 Transformer

### 3.4 Hybrid Transformer-SSM 的 Serving 挑战

混合架构（如 Jamba）在 serving 时的特殊问题：

```
挑战 1: Prefill 阶段的计算模式不同
  Attention 层: 标准矩阵乘 + softmax (GPU 友好)
  Mamba 层: chunk-wise SSM (需要专用 kernel)
  → 需要在同一个 forward pass 中混合两种计算模式

挑战 2: Decode 阶段的内存访问模式不同
  Attention 层: 读取大量 KV Cache (memory-bound)
  Mamba 层: 读取小 state + 计算更新 (compute-bound)
  → batching 策略需要同时优化两种 bound

挑战 3: Speculative decoding 不直接适用
  标准 spec decode: draft model 生成 tokens → target model verify
  混合模型: SSM state 的 verify 逻辑不同于 KV Cache
  → 需要新的 spec decode 协议
```

## 4. FlashAttention-3 与 Hopper 特性

### 4.1 FlashAttention-3 的改进

FlashAttention-3 (Tri Dao, 2024) 专门针对 NVIDIA Hopper 架构（H100/H200）优化：

```
关键优化:

1. Warp Specialization (warp 特化)
   ┌─────────────────────────────┐
   │ Warp 0-1: Producer (加载数据) │
   │ Warp 2-3: Consumer (计算)    │
   └─────────────────────────────┘
   → 加载和计算可以完全重叠 (pipeline)

2. FP8 支持
   H100 的 FP8 Tensor Core 提供 2× FLOPS (vs FP16)
   FlashAttention-3 + FP8 → 接近峰值性能
   
   FP8 注意力:
   Q_fp8 = quantize(Q)  # FP16 → FP8
   K_fp8 = quantize(K)
   S = Q_fp8 @ K_fp8^T  # FP8 tensor core, 高吞吐
   P = softmax(S)        # 仍用 FP32 做 softmax (精度敏感)
   O = P @ V             # 可以用 FP8 或 FP16

3. TMA (Tensor Memory Accelerator) 利用
   H100 的 TMA 可以异步加载多维 tensor
   → 进一步减少数据搬运延迟
   
4. Pingpong scheduling
   在两个 warpgroup 之间交替执行
   → 隐藏 softmax 的 non-matmul 计算延迟
```

### 4.2 性能对比

```
H100 SXM 80GB 上的 Attention 吞吐 (seq_len=8K, head_dim=128):

FlashAttention-2 (FP16):  ~330 TFLOPS
FlashAttention-3 (FP16):  ~510 TFLOPS  (1.55×)
FlashAttention-3 (FP8):   ~740 TFLOPS  (2.24×)
H100 理论峰值 (FP8):       989 TFLOPS
→ FlashAttention-3 FP8 达到 ~75% 峰值利用率
```

### 4.3 对 Serving 的影响

```python
# FlashAttention-3 + FP8 对 serving 的意义:

# 1. Prefill 速度大幅提升
#    FP8 attention 将 prefill TTFT 降低 30-50%
#    长 prompt 场景收益更大

# 2. KV Cache 可以用 FP8 存储
#    显存占用减半 (vs FP16)
#    且 attention 计算无需反量化 (直接 FP8 Tensor Core)

# 3. 与 KV Cache 量化的协同
#    Ch03 讨论的 KIVI/KVQuant 方案可以与 FP8 attention 结合:
#    存储: INT4/FP8 量化 KV Cache
#    计算: FP8 Tensor Core
#    → 4× 显存压缩 + 2× 计算加速
```

## 5. Sliding Window Attention 的特殊处理

### 5.1 Sliding Window 的 Serving 优化

Sliding Window Attention (SWA) 限制每个 token 只关注最近 W 个 token，这对 serving 有重要影响：

```
Full Attention (seq_len = 32K):
  每个 decode step 需要读取 32K 个 token 的 KV Cache
  → memory bandwidth 是瓶颈

Sliding Window (W = 4096):
  每个 decode step 最多读取 4096 个 token 的 KV Cache
  → 减少 8× memory read
  → decode latency 显著降低
```

### 5.2 KV Cache 管理优化

```python
# Sliding Window 的 KV Cache 可以使用环形缓冲:

class SlidingWindowKVCache:
    def __init__(self, window_size: int, num_heads: int, head_dim: int):
        self.window_size = window_size
        # 预分配固定大小的缓冲区
        self.k_cache = torch.zeros(window_size, num_heads, head_dim)
        self.v_cache = torch.zeros(window_size, num_heads, head_dim)
        self.position = 0  # 环形指针
    
    def append(self, k: Tensor, v: Tensor):
        # 写入当前位置
        idx = self.position % self.window_size
        self.k_cache[idx] = k
        self.v_cache[idx] = v
        self.position += 1
    
    # 优势:
    # 1. 不需要 PagedAttention → 无 block table 开销
    # 2. 显存使用恒定 → 不会 OOM
    # 3. Memory access 模式更规律 → cache 友好
```

### 5.3 混合 Full + Sliding Window

Mistral/Mixtral 风格的模型混合使用 full attention 和 sliding window attention：

```
Layer 0:  Full Attention    → 需要完整 KV Cache
Layer 1:  Sliding Window    → 只需要最近 W 个 token
Layer 2:  Full Attention    → 需要完整 KV Cache
Layer 3:  Sliding Window    → 只需要最近 W 个 token
...

显存节省:
  如果 50% 层是 sliding window (W=4096), seq_len=32K:
  Full attention 层: 32K tokens × 50% layers
  Sliding window 层: 4K tokens × 50% layers
  总显存 = 50% × 32K + 50% × 4K = 18K (vs 32K pure full)
  → 节省 ~44% KV Cache 显存
```

## 6. Cross Attention 的 Serving 处理

### 6.1 Encoder-Decoder 模型的 Cross Attention

对于 encoder-decoder 架构（如 T5, Whisper）以及 Vision-Language 模型中的 cross attention：

```
Cross Attention 特点:
  Q: 来自 decoder (逐 token 生成)
  K, V: 来自 encoder output (固定, 不变)
  
  → K, V 在生成过程中不会增长!
  → 可以一次性计算并缓存 encoder 的 KV
  → 不需要 PagedAttention (大小固定)
```

### 6.2 VLM (Vision-Language Model) 的 Cross Attention

现代 VLM（如 LLaVA-OneVision, Qwen-VL2）的 cross attention 处理：

```python
# 图像 token 的 KV Cache 特殊性:

# 1. 图像 token 数量固定 (例如 576 个 patch token)
#    可以预分配, 不需要动态管理

# 2. 多个 request 可能使用相同图像
#    → 图像 KV Cache 可以共享 (类似 prefix caching)

# 3. 图像 KV Cache 通常较大
#    高分辨率图像: 2048+ patch tokens
#    多图: 每图 576 tokens × 多图
#    → 可能占 KV Cache 总量的 50%+
```

## 7. 各方案对比总结

| 特性 | FlashAttention-2 | FlashAttention-3 | FlashInfer | xFormers |
|------|-----------------|-----------------|------------|----------|
| 硬件支持 | Ampere+ | Hopper | Ampere+ | 广泛 |
| FP8 支持 | 否 | 是 | 是 | 部分 |
| Paged KV | 社区 fork | 社区 fork | 原生支持 | 部分 |
| Plan-Run API | 否 | 否 | 是 | 否 |
| CUDA Graph 友好 | 一般 | 改进 | 好 | 一般 |
| MLA 支持 | 需修改 | 需修改 | 开发中 | 否 |
| Serving 集成 | vLLM 默认 | 实验中 | SGLang 默认 | 早期 vLLM |

## 8. 实践建议

### 对 Serving 系统开发者

1. **优先采用 FlashInfer**：如果你在开发新的 serving 系统，FlashInfer 的 Plan-Run API 和丰富的 KV 布局支持是更好的选择
2. **关注 FP8 Attention**：H100/H200 上，FP8 attention 是性能提升的最大杠杆
3. **为混合 attention 做准备**：未来模型很可能混合使用 full/sliding window/linear attention

### 对模型部署者

1. **根据硬件选择 backend**：H100 上用 FlashAttention-3 或 FlashInfer；A100 上用 FlashAttention-2 或 FlashInfer
2. **MLA 模型（DeepSeek 系列）**：确认 serving 框架对 MLA 的支持情况，优先选择有优化 kernel 的版本
3. **Sliding window 模型**：确认 KV Cache 管理正确利用了 window 限制来节省显存

## 9. 小结

Attention 架构的创新正在多个方向同时推进：

1. **Kernel 层面**：FlashInfer 提供了更灵活的 serving 优化接口，FlashAttention-3 针对 Hopper 架构极致优化
2. **架构层面**：MLA 大幅压缩 KV Cache，Linear Attention/SSM 提供 O(1) decode
3. **混合层面**：Transformer-SSM 混合架构结合两者优势，但增加了 serving 复杂度
4. **硬件层面**：FP8 支持和 Hopper 特性（TMA, warp specialization）带来新的优化空间

这些创新对 serving 系统的影响是全方位的——从 kernel 实现到内存管理到调度策略，都需要适应新的 attention 范式。
