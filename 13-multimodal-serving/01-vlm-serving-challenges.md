# VLM 推理挑战

> 当图像变成 token，一切关于显存和延迟的假设都需要重新审视

## 1. VLM 架构回顾

### 1.1 基本流水线

Vision-Language Model (VLM) 的核心架构可以概括为三个阶段：

```
Image ──→ [Vision Encoder (ViT)] ──→ [Projector] ──→ Visual Tokens ──→ ┐
                                                                        ├──→ [LLM] ──→ Output
Text  ──→ [Tokenizer]           ──→ Text Tokens   ──→────────────────→ ┘
```

各组件的职责：

| 组件 | 功能 | 典型实现 |
|------|------|----------|
| Vision Encoder | 将图像编码为特征向量序列 | ViT-L/14, InternViT-6B, SigLIP |
| Projector | 将视觉特征映射到 LLM 的 embedding 空间 | MLP (2 层), Cross-Attention, Q-Former |
| LLM | 接收混合的 visual + text tokens，生成文本 | LLaMA, Qwen, InternLM |

### 1.2 视觉 token 的数量

一张图像经过 Vision Encoder 后，会产生**数百到数千个**视觉 token。这个数量取决于图像分辨率和 ViT 的 patch size：

$$N_{visual} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$

其中 $H, W$ 是图像分辨率，$P$ 是 patch size（通常为 14 或 16）。

**典型模型的视觉 token 数量：**

| 模型 | 图像分辨率 | Patch Size | 视觉 Token 数/图 |
|------|-----------|------------|------------------|
| LLaVA-1.5 | 336×336 | 14 | 576 |
| Qwen-VL | 448×448 | 14 | 1024 |
| InternVL-1.5 | 448×448 (per tile) | 14 | 1024 × tile 数 |
| LLaVA-NeXT | 动态分辨率 | 14 | 576–2880 |

!!! warning "关键洞察"
    一张图像的视觉 token 数量往往**超过大多数用户 prompt 的文本 token 数**。一张 336×336 的图像就有 576 个 token，而一个典型的用户问题可能只有 20–50 个 text token。

### 1.3 从 serving 视角看差异

与纯文本 LLM 相比，VLM 在 serving 层面引入了三个根本性挑战：

1. **Prefill 计算量爆炸** — 数百个视觉 token 意味着更长的 prefill
2. **KV Cache 显存压力** — 每个视觉 token 都会在每一层生成 KV Cache
3. **动态不可预测性** — 不同图像分辨率产生不同数量的视觉 token

下面逐一深入分析。

---

## 2. Prefill 计算量爆炸

### 2.1 视觉 token 的 prefill 代价

VLM 的 prefill 由两个串行阶段组成：

```
阶段 1: ViT Encoding    — 将图像编码为视觉特征 (compute-bound)
阶段 2: LLM Prefill     — 处理 [visual tokens + text tokens] (compute-bound)
```

**阶段 1：ViT Encoding** 是纯文本 LLM 中不存在的额外开销。以 ViT-L/14 (304M params) 为例：

```
ViT-L/14 encoding latency (A100):
  336×336 (576 patches):   ~5 ms
  448×448 (1024 patches):  ~8 ms
  672×672 (2304 patches):  ~18 ms
```

单独看 ViT encoding 似乎不贵，但它是串行开销——必须等 ViT 完成后，视觉 token 才能进入 LLM。

**阶段 2：LLM Prefill** 才是真正的瓶颈。视觉 token 在 LLM 看来和文本 token 无异，它们参与完整的 Self-Attention 计算。一个包含 576 个视觉 token + 30 个文本 token 的请求，其 LLM prefill 等价于一个 606 token 的纯文本 prefill。

### 2.2 计算量对比

我们来量化 VLM 请求 vs 纯文本请求的 prefill 计算量差异。

**场景对比：**

| 请求类型 | 输入 Token 数 | Prefill 计算量（相对值）|
|---------|--------------|----------------------|
| 纯文本：短问答 | 30 | 1× |
| 纯文本：带 system prompt | 200 | ~6.7× |
| VLM：单图 + 短问 (LLaVA) | 576 + 30 = 606 | ~20× |
| VLM：单图 + 短问 (Qwen-VL) | 1024 + 30 = 1054 | ~35× |
| VLM：4 tiles + 短问 (InternVL) | 4096 + 30 = 4126 | ~138× |

Prefill 的计算复杂度中，Self-Attention 部分为 $O(n^2 \cdot d)$，FFN 部分为 $O(n \cdot d^2)$。在 token 数量 $n$ 较大时，Attention 的二次项开始主导：

$$\text{FLOPs}_{prefill} \approx 2 \times n \times d_{model}^2 \times L \times 12 + 2 \times n^2 \times d_{model} \times L$$

其中前一项是 FFN+QKV projection 的贡献，后一项是 Attention 的贡献。

### 2.3 实际延迟影响

以 LLaMA-3-8B 在 A100 上的典型数据为例：

```
纯文本 30 tokens prefill:     ~3 ms
VLM 606 tokens prefill:       ~45 ms   (15x slower)
VLM 1054 tokens prefill:      ~85 ms   (28x slower)
VLM 4126 tokens prefill:      ~380 ms  (127x slower)
```

!!! danger "核心问题"
    一个 VLM 请求的 prefill 时间可以达到纯文本请求的 **10-100 倍**。在 continuous batching 场景下，这意味着一个 VLM 请求的 prefill 会严重阻塞同批次 decode 请求的 TBT——这正是 Ch08 Chunked Prefill 要解决的问题在 VLM 场景下被放大的体现。

---

## 3. KV Cache 显存压力

### 3.1 视觉 token 的 KV Cache 占用

每个视觉 token 在通过 LLM 时，会在**每一层**生成 K 和 V 向量，存入 KV Cache。视觉 token 的 KV Cache 与文本 token 的 KV Cache **完全相同**——LLM 无法区分它们。

回顾 Ch01 中的 KV Cache 公式：

$$\text{KV}_{per\_token} = 2 \times L \times n_{kv} \times d_h \times \text{dtype\_bytes}$$

**代入具体模型计算每图的视觉 token KV Cache：**

=== "LLaVA-1.5 (Vicuna-7B)"

    ```python
    L = 32          # 层数
    n_kv = 32       # KV heads (MHA)
    d_h = 128       # head dim
    dtype_bytes = 2  # BF16
    N_visual = 576   # 视觉 token 数

    kv_per_token = 2 * L * n_kv * d_h * dtype_bytes
    # = 2 × 32 × 32 × 128 × 2 = 524,288 bytes = 512 KB

    kv_per_image = kv_per_token * N_visual
    # = 512 KB × 576 = 288 MB
    ```

=== "Qwen-VL (Qwen-7B)"

    ```python
    L = 32
    n_kv = 32       # MHA
    d_h = 128
    dtype_bytes = 2
    N_visual = 1024

    kv_per_token = 2 * 32 * 32 * 128 * 2  # = 512 KB
    kv_per_image = 512 * 1024 * 1024  # 512 KB × 1024 = 512 MB
    ```

=== "InternVL-1.5 (InternLM2-20B, 4 tiles)"

    ```python
    L = 48
    n_kv = 8        # GQA-8
    d_h = 128
    dtype_bytes = 2
    N_visual = 1024 * 4  # 4 tiles

    kv_per_token = 2 * 48 * 8 * 128 * 2  # = 192 KB
    kv_per_image = 192 * 1024 * 4096  # 192 KB × 4096 = 768 MB
    ```

### 3.2 与纯文本请求的对比

为了直观理解视觉 token 对 KV Cache 的影响，我们对比两种请求：

```
场景 A (纯文本): system prompt (200 tokens) + user query (50 tokens)
场景 B (VLM):    system prompt (200 tokens) + image (576 visual tokens) + user query (50 tokens)

以 LLaVA-1.5 (Vicuna-7B) 为例，KV per token = 512 KB:

场景 A: 250 tokens × 512 KB = 125 MB KV Cache
场景 B: 826 tokens × 512 KB = 413 MB KV Cache

KV Cache 膨胀: 3.3x
```

对于使用 GQA 的现代模型（如 LLaMA-3-8B, $n_{kv} = 8$），单 token KV Cache 更小，但图像带来的**相对增量**依然显著：

```
LLaMA-3-8B (GQA-8): KV per token = 2 × 32 × 8 × 128 × 2 = 128 KB

场景 A: 250 × 128 KB = 31.25 MB
场景 B: 826 × 128 KB = 103.25 MB

KV Cache 膨胀: 3.3x（相同比例）
```

### 3.3 无法前缀缓存

在纯文本场景中，system prompt 的 KV Cache 可以通过 **Prefix Caching**（参考 Ch02）在多个请求间共享，大幅减少重复计算和显存占用。

但视觉 token 的 KV Cache **几乎无法共享**：

- **每张图像都是唯一的** — 不同用户上传不同图片，visual token 完全不同
- **图像 hash 不同** — 即使是相同图片，不同分辨率、裁剪方式也会导致不同的 visual token
- **token 序列位置不同** — 如果图像在 prompt 中的位置不同，KV Cache 也不同（因为 RoPE positional encoding）

!!! info "例外情况"
    如果系统中存在 **固定的参考图像**（如 few-shot 示例中的图片），其 ViT 输出可以缓存并复用。但这属于 ViT compute caching（见 13.3 节），而非 KV Cache 前缀共享。

### 3.4 多图输入的压力倍增

现代 VLM 支持多图输入，这使 KV Cache 压力呈线性增长：

```
模型: LLaVA-1.5 (Vicuna-7B), KV per token = 512 KB

单图请求:   576 visual tokens → 288 MB KV Cache
3 图请求:  1728 visual tokens → 864 MB KV Cache
5 图请求:  2880 visual tokens → 1,440 MB KV Cache ≈ 1.4 GB
10 图请求: 5760 visual tokens → 2,880 MB KV Cache ≈ 2.8 GB
```

在一张 80 GB 的 A100 上，部署 7B 模型后（权重 ~14 GB BF16），可用 KV Cache 空间约 50–55 GB。

```
最大并发纯文本请求 (250 tokens):  50 GB / 125 MB  ≈ 400 请求
最大并发单图 VLM 请求 (826 tokens): 50 GB / 413 MB ≈ 121 请求
最大并发 5 图 VLM 请求 (3130 tokens): 50 GB / 1565 MB ≈ 32 请求
```

!!! warning "容量规划警告"
    多图 VLM 场景下，单 GPU 的并发容量可以降低到纯文本场景的 **1/10 甚至更低**。容量规划必须考虑最坏情况下的图像数量。

---

## 4. 动态分辨率

### 4.1 从固定到动态

早期 VLM（如 LLaVA-1.5）使用固定分辨率：所有图像都被 resize 到 336×336，视觉 token 数量固定为 576。这使得 serving 系统的内存管理相对简单。

现代 VLM（Qwen-VL、InternVL、LLaVA-NeXT）引入了**动态分辨率**策略：根据输入图像的实际尺寸和宽高比，将图像切分为不同数量的 tile，每个 tile 独立经过 ViT 编码。

```
动态分辨率示意 (InternVL):

小图 (256×256):    → 1 tile  → 1024 visual tokens
中图 (512×512):    → 4 tiles → 4096 visual tokens
大图 (1024×768):   → 6 tiles → 6144 visual tokens
超大图 (2048×1024): → 12 tiles → 12288 visual tokens
```

### 4.2 对 Serving 的影响

动态分辨率给 serving 系统带来了一个根本性挑战：**同类请求的资源需求不再可预测**。

**内存预算不可预知：**

```
请求 A: 用户上传一张小缩略图  → 1 tile  → 1024 visual tokens → KV 需求小
请求 B: 用户上传一张高清照片  → 12 tiles → 12288 visual tokens → KV 需求 12x

在请求到达时，调度器必须先解析图像分辨率，才能估算内存需求。
```

**调度器需要图像感知：**

传统 LLM 调度器（参考 Ch08）在 admission 时只需知道 text token 数量。VLM 调度器则需要在 admission 时就获取以下信息：

1. 图像数量
2. 每张图像的分辨率 → tile 数量
3. 每个 tile 的视觉 token 数量
4. 总视觉 token 数 = $\sum_{i} \text{tiles}_i \times \text{tokens\_per\_tile}$

```python
# 伪代码：VLM 请求的 token 预算估算
def estimate_request_tokens(request):
    text_tokens = len(tokenizer.encode(request.text))
    visual_tokens = 0
    for image in request.images:
        num_tiles = compute_tile_count(image.width, image.height)
        visual_tokens += num_tiles * TOKENS_PER_TILE  # e.g., 1024
    return text_tokens + visual_tokens
```

### 4.3 PagedAttention 的适配

好消息是，PagedAttention（参考 Ch04）的分页机制天然适合动态分辨率场景：

- **按需分配 Block** — 视觉 token 产生多少 KV，就分配多少 Block，不需要预分配最大数量
- **无内部碎片** — 即使不同请求的视觉 token 数量差异很大，每个 Block 内的浪费仅限于最后一个 Block 的剩余 slot

但 PagedAttention 的调度器仍然需要知道**预期的总 token 数**，以判断是否接纳该请求。如果不知道一张图像会产生多少 visual token，可能导致：

- **过度接纳** — 接纳了太多高分辨率图像的请求，显存不足触发 preemption（参考 Ch04.3）
- **过度保守** — 按最大 tile 数预估，导致大量高分辨率请求被不必要地拒绝

!!! tip "最佳实践"
    在 API 层面限制最大图像分辨率和最大 tile 数，既能保证用户体验，又能简化调度器的内存预算管理。例如 Qwen-VL 默认最大 tile 数为 6，InternVL 默认最大为 12。

---

## 5. 小结：VLM Serving 的核心矛盾

VLM serving 的核心矛盾可以总结为：

```
                    ┌──────────────────────────────┐
                    │   图像是信息密集的输入       │
                    │   但 serving 系统按 token 计费 │
                    └──────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    Prefill 计算爆炸    KV Cache 膨胀     动态不可预测
    (10-100x 文本)    (3-12x 文本)     (tile 数不固定)
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                   需要专门的调度和优化策略
                      (见 13.2 和 13.3)
```

**关键数字速查表：**

| 指标 | 纯文本 (典型) | VLM 单图 | VLM 多图 (5 张) |
|------|-------------|---------|----------------|
| 输入 token 数 | 50–200 | 600–5000 | 3000–15000 |
| Prefill 时间 (8B 模型, A100) | 3–15 ms | 45–400 ms | 200–1200 ms |
| KV Cache / 请求 (GQA-8 8B) | 6–25 MB | 75–640 MB | 380–1920 MB |
| 可前缀缓存比例 | 高 (system prompt) | 低 (图像唯一) | 极低 |
| 最大并发 (50 GB KV 空间) | ~400 | ~70–120 | ~25–50 |

下一节将讨论如何通过调度策略缓解这些挑战。
