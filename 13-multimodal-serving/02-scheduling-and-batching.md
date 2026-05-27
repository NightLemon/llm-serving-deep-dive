# 混合模态调度与批处理

> 当文本请求和图像请求共享同一个 GPU，调度器必须学会"区别对待"

## 1. 混合模态批处理的挑战

### 1.1 问题：VLM Prefill 阻塞文本请求

在一个同时服务文本和图像请求的系统中，混合模态批处理面临的核心挑战是 **prefill 时间的巨大方差**。

回顾 Ch08 中 Continuous Batching 的基本机制：每个 iteration，调度器可以将新请求的 prefill 与现有请求的 decode 混合执行。但当 VLM 请求出现时：

```
纯文本环境:
  Iteration 1: [prefill 50 tok] [decode ×30]      ~6 ms
  Iteration 2:                  [decode ×31]       ~5 ms
  用户感知 TBT: ~5-6 ms ✓

混合模态环境:
  Iteration 1: [prefill 1054 tok(VLM)] [decode ×30]  ~90 ms
  Iteration 2:                         [decode ×31]   ~5 ms
  Iteration 1 的 decode 用户感知 TBT: ~90 ms ✗ (18x 正常值)
```

一个 VLM 请求的 prefill 可以轻松将整个 batch 的 iteration 时间拉高 **10–50 倍**，直接破坏所有 decode 请求的 TBT SLA。

### 1.2 量化影响

假设一个服务同时处理文本和 VLM 请求，VLM 请求占比 10%：

```
工作负载:
  - 90% 纯文本请求: 平均 100 tokens prefill
  - 10% VLM 请求: 平均 1000 tokens prefill (单图 + 文本)
  - 平均并发 decode 请求: 50

无差别调度 (FCFS):
  - 每 10 个新请求中有 1 个 VLM 请求
  - VLM 请求的 prefill iteration: ~85 ms
  - 其余 50 个 decode 请求在这个 iteration 被阻塞
  - TBT P99 ≈ 85 ms (由最慢的 prefill iteration 决定)

理想情况 (无 VLM 干扰):
  - 所有 prefill 都是 100 tokens: ~10 ms/iteration
  - TBT P99 ≈ 10-15 ms
```

**TBT P99 恶化 6–8 倍**，仅因 10% 的 VLM 请求。这就是 "延迟污染"（latency pollution）。

---

## 2. Chunked Prefill 在 VLM 中的适配

### 2.1 回顾 Chunked Prefill

Chunked Prefill（详见 Ch08.3）的核心思想是：将长 prompt 的 prefill 切分为固定大小的 chunk（如 512 tokens），每个 iteration 只处理一个 chunk，避免长 prefill 阻塞 decode 请求。

VLM 请求天然适合使用 chunked prefill 来缓解 prefill 阻塞问题。但视觉 token 的特殊性带来了额外的考量。

### 2.2 挑战：视觉 token 的切分

与文本 token 不同，视觉 token 有内在的空间结构——它们来自图像的不同 patch，且在动态分辨率方案中被组织为 tile。

**方案 1：按 tile 边界切分**

每个 tile 是一个独立的 ViT 编码单元（如 1024 tokens/tile）。可以将 tile 作为 chunk 的自然边界：

```
输入: [text_prefix (30 tok)] [tile_1 (1024 tok)] [tile_2 (1024 tok)] [text_suffix (20 tok)]
总计: 2098 tokens

Chunk size = 1024:
  Chunk 1: [text_prefix (30)] [tile_1 的前 994 tokens]    → 1024 tokens
  Chunk 2: [tile_1 的后 30 tokens] [tile_2 的前 994 tok]  → 1024 tokens
  Chunk 3: [tile_2 的后 30 tokens] [text_suffix (20)]     → 50 tokens
```

这种方案的优点是可以直接复用 Ch08 的 chunked prefill 机制，无需感知 tile 边界。缺点是视觉 token 可能被切到不同的 chunk 中。

**方案 2：ViT 与 LLM 分离处理**

将 ViT encoding 独立处理，生成所有视觉 token 的 embedding 后，再按标准 chunked prefill 流程处理 projected visual tokens：

```
Step 1 (独立): ViT encoding 所有图像 → visual embeddings
Step 2 (chunked): [visual_emb + text_tokens] 按标准 chunk size 切分并 prefill

优点: ViT encoding 只需做一次
缺点: 需要缓存 visual embeddings 直到所有 chunk 处理完毕
```

### 2.3 实践建议

```
推荐策略:
  1. 对于单 tile VLM 请求 (≤1024 visual tokens):
     → 直接使用标准 chunked prefill，chunk size = 512-1024
     → 视觉 token 和文本 token 一起切分，无需特殊处理

  2. 对于多 tile VLM 请求 (>2048 visual tokens):
     → 先做 ViT encoding (异步，见 13.3 节)
     → 将 projected embeddings 按标准 chunk size 切分
     → 每个 iteration 处理一个 chunk + decode batch
```

---

## 3. 分池策略

### 3.1 Prefill-Decode 分离 + 模态分离

Ch05 介绍的 Prefill-Decode 分离架构在 VLM 场景下**价值更大**。原因很简单：VLM prefill 与 text decode 的计算特征差异比纯文本场景更极端。

在纯文本场景下：

```
Prefill: compute-bound, 但时间尚可 (10-50 ms for 200-1000 tokens)
Decode:  memory-bound,  每步 ~5-15 ms
差异: 约 3-10x
```

在 VLM 场景下：

```
VLM Prefill: ViT encoding + 长序列 LLM prefill, 可达 100-500 ms
Text Decode: 依然 memory-bound, 每步 ~5-15 ms
差异: 约 20-100x
```

### 3.2 三池架构

针对 VLM serving 的最优分离架构是将 GPU 分为三个功能池：

```
                    ┌─────────────────┐
                    │   Load Balancer  │
                    │  (content-aware) │
                    └─────┬───────────┘
                          │
              ┌───────────┼──────────────┐
              ▼           ▼              ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ VLM Prefill  │ │ Text Prefill │ │ Decode Pool  │
    │    Pool      │ │    Pool      │ │              │
    │              │ │              │ │ (共享)        │
    │ ViT + LLM   │ │ LLM only     │ │ LLM only    │
    │ prefill      │ │ prefill      │ │ decode      │
    └──────────────┘ └──────────────┘ └──────────────┘
```

**路由规则：**

| 请求类型 | 路由目标 | 原因 |
|---------|---------|------|
| 文本请求 | Text Prefill Pool → Decode Pool | 避免被 VLM prefill 阻塞 |
| VLM 请求 | VLM Prefill Pool → Decode Pool | 包含 ViT encoding + 长 prefill |

**为什么三池优于两池？**

如果只做 prefill-decode 分离（两池），VLM prefill 和 text prefill 混在同一个 prefill pool 中。VLM prefill 耗时长，会占用 prefill pool 的 GPU 时间，导致 text prefill 排队：

```
两池方案中的 Prefill Pool:
  [VLM prefill 400ms] [text prefill 10ms] [VLM prefill 300ms] [text prefill 10ms]
                       ↑ 被 VLM 阻塞       ↑ 等待前面的 VLM

文本请求的 TTFT 被 VLM 请求间接拉高。
```

三池方案彻底隔离了两种 prefill 的资源竞争。

### 3.3 简化方案：两池 + 优先级

对于 VLM 请求占比较低的场景（<10%），三池架构可能过度设计。一个更实用的方案是：

```
两池 + 优先级:
  Prefill Pool: text prefill 高优先级, VLM prefill 低优先级
  Decode Pool:  统一处理

效果:
  - 当 Prefill Pool 空闲时，VLM 和 text prefill 都能及时处理
  - 当 Prefill Pool 繁忙时，text prefill 优先，VLM prefill 排队
  - VLM 请求的 TTFT 可能稍高，但 text 请求的 TTFT 得到保障
```

---

## 4. 调度策略

### 4.1 预算感知接纳控制

传统 LLM 调度器的 admission 预算基于两个维度（参考 Ch08.2）：

- `max_num_seqs` — 最大并发请求数
- `max_num_batched_tokens` — 单 iteration 最大 token 数

VLM 调度器需要将**视觉 token 计入预算**：

```python
# 伪代码：VLM-aware admission control
def can_admit(request, current_budget):
    text_tokens = count_text_tokens(request)
    visual_tokens = count_visual_tokens(request)  # 需要解析图像
    total_tokens = text_tokens + visual_tokens

    # KV Cache 预算检查
    kv_blocks_needed = ceil(total_tokens / block_size)
    if free_blocks < kv_blocks_needed:
        return False  # KV Cache 不足

    # Prefill token 预算检查
    if current_budget.batched_tokens + total_tokens > max_num_batched_tokens:
        return False  # 超过 iteration token 预算

    # 并发数检查
    if current_budget.num_seqs + 1 > max_num_seqs:
        return False

    return True
```

!!! warning "关键点"
    `count_visual_tokens()` 必须在 admission 时就执行，这意味着需要在调度器中**提前解析图像元数据**（分辨率 → tile 数 → token 数），而不是等到 ViT encoding 时才知道。

### 4.2 优先级策略

在混合模态工作负载下，合理的优先级策略可以有效防止延迟污染：

**策略 1：文本请求优先**

```
优先级排序: text decode > text prefill > VLM prefill

效果:
  - text 请求的 TBT 和 TTFT 不受 VLM 影响
  - VLM 请求的 TTFT 可能被推迟
  - 适合: 文本交互为主，图像理解为辅的场景
```

**策略 2：按 token 成本加权**

```
请求优先级 = base_priority × (1 / total_tokens)

效果:
  - token 数少的请求优先处理 (SJF 思想)
  - VLM 请求因 token 多而自然被降低优先级
  - 更公平但实现稍复杂
```

**策略 3：deadline-aware**

```
每个请求有 TTFT deadline:
  text 请求: TTFT deadline = 200ms
  VLM 请求:  TTFT deadline = 2000ms (用户预期更长的等待)

调度器优先处理 deadline 最近的请求。
```

---

## 5. vLLM 多模态支持

### 5.1 整体架构

vLLM 从 v0.4 开始支持多模态输入。其实现的核心思路是：**将多模态输入标准化为 token 序列**，尽可能复用已有的 text-only 推理路径。

```
vLLM 多模态处理流程:

User Request (含图像)
  → API Server: 解析图像，构造 MultiModalInputs
  → Scheduler: 将 visual tokens 计入 token budget
  → Model Runner: ViT encoding + LLM forward
  → Output: 生成文本
```

### 5.2 MultiModalInputs

vLLM 使用 `MultiModalInputs` 数据结构封装多模态数据：

```python
# vLLM 中多模态输入的简化表示
class MultiModalInputs:
    """封装多模态数据，传递给 model runner。"""
    type: str           # "image", "video", "audio"
    data: dict          # 原始数据 (pixel_values, etc.)
    placeholder_range: PlaceholderRange  # token 序列中的占位范围
```

在 token 序列中，视觉 token 由**占位符 token** 表示。当 model runner 执行 forward 时，这些占位符会被 ViT 编码后的真实 visual embedding 替换：

```
Token sequence (逻辑视图):
  [BOS] [text_1] [text_2] [IMG] [IMG] [IMG] ... [IMG] [text_3] [text_4]
                           └──── 576 个占位符 ────┘

Model runner forward:
  1. ViT(image) → visual_embeddings [576, hidden_dim]
  2. 用 visual_embeddings 替换 [IMG] 占位符
  3. LLM forward 正常执行
```

### 5.3 图像预处理流程

vLLM 为每个 VLM 模型注册了专用的 image processor：

```python
# 简化的图像预处理流程
def process_images(images, model_config):
    """将原始图像转换为模型输入。"""
    processor = get_image_processor(model_config)

    # 1. Resize / normalize / tile
    pixel_values = processor.preprocess(images)

    # 2. 计算 visual token 数量
    num_visual_tokens = compute_num_tokens(
        image_size=images[0].size,
        patch_size=model_config.patch_size,
        max_tiles=model_config.max_tiles,
    )

    # 3. 在 token 序列中插入占位符
    placeholder_ids = [IMG_TOKEN_ID] * num_visual_tokens

    return pixel_values, placeholder_ids, num_visual_tokens
```

### 5.4 调度器适配

vLLM 调度器在处理 VLM 请求时，将 visual token 的占位符计入 token budget：

```
调度器视角:
  Request A (text-only):  prompt_tokens = 50
  Request B (VLM):        prompt_tokens = 50 (text) + 576 (visual placeholders) = 626

  max_num_batched_tokens = 2048 时:
    一个 iteration 可以容纳 ~3 个 VLM 请求的 prefill
    或 ~40 个纯文本请求的 prefill
    或混合组合
```

!!! tip "实践指南"
    在 vLLM 中使用 VLM 时，`max_num_batched_tokens` 需要设置得更大（建议 4096–8192），以避免 VLM 请求因 token 预算不足而频繁排队。同时应启用 `--enable-chunked-prefill` 来缓解 VLM prefill 对 decode 的阻塞。

---

## 6. 小结

| 策略 | 适用场景 | 复杂度 | 效果 |
|------|---------|-------|------|
| Chunked Prefill | 所有 VLM serving | 低 | 缓解 prefill 阻塞，但不能完全隔离 |
| 三池分离 (VLM prefill / text prefill / decode) | 大规模混合部署 | 高 | 完全隔离模态间干扰 |
| 两池 + 优先级 | 中等规模，VLM 占比 <10% | 中 | 性价比高的折中方案 |
| 预算感知 admission | 所有 VLM serving | 低 | 防止显存 OOM |
| 文本优先调度 | 文本 SLA 严格的场景 | 低 | 保护文本请求的延迟 |

下一节将讨论在模型和系统层面优化 VLM serving 性能的具体技术。
