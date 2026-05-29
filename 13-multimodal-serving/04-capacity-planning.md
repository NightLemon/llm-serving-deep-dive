# VLM Serving 容量规划与降级策略

> 多模态服务的容量问题不是“能跑多大的模型”，而是“在图像数量、分辨率和输出长度都不可控时，如何保证不把 GPU 填爆”。本节给出一套可落地的预算、接纳控制和降级方案。

## 1. 请求成本模型

对一个 VLM 请求，最重要的三个成本是：

```text
T_total = T_image_preprocess + T_vit + T_llm_prefill + T_decode
M_kv    = (N_text + N_visual + N_output) × KV_per_token
Budget  = max(N_text + N_visual, chunk_size) + active_decode_tokens
```

其中：

- `N_visual` 由图像数量、分辨率、tile 策略、patch size 决定。
- `T_vit` 只在图像进入服务时发生，可以被缓存或异步隐藏。
- `T_llm_prefill` 和 `M_kv` 会随视觉 token 线性增长，在长上下文和多图场景下成为主要瓶颈。

## 2. Admission Control

### 2.1 必要输入

接纳一个 VLM 请求前，调度器至少要知道：

| 字段 | 用途 |
|------|------|
| image_count | 限制多图爆炸 |
| width / height | 估算 tile 数 |
| max_tiles_per_image | 控制动态分辨率上限 |
| text_prompt_tokens | 估算 prefill 和 KV |
| max_new_tokens | 估算最坏 KV |
| request_class | 区分 interactive / batch / best-effort |

不能等 ViT encoding 完成后才知道 token 数。那时请求已经占用了队列位置，过载时会把后续请求一起拖慢。

### 2.2 接纳逻辑

```python
def estimate_visual_tokens(images, policy):
    total = 0
    for image in images:
        tiles = compute_tiles(
            width=image.width,
            height=image.height,
            max_tiles=policy.max_tiles_per_image,
        )
        total += tiles * policy.tokens_per_tile
    return total


def can_admit_vlm(req, state, policy):
    text_tokens = tokenizer_count(req.text)
    visual_tokens = estimate_visual_tokens(req.images, policy)
    expected_total = text_tokens + visual_tokens + req.max_new_tokens

    kv_blocks = ceil(expected_total / state.block_size)
    if kv_blocks > state.free_blocks * policy.reserve_ratio:
        return False, "kv_budget_exceeded"

    prefill_tokens = text_tokens + visual_tokens
    if prefill_tokens > policy.max_prefill_tokens_per_request:
        return False, "prefill_too_large"

    if len(req.images) > policy.max_images:
        return False, "too_many_images"

    return True, "ok"
```

`reserve_ratio` 通常小于 1，例如 0.8。剩余 20% 留给 decode 增长、调度误差和短时流量尖峰。

## 3. SLO 分层

VLM 请求的延迟天然高于文本请求，因此不要把所有请求放进同一套 SLO。

| 类别 | 示例 | TTFT 目标 | 策略 |
|------|------|----------|------|
| interactive text | 聊天、搜索改写 | 200-500 ms | 最高优先级 |
| interactive vision | 单图问答 | 1-3 s | chunked prefill + 限 tile |
| document vision | OCR/截图理解 | 3-10 s | batch-friendly，允许排队 |
| video / multi-image | 视频摘要、多页文档 | 10s+ | 异步 job，不进在线池 |

关键原则：**视频和多页文档不要和在线聊天共享同一个 decode SLA**。它们应该进入异步队列或单独的 batch pool。

## 4. 降级策略

当系统接近过载时，VLM 有比文本更多的“优雅降级”空间。

### 4.1 图像侧降级

| 降级动作 | 影响 | 适用场景 |
|----------|------|----------|
| 降低 max tiles | 直接减少 visual tokens | 通用 |
| 限制最大边长 | 降低 ViT 和 LLM prefill | 截图、照片 |
| 单图优先，拒绝多图 | 防止请求级爆炸 | 在线交互 |
| 视频降低采样帧数 | 10-100x 降本 | 视频摘要 |
| OCR 先行抽文本 | 将图像问题转文本问题 | 文档/截图 |

### 4.2 模型侧降级

| 降级动作 | 影响 | 风险 |
|----------|------|------|
| 切到小 VLM | 显著降延迟 | 质量下降 |
| 启用 visual token pruning | 降 KV 和 prefill | 细节问题变差 |
| FP8 KV cache | KV 减半 | 极少数任务质量轻微波动 |
| 限制 max_new_tokens | 控制 decode 阶段增长 | 回答可能截断 |

### 4.3 路由侧降级

```text
正常:      high-res VLM -> online VLM pool
轻度过载:  high-res VLM -> lower tile policy
中度过载:  multi-image -> async batch pool
重度过载:  vision request -> 429 / retry-after / smaller image hint
```

降级策略要在 API 层显式暴露，例如返回：`image_downsampled: true`、`tiles_used: 4`、`degraded_reason: "load_shedding"`。否则用户会把质量变化误判为模型随机性。

## 5. Capacity Planning 示例

场景：Qwen2-VL-7B，A100-80GB，BF16，GQA-4，$L=28$，$d_h=128$。

```
KV_per_token = 2 × 28 × 4 × 128 × 2 bytes = 57,344 bytes ≈ 56 KB
可用 KV pool 假设 = 50 GB
```

请求类型：

| 类型 | visual tokens | text + output | total tokens | KV / req | 50GB 可容纳 |
|------|---------------|---------------|--------------|----------|------------|
| 纯文本 | 0 | 250 | 250 | 14 MB | ~3600 |
| 单图低清 | 1024 | 250 | 1274 | 71 MB | ~720 |
| 单图高清 | 4096 | 250 | 4346 | 243 MB | ~210 |
| 6 tile | 6144 | 250 | 6394 | 358 MB | ~143 |
| 4 图高清 | 16384 | 250 | 16634 | 931 MB | ~55 |

这张表不是最终并发上限，因为实际还受 scheduler token budget、ViT throughput、decode batch 大小约束。但它能快速告诉你：多图高清请求会把可服务并发压低两个数量级。

## 6. 监控指标

### 6.1 请求结构指标

- `vlm_request_images_count`
- `vlm_request_visual_tokens`
- `vlm_request_tiles_count`
- `vlm_request_downsampled_total`
- `vlm_request_rejected_total{reason=...}`

### 6.2 性能指标

- `vit_encode_seconds`
- `vlm_prefill_seconds`
- `vlm_decode_tpot_seconds`
- `vlm_chunked_prefill_chunks`
- `vlm_kv_blocks_reserved`
- `vlm_kv_blocks_released`

### 6.3 质量/降级指标

- `vlm_degraded_total{policy=...}`
- `vlm_ocr_fallback_total`
- `vlm_token_pruning_ratio`
- `vlm_user_retry_rate`

如果只监控总体 TTFT，你会看不到问题来源。VLM 必须把 `ViT encoding`、`LLM prefill`、`decode` 拆开看。

## 7. 上线 Checklist

- [ ] API 层限制最大图像数、最大边长、最大 tile 数。
- [ ] admission control 在入队前估算 visual tokens。
- [ ] online pool 与 batch/video pool 隔离。
- [ ] chunked prefill 默认开启，并验证 text TBT 不被 VLM prefill 污染。
- [ ] 监控按 text-only / single-image / multi-image / video 分组。
- [ ] 降级策略有用户可见标记。
- [ ] 压测覆盖 P50 小图、P95 大图、P99 多图。
- [ ] 成本模型按 visual tokens 而不是请求数估算。

## 8. 一个实用默认配置

```yaml
interactive_vlm_policy:
  max_images: 1
  max_edge: 1536
  max_tiles_per_image: 4
  max_new_tokens: 512
  reserve_ratio: 0.8
  enable_chunked_prefill: true
  max_num_batched_tokens: 8192
  overload_degradation:
    - max_tiles_per_image: 2
    - max_edge: 1024
    - reject_multi_image: true

batch_vlm_policy:
  max_images: 16
  max_tiles_per_image: 12
  max_new_tokens: 2048
  queue: async_batch
```

默认策略应该保护在线交互体验，而不是追求“任何图片都能最高分辨率处理”。真正需要高分辨率、多图、视频的请求，应进入显式的 batch 路径。
