# 动手练习：VLM Serving 分析与实验

> 通过计算和分析，深入理解多模态推理的资源开销和优化空间

---

## 练习 1：VLM KV Cache 预算计算

### 题目

你的团队计划部署 **Qwen2-VL-7B** 用于图像问答服务。硬件为 **1 张 NVIDIA A100-80GB**。

已知模型参数：
- LLM 部分：层数 $L = 28$，KV heads = 4 (GQA)，head dim = 128，BF16
- ViT 部分：patch size = 14，支持动态分辨率，最大 6 tiles，每 tile 1024 visual tokens
- 模型权重（BF16）约 14 GB

业务需求：
- 每个请求包含 1 张图像 + 50 tokens 文本
- 图像分辨率分布：40% 低分辨率（1 tile），40% 中分辨率（4 tiles），20% 高分辨率（6 tiles）
- 平均生成长度 200 tokens
- `gpu_memory_utilization` = 0.9

**问题**：

**(a)** 计算每个 token 的 KV Cache 大小。

**(b)** 分别计算低/中/高分辨率图像请求在 prefill 完成后的 KV Cache 占用（含 text tokens + visual tokens + 预期生成 tokens）。

**(c)** 假设请求按上述分布到达，计算加权平均单请求 KV Cache 大小。

**(d)** 估算最大并发请求数。对比纯文本场景（仅 50 tokens prompt + 200 tokens 生成），VLM 场景的并发容量下降了多少？

---

## 练习 2：Prefill 延迟影响分析

### 题目

在一个使用 Continuous Batching 的系统中（参考 Ch08），同时有纯文本和 VLM 请求。

**场景设定**：
- 模型：LLaMA-3-8B (GQA-8) + ViT-L/14
- GPU：A100-80GB
- 当前有 40 个请求在 decode（平均序列长度 500 tokens）
- Decode iteration 正常耗时 ~8 ms
- 新到达 1 个 VLM 请求：1 张 448×448 图片（1024 visual tokens）+ 30 text tokens

**问题**：

**(a)** 不使用 chunked prefill 时，该 VLM 请求的 prefill 在下一个 iteration 会产生多少 tokens 的计算量？估算该 iteration 的耗时（假设 A100 上 LLaMA-3-8B 的 prefill 吞吐量约 20,000 tokens/s）。

**(b)** 这个 iteration 中，40 个 decode 请求的 TBT 会变成多少？相比正常值恶化了多少倍？

**(c)** 如果启用 chunked prefill（`max_num_batched_tokens = 512`），这个 VLM 请求需要多少个 iteration 才能完成 prefill？在此期间 decode 请求的 TBT 大约是多少？

**(d)** 如果同时到达 3 个 VLM 请求（每个 1024 visual tokens），不使用 chunked prefill 时情况会怎样？

---

## 练习 3：优化收益估算

### 题目

给定以下 VLM serving 场景，估算各种优化技术的收益：

**基线场景**：
- 模型：LLaVA-1.5-7B（ViT-L/14 + Vicuna-7B MHA）
- 每请求：1 张图 (336×336, 576 visual tokens) + 50 text tokens + 200 生成 tokens
- 单图 ViT encoding: 5 ms
- KV per token: 512 KB (MHA, BF16)
- 并发 decode 请求：30

**计算以下优化的预期收益**：

**(a)** FastV 50% 剪枝（第 2 层后剪掉 288 个视觉 token）：
  - Prefill 计算量减少多少？（注意前 2 层仍处理全部 token）
  - 每请求 KV Cache 节省多少 MB？
  - 并发容量提升多少？

**(b)** ViT Compute Cache（假设同图多问场景，命中率 80%）：
  - ViT 计算时间平均节省多少？
  - 对端到端 TTFT 的影响？

**(c)** FP8 KV Cache（视觉 token 和文本 token 的 KV 都从 BF16 → FP8）：
  - 每请求 KV Cache 从多少 MB 降到多少 MB？
  - 并发容量提升多少？

**(d)** 组合优化（FastV 50% + FP8 KV Cache）：
  - 综合效果如何？并发容量对比基线提升多少？

---

## 通用提示

- **计算验证**：建议用 Python 代码验证所有手动计算
- **单位注意**：1 GB = $1024^3$ bytes
- **参考章节**：KV Cache 计算公式参考 Ch01.3，Chunked Prefill 原理参考 Ch08.3

---

## 练习 4：Vision API 计费规律观察（🟢 纯 API）

**目标**：通过观察大厂 API 计费，反推出图像 token 化的工程取舍。

### 准备

任选 Anthropic Claude 或 OpenAI GPT-4o（都支持 vision）。准备 4 张测试图：

- A: 256×256 简单图（如 logo）
- B: 1024×1024 中等图（如风景照）
- C: 2048×2048 高分辨率图（如截图）
- D: 4096×4096 超大图（如扫描文档）

### 实验

对每张图发同一个简短问题（如 "describe this in one sentence"），记录响应里的 `usage` 字段。

```python
# Anthropic 示例
import anthropic, base64

client = anthropic.Anthropic()
for path in ["a.png", "b.png", "c.png", "d.png"]:
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=64,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": "describe this in one sentence"}
            ]
        }]
    )
    print(path, resp.usage)
```

### 分析任务

1. 列表对比四张图的 `input_tokens`，画出"分辨率 vs token 数"的散点图
2. 单图 token 数有没有上限？两家 API 的策略一样吗？（去看官方文档验证）
3. 如果你的业务大量传 4K 截图，按这套计费 1M 请求要多少美元？相比传文本（假设等价信息量约 500 tokens）贵多少倍？
4. **反推**：这套计费规则下，VLM 服务商在内部一定做了什么优化才能保住毛利？（提示：think "动态分辨率 / token 剪枝 / cache"）

### 验收

能用一段话向产品经理解释："为什么 vision API 那么贵"，并能给出至少 2 个降本建议（业务侧的、不需要换模型的）。

---

## 练习 5：3070 上本地跑 VLM 可行性测试（🟢 本地 · 选做）

**目标**：把"3070 跑得动什么 VLM"摸清边界，作为后续选型依据。

### 候选模型（按显存预算从小到大）

| 模型 | 量化 | 估算显存 | 是否可行 |
|------|------|---------|---------|
| Moondream-2B | FP16 | ~4 GB | ✅ |
| Qwen2-VL-2B-Instruct | AWQ INT4 | ~3 GB | ✅ |
| LLaVA-1.5-7B | AWQ INT4 | ~5-6 GB | ⚠️ 紧 |
| Qwen2-VL-7B-Instruct | AWQ INT4 | ~6 GB | ⚠️ 紧 |
| InternVL2-8B | AWQ INT4 | ~7 GB | ❌ 大概率 OOM |

### 任务

1. 用 `transformers` 或 `vllm`（注意 vLLM 对 multimodal 的支持版本要求）跑 Moondream-2B + Qwen2-VL-2B-AWQ
2. 测同一张图 + 同一段问题在两个模型上的：
   - 输出质量（主观打分 1-5）
   - First token 延迟
   - 总延迟
   - 峰值显存（用 `nvidia-smi` 或 `torch.cuda.max_memory_allocated()`）
3. 尝试加载 LLaVA-1.5-7B-AWQ：能否塞进 8GB？如果 OOM 在哪一步？
4. 写一份"3070 VLM 部署可行性"短报告（≤ 1 页），结论 + 数据 + 推荐模型

### 验收

能回答："如果我要在 3070 上部署一个 VLM 做内部 demo，选哪个模型、为什么、会有什么质量妥协。"

---

## 练习 6：VLM Admission Control 设计（📖 设计题）

**目标**：把 [04-capacity-planning.md](04-capacity-planning.md) 中的容量规划方法应用到真实策略设计。

### 场景

你要上线一个图片问答服务：

- 模型：Qwen2-VL-7B，A100-80GB × 2
- 业务流量：80% 单图问答，15% 多图对比，5% 长截图/文档
- SLA：单图 TTFT P95 < 2s，多图 TTFT P95 < 6s
- 用户上传图片分辨率不可控，最大可能到 4096×4096
- 希望高峰时优先保护单图问答体验

### 任务

1. 设计 online policy：`max_images`、`max_edge`、`max_tiles_per_image`、`max_new_tokens`。
2. 写出 admission control 伪代码：什么情况下接纳、降级、转 async queue、拒绝？
3. 设计过载降级顺序：先降 tiles、还是先限多图、还是先切小模型？说明理由。
4. 列出 dashboard 指标：至少包含 visual tokens、tile 数、rejection reason、ViT latency、LLM prefill latency。

### 验收

能交付一份上线策略表，并回答："为什么 4096×4096 图不应该默认按最高分辨率进入在线池？"
