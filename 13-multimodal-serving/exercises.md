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
