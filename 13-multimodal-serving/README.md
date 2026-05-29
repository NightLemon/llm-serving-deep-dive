# Ch13: 多模态推理

> 前置知识：Ch01 KV Cache 深度剖析、Ch04 PagedAttention、Ch05 Prefill-Decode 分离、Ch08 调度与批处理

## 🎯 学习目标

- 理解 Vision-Language Model (VLM) 的推理架构及其与纯文本 LLM 的本质差异
- 掌握视觉 token 对 prefill 计算量和 KV Cache 显存的爆炸性影响
- 理解混合模态批处理的调度挑战及解决方案
- 掌握视觉 token 压缩、ViT 缓存、异步流水线等关键优化技术
- 能够对 VLM serving 系统进行容量规划和性能调优

## 📑 内容大纲

| 节 | 文件 | 主题 |
|---|------|------|
| 13.1 | `01-vlm-serving-challenges.md` | VLM 推理挑战：prefill 爆炸、KV Cache 压力、动态分辨率 |
| 13.2 | `02-scheduling-and-batching.md` | 混合模态调度：分池策略、chunked prefill 适配、vLLM 多模态支持 |
| 13.3 | `03-optimization.md` | 优化技术：视觉 token 压缩、ViT 缓存、异步流水线、量化 |
| 13.4 | `04-capacity-planning.md` | 容量规划：admission control、SLO 分层、降级策略、监控指标 |

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | 2023 | Visual instruction tuning 范式 |
| [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966) | 2023 | 动态分辨率 VLM |
| [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238) | 2023 | 大规模 ViT + LLM |
| [FastV: An Image is Worth 1/2 Tokens After Layer 2](https://arxiv.org/abs/2403.06764) | 2024 | 视觉 token 剪枝 |
| [LLaVA-PruMerge](https://arxiv.org/abs/2403.15388) | 2024 | 视觉 token 合并 |

## 📁 文件清单

- [x] `01-vlm-serving-challenges.md` — VLM 推理挑战
- [x] `02-scheduling-and-batching.md` — 混合模态调度
- [x] `03-optimization.md` — VLM 优化技术
- [x] `04-capacity-planning.md` — VLM 容量规划与降级策略
- [x] `exercises.md` — 动手练习
