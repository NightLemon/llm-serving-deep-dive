# Ch03: KV Cache 压缩

> 前置知识：Ch01 KV Cache 深度剖析

## 🎯 学习目标

- 理解 KV Cache 量化（FP8 / INT8 / INT4）的原理与精度影响
- 深入理解 MLA（Multi-head Latent Attention）架构如何从根本上压缩 KV Cache
- 了解选择性缓存策略（H2O、StreamingLLM、Attention Sink）
- 能够根据场景选择合适的 KV Cache 压缩方案

## 📑 内容大纲

### 1. KV Cache 量化（01-quantization.md）

**原理：**
- 为什么 KV Cache 可以量化？—— K/V 向量的数值分布特征
- Per-tensor vs Per-token vs Per-channel 量化粒度
- FP8 (E4M3 / E5M2) 量化：精度损失最小的方案
- INT8 / INT4 量化：更激进的压缩
- 量化引入的误差如何在多层注意力中累积？

**源码走读：**
- vLLM Quantized KV Cache 配置：`--kv-cache-dtype fp8` / `fp8_e4m3`
- 量化感知的 attention kernel（FlashAttention 对 FP8 KV 的支持）

**Benchmark：**
- 不同量化精度下的生成质量对比（perplexity、下游任务指标）
- 显存节省 vs 精度损失的 Pareto 分析

### 2. MLA: Multi-head Latent Attention（02-mla.md）

**论文解读：**
- DeepSeek-V2 中 MLA 的设计动机
- 传统 MHA：KV Cache = $2 \times n_h \times d_h$ per token per layer
- MLA：将 K、V 投影到低维 latent space $c_t^{KV}$，KV Cache = $d_c$（远小于 $2 \times n_h \times d_h$）
- 推理时的 KV Cache 大小对比：
  - MHA (LLaMA-65B)：约 40 bytes/token/layer
  - MLA (DeepSeek-V2-236B)：约 4.5 bytes/token/layer（~9x 压缩）
- Decoupled RoPE 的必要性：位置编码不能被压缩到 latent space

**源码走读：**
- vLLM 中 MLA 的实现路径
- `vllm/model_executor/layers/mla.py` — 压缩与解压逻辑
- MLA 对 attention backend 的要求（FlashMLA）

**关键讨论：**
- MLA vs GQA：二者的设计哲学差异
- MLA 对 prefix caching 的影响（latent space 的 hash 是否兼容？）

### 3. 选择性缓存与上下文压缩（03-selective-caching.md）

**核心思路：不是所有 token 的 KV 都同等重要。**

**H2O (Heavy-Hitter Oracle)：**
- 论文：[H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs](https://arxiv.org/abs/2306.14048)
- 基于注意力分数识别 "heavy-hitter" token
- 保留重要 token 的 KV，丢弃不重要的
- 固定大小 KV Cache 下的近似精度

**StreamingLLM / Attention Sink：**
- 论文：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- 发现：初始 token 的 KV 具有 "attention sink" 效应
- 保留 sink token + 最近的 sliding window → 无限长度推理
- 局限性：中间上下文的信息会丢失

**工程考量：**
- 选择性缓存 vs 全量缓存的适用场景
- 与 prefix caching 的兼容性问题

### 4. GQA/MQA 深度分析（04-gqa-mqa.md）

**对比分析：**
- MHA → MQA → GQA 的演化路径
- 各方案的 KV Cache 大小计算与对比
- GQA 分组数对推理性能的影响（不只是显存，还有 bandwidth）
- 主流模型的 Attention 方案选择（LLaMA-3: GQA-8, Qwen-2.5: GQA, DeepSeek-V3: MLA）

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | MLA 架构 |
| [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) | 2023 | 选择性 KV 驱逐 |
| [Efficient Streaming LMs with Attention Sinks](https://arxiv.org/abs/2309.17453) | 2023 | StreamingLLM |
| [GQA: Training Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) | 2023 | GQA |
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | 2024 | 2-bit KV 量化 |
| [KVQuant: Towards 10 Million Context Length LLM Inference](https://arxiv.org/abs/2401.18079) | 2024 | 超长上下文 KV 量化 |

## 📁 文件清单

- [x] `01-quantization.md` — KV Cache 量化
- [x] `02-mla.md` — MLA 深度解读
- [x] `03-selective-caching.md` — 选择性缓存
- [x] `04-gqa-mqa.md` — GQA/MQA 深度分析
- [x] `exercises.md` — 动手练习
