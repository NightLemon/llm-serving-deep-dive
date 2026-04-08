# Ch01: KV Cache 深度剖析

> 前置知识：[gpu-ai-systems-learning/07/01-kv-cache.md](../../gpu-ai-systems-learning/07-inference-optimization/01-kv-cache.md)

## 🎯 学习目标

- 理解 KV Cache 在 GPU 显存中的**物理布局**，而不仅仅是逻辑概念
- 掌握不同 Attention 变体（MHA / MQA / GQA / MLA）对 KV Cache 大小和布局的影响
- 理解 Block Table 数据结构及其在 vLLM 中的实现
- 能够精确计算任意模型配置下的 KV Cache 显存占用
- 理解 Prefill 和 Decode 阶段 KV Cache 行为的本质差异

## 📑 内容大纲

### 1. KV Cache 内存布局（01-memory-layout.md）

**论文/原理：**
- Attention 计算中 K、V 张量的形状与存储
- 连续内存 vs 分页内存：为什么传统实现会浪费显存？
- MHA 布局：`[num_layers, 2, batch, num_heads, seq_len, head_dim]`
- GQA 布局：KV head 数量 < Q head 数量时的存储优化
- MLA 布局（DeepSeek-V2）：将 KV 压缩到低维 latent，布局完全不同

**源码走读：**
- vLLM `KVCache` 张量分配（`vllm/v1/worker/gpu_model_runner.py`）
- HuggingFace Transformers 中的 `DynamicCache` vs `StaticCache`
- Block Table 的定义与索引方式

**工程关注点：**
- 不同精度（FP16 / BF16 / FP8）下的显存占用对比
- KV Cache 占模型总显存的比例（典型场景分析）

### 2. Prefill vs Decode 的本质差异（02-prefill-decode.md）

**论文/原理：**
- Prefill：计算密集（compute-bound），一次性生成整个 prompt 的 KV
- Decode：访存密集（memory-bound），每步只生成 1 个 token 的 KV
- 为什么 Prefill 吞吐量远高于 Decode？（roofline model 分析）
- Batch 中混合 prefill 和 decode 请求的挑战

**源码走读：**
- vLLM 中 prefill 和 decode 的代码路径差异
- `is_prompt` 标志在调度器和 model runner 中的传播

**工程关注点：**
- 如何用 profiler 区分 prefill 和 decode 的 GPU 利用率
- TTFT (Time To First Token) vs TBT (Time Between Tokens) 的权衡

### 3. 显存占用精确计算（03-memory-calculation.md）

**核心公式推导：**
- 基础公式：$\text{KV\_size} = 2 \times L \times n_{kv} \times d_h \times s \times b \times \text{dtype\_bytes}$
- 常见模型实例计算（LLaMA-3-70B, DeepSeek-V3, Qwen-2.5-72B）
- KV Cache vs 模型权重的显存占比分析
- `gpu_memory_utilization` 参数对可用 KV Cache 空间的影响

**动手练习：**
- 给定模型参数，计算最大可服务并发数
- 调整 `max_model_len` 对吞吐量的影响

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Self-Attention 原始定义 |
| [GQA: Training Generalized Multi-Query Attention](https://arxiv.org/abs/2305.13245) | 2023 | GQA 架构 |
| [DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM](https://arxiv.org/abs/2405.04434) | 2024 | MLA 架构 |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 2023 | Block-based KV Cache 管理 |

## 📁 文件清单

- [x] `01-memory-layout.md` — KV Cache 内存布局
- [x] `02-prefill-decode.md` — Prefill vs Decode
- [x] `03-memory-calculation.md` — 显存占用精确计算
- [x] `exercises.md` — 动手练习
