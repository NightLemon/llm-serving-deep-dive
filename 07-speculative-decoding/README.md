# Ch07: 投机解码进阶

> 前置知识：Ch01 KV Cache 深度剖析、[gpu-ai-systems-learning/07/04-speculative-decoding.md](../../gpu-ai-systems-learning/07-inference-optimization/04-speculative-decoding.md)

## 🎯 学习目标

- 深入理解 Speculative Decoding 的数学保证（无损性证明）
- 掌握主流投机解码方案（EAGLE、Medusa、MTP）的架构差异
- 理解 Tree Attention 在投机解码中的作用
- 走读 vLLM 中多种投机解码策略的实现
- 能够根据场景选择最优的投机解码方案并调参

## 📑 内容大纲

### 1. 投机解码数学基础（01-math-foundation.md）

**核心定理：Rejection Sampling 保证无损性**
- Draft model 生成 $\gamma$ 个 candidate tokens
- Target model 并行验证所有 candidate
- 基于 acceptance-rejection sampling 决定接受哪些 token
- 关键证明：接受的 token 来自 target model 的分布（无偏）
- 期望加速比：$E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$
  - $\alpha$：draft model 与 target model 的分布匹配程度

**直觉理解：**
- Draft model "赌" 未来几个 token
- Target model 一次性验证（verification 的成本 ≈ 单个 token 的生成成本）
- 赌对了 → 一次生成多个 token
- 赌错了 → 至少还能生成 1 个 token（从修正分布采样）

### 2. EAGLE 系列（02-eagle.md）

**EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)：**
- 论文：[EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- 核心思想：用 target model 的 hidden states 作为 draft model 的输入
- 不需要独立的 draft model，用轻量级 head 直接预测
- 架构：LM head 之前的特征 + 可训练的 EAGLE head

**EAGLE-2：**
- 动态调整 draft tree 的形状（token tree，而非固定长度链）
- 基于 confidence 决定是否继续扩展

**EAGLE-3：**
- vLLM 对 EAGLE 的进一步优化
- 更好的 tree attention 集成

**源码走读：**
- `vllm/v1/spec_decode/eagle/` — EAGLE 推理引擎
- `vllm/v1/worker/gpu/spec_decode/eagle/` — GPU 端实现

### 3. Medusa（03-medusa.md）

**核心思想：**
- 论文：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- 在 target model 上附加多个平行 head，每个 head 预测不同位置的 token
- 不需要额外的 draft model，只需要几个小的 MLP head
- 使用 tree attention 验证多条候选路径

**与 EAGLE 的对比：**
| 特性 | EAGLE | Medusa |
|------|-------|--------|
| 额外参数 | EAGLE head | 多个 Medusa head |
| 输入 | hidden states | hidden states |
| draft 策略 | 自回归 multi-step | 并行 multi-head |
| 树结构 | 动态调整 | 拓扑结构固定 |
| 训练成本 | 中 | 低 |

**源码走读：**
- `vllm/v1/spec_decode/medusa.py`

### 4. MTP: Multi-Token Prediction（04-mtp.md）

**核心思想：**
- 论文：[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- 与 Medusa 类似但在**训练阶段**就优化多 token 预测能力
- DeepSeek-V3 / Qwen3 等模型原生支持 MTP
- 推理时直接用 MTP head 做投机解码

**源码走读：**
- `vllm/v1/spec_decode/` 中的 MTP 相关实现
- 模型层面：`deepseek_mtp.py`、`qwen3_5_mtp.py` 等

### 5. Draft Model 选择与 N-gram 方案（05-draft-selection.md）

**Draft Model 选择策略：**
- 同系列小模型（如 LLaMA-3-8B → LLaMA-3-70B）
- 量化后的同模型（FP8/INT4 版本作为 draft）
- MLPSpeculator：轻量级 MLP 作为 draft

**N-gram Speculation：**
- 无需任何额外模型
- 从 prompt 或已生成的文本中匹配 n-gram pattern
- 适合代码补全等重复性高的场景
- vLLM 支持：`--speculative-model [ngram]`

**Suffix Decoding：**
- 基于 suffix tree 的匹配
- 利用上下文中的重复模式

### 6. Tree Attention（06-tree-attention.md）

**为什么需要 Tree Attention？**
- 投机解码生成的是一棵 token tree，不是线性序列
- 验证阶段需要同时评估多条路径
- Tree attention mask 确保每个 token 只 attend 到其祖先节点

**实现细节：**
- Tree attention mask 的构建
- 如何利用 FlashAttention 处理 tree 结构
- KV Cache 与 tree 结构的交互

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192) | 2022 | 原始 speculative decoding |
| [EAGLE](https://arxiv.org/abs/2401.15077) | 2024 | Feature-level draft |
| [EAGLE-2](https://arxiv.org/abs/2406.16858) | 2024 | Dynamic draft tree |
| [Medusa](https://arxiv.org/abs/2401.10774) | 2024 | Multi-head parallel draft |
| [MTP](https://arxiv.org/abs/2404.19737) | 2024 | 训练阶段 multi-token |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | 2023 | Tree-based speculative inference |

## 📁 文件清单

- [ ] `01-math-foundation.md` — 数学基础
- [ ] `02-eagle.md` — EAGLE 系列
- [ ] `03-medusa.md` — Medusa
- [ ] `04-mtp.md` — Multi-Token Prediction
- [ ] `05-draft-selection.md` — Draft Model 选择
- [ ] `06-tree-attention.md` — Tree Attention
- [ ] `exercises.md` — 动手练习（对比不同投机解码方案的加速比）
