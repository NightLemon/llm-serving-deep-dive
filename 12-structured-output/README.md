# Ch12: 结构化输出与 Constrained Decoding

> 前置知识：Ch02 前缀缓存、Ch07 投机解码、Ch08 调度与批处理

> 本章聚焦结构化输出（Structured Output）的底层实现机制——Constrained Decoding，以及它与 LLM Serving 各核心组件（调度、批处理、投机解码、KV Cache）之间的交互关系。

## 🎯 学习目标

- 理解 Constrained Decoding 的核心机制：从 JSON Schema 到 logit masking 的完整链路
- 掌握 FSM-based 与 CFG-based 两种主流约束解码方案的原理与取舍
- 深入分析 Constrained Decoding 与 continuous batching、speculative decoding、KV cache 等 serving 组件的交互影响
- 了解生产环境中结构化输出的常见模式、性能优化与监控策略

## 📑 内容大纲

### 1. Constrained Decoding 机制（01-constrained-decoding.md）

**核心问题：LLM 生成自由文本，但 API 需要结构化数据。** 从 post-hoc parsing 到 grammar-based constrained decoding 的演进路线；FSM（有限状态机）与 CFG（上下文无关文法）两种约束方式的原理与实现。

### 2. 与 Serving 系统的交互（02-serving-interaction.md）

**Constrained Decoding 不是孤立的——它深刻影响 serving pipeline 的每个环节。** 分析 logit masking 对吞吐量的影响、与 continuous batching 的兼容性、与 speculative decoding 的冲突、以及 KV cache 层面的优化机会。

### 3. 生产环境模式（03-production-patterns.md）

**从实验到生产的最后一公里。** JSON mode vs structured output 的区别、streaming 场景下的结构化输出、性能优化策略、以及关键监控指标。

## 📄 参考资料

| 项目/论文 | 核心贡献 |
|----------|----------|
| [Outlines](https://github.com/dottxt-ai/outlines) | 基于 FSM 的结构化生成框架，JSON Schema → Regex → DFA |
| [llama.cpp GBNF](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) | 基于 GBNF 文法的 CFG constrained decoding |
| [vLLM Guided Decoding](https://docs.vllm.ai/en/latest/features/structured_outputs.html) | vLLM 集成的多后端结构化输出支持 |
| [SGLang Constrained Decoding](https://arxiv.org/abs/2312.07104) | 高效的 grammar-based decoding 与 RadixAttention 集成 |
| [Guidance](https://github.com/guidance-ai/guidance) | 基于模板的结构化生成，交错文本与约束 |

## 📁 文件清单

- [x] `01-constrained-decoding.md` — Constrained Decoding 机制
- [x] `02-serving-interaction.md` — 与 Serving 系统的交互
- [x] `03-production-patterns.md` — 生产环境模式
- [x] `exercises.md` — 动手练习
