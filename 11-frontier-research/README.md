# Ch11: 前沿研究

> 前置知识：Ch03 KV Cache 压缩、Ch07 投机解码、Ch10 生产环境实践

> 本章跟踪 LLM Serving 领域最新的研究进展，定期更新。

## 🎯 学习目标

- 跟踪 LLM 推理优化领域的最新研究动态
- 理解每篇论文/技术的核心思想和适用场景
- 建立对技术演进方向的判断力

## 📑 内容大纲

### 1. Hybrid KV Cache Manager（01-hybrid-kv-cache.md）

**问题：不同类型的 attention 需要不同的 KV Cache 管理策略**

- vLLM 的 Hybrid KV Cache Manager 设计
- 支持混合架构模型（如 Transformer + Mamba 混合层）
- 不同 layer group 使用不同的 cache 策略
- 源码：`vllm/v1/core/kv_cache_coordinator.py`

### 2. Attention 架构创新（02-attention-innovations.md）

**FlashInfer：**
- 高性能 attention kernel 库
- 支持 PagedKV、Ragged Tensor 等多种 KV 布局
- 替代 FlashAttention 的选择

**Multi-head Latent Attention (MLA) 后续发展：**
- 更多模型采用类似 MLA 的 KV 压缩方案
- 与 FlashAttention/FlashInfer 的兼容性

**Linear Attention / State Space Models：**
- Mamba / Mamba-2 的 KV Cache 等价物
- 与 Transformer 混合架构的 serving 挑战

### 3. 推理编译优化（03-compilation.md）

**torch.compile for inference：**
- vLLM 对 `torch.compile` 的集成
- Fusion pass：将多个 op 融合为一个 kernel
- CUDA Graph：减少 kernel launch overhead
- 与 FlashAttention 的交互

**TensorRT-LLM vs vLLM 编译策略对比：**
- TRT-LLM：静态图优化（编译时优化）
- vLLM + torch.compile：动态图 + JIT 编译

### 4. 推理成本前沿（04-cost-frontier.md）

**Flex Inference / Priority Inference（Google）：**
- 非实时任务使用闲置 GPU 资源
- 更低价格，但延迟不保证

**Batch API（OpenAI / Anthropic）：**
- 异步批量处理，50% 成本折扣
- 24 小时内返回结果

**成本下降趋势分析：**
- 2023-2026 每百万 token 成本变化曲线
- 驱动成本下降的技术因素

### 5. 论文阅读清单（05-paper-list.md）

**按主题分类的最新论文索引，每篇含一句话总结。**

#### KV Cache 管理
| 论文 | 一句话 |
|------|--------|
| PagedAttention (SOSP'23) | 用 OS 分页管理 KV Cache |
| SGLang RadixAttention | 用 Radix Tree 管理共享前缀 |
| CacheGen | KV Cache 压缩传输 |
| InfiniGen | 选择性 prefetch KV Cache |

#### 解码加速
| 论文 | 一句话 |
|------|--------|
| Speculative Decoding (ICML'23) | Draft-verify 范式 |
| EAGLE / EAGLE-2 | Feature-level 投机 |
| Medusa | 多头并行 draft |
| Lookahead Decoding | Jacobi 迭代式并行解码 |

#### 分离架构
| 论文 | 一句话 |
|------|--------|
| Splitwise | Prefill-decode phase splitting |
| DistServe | 跨节点分离的系统优化 |
| Mooncake | KV-centric 分离架构 |

#### 调度优化
| 论文 | 一句话 |
|------|--------|
| Orca (OSDI'22) | Iteration-level scheduling |
| Sarathi-Serve | Chunked prefill 系统化 |
| FastServe | Preemption-based scheduling |

#### 压缩与量化
| 论文 | 一句话 |
|------|--------|
| KIVI | 2-bit 非对称 KV 量化 |
| KVQuant | 面向超长上下文的 KV 量化 |
| H2O | Heavy-hitter KV 驱逐 |
| StreamingLLM | Attention sink + sliding window |

#### 分布式推理
| 论文 | 一句话 |
|------|--------|
| Ring Attention | 序列维度并行 |
| Infinite-LLM | 分布式 KV Cache |
| DeepSeek-V3 Report | MoE + EP 生产实践 |

### 6. 技术趋势展望（06-trends.md）

**短期趋势（2026-2027）：**
- KV Cache 压缩成为标配（FP8 KV 已普及）
- Disaggregated serving 在大规模部署中广泛采用
- Speculative decoding 集成到更多模型的训练过程（MTP）
- 超长上下文（>1M tokens）推理的标准化

**中期趋势（2027-2028）：**
- 硬件感知的 KV Cache 管理（存算一体芯片？）
- 跨模型的 KV Cache 共享（同族模型间复用 KV）
- 端侧推理与云端协同（KV Cache 在设备间流动）

## 📄 参考论文

> 完整论文清单见 [05-paper-list.md](05-paper-list.md)，以下列出本章各节涉及的核心论文。

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) | 2024 | Transformer + Mamba 混合架构，推动 Hybrid KV Cache 需求 |
| [FlashInfer: Efficient and Customizable Attention Engine](https://arxiv.org/abs/2501.01005) | 2025 | 高性能可定制 attention kernel 库 |
| [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | 2024 | MLA 架构，KV Cache 压缩代表作 |

## 📁 文件清单

- [x] `01-hybrid-kv-cache.md` — Hybrid KV Cache Manager
- [x] `02-attention-innovations.md` — Attention 架构创新
- [x] `03-compilation.md` — 推理编译优化
- [x] `04-cost-frontier.md` — 推理成本前沿
- [x] `05-paper-list.md` — 论文阅读清单（定期更新）
- [x] `06-trends.md` — 技术趋势展望
- [x] `exercises.md` — 动手练习（论文精读与趋势分析）
