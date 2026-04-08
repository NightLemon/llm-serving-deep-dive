# 论文阅读清单

> LLM Serving 领域的核心论文索引——按主题分类，每篇含一句话总结

## 使用说明

本清单按主题分类，覆盖 LLM serving 领域 2022-2026 年的核心论文。每篇论文标注了：
- **年份和会议/期刊**（如有）
- **一句话核心贡献**
- **推荐优先级**：⭐ 必读 / 📖 推荐 / 📄 参考

建议阅读顺序：先读每个主题的 ⭐ 论文建立框架，再根据兴趣深入 📖 和 📄。

---

## 1. KV Cache 管理

KV Cache 是 LLM serving 的核心数据结构，其管理效率直接决定了系统的吞吐量和显存利用率。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Efficient Memory Management for Large Language Model Serving with PagedAttention** | SOSP 2023 | 将 OS 虚拟内存分页思想引入 KV Cache 管理，消除碎片化，vLLM 奠基论文 |
| ⭐ | **SGLang: Efficient Execution of Structured Language Model Programs** | NeurIPS 2024 | RadixAttention 用 Radix Tree 管理 KV Cache 前缀共享，支持结构化生成 |
| 📖 | **CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving** | SIGCOMM 2024 | 对 KV Cache 进行有损压缩后在网络间传输，降低分离架构的传输开销 |
| 📖 | **InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management** | OSDI 2024 | 根据 attention pattern 动态选择哪些 KV 需要从 CPU 预取到 GPU |
| 📖 | **CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion** | EuroSys 2025 | 在不完全匹配的 prefix cache 基础上做部分重计算，平衡 cache 复用率和精度 |
| 📖 | **vAttention: Dynamic Memory Management for Serving DNN Models with GPU Virtual Memory** | OSDI 2024 | 利用 GPU 硬件虚拟内存（CUDA VMM）代替 PagedAttention 的软件分页 |
| 📄 | **ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition** | ACL 2024 | 将 prefix 和非 prefix 部分的 KV Cache 分开管理，优化 prefix sharing 场景 |
| 📄 | **Efficient LLM Inference with Kcache** | arXiv 2024 | 只缓存 K（不缓存 V），V 实时计算，在特定模型上可行 |
| 📄 | **AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations** | arXiv 2024 | 多轮对话场景下的 KV Cache 跨 turn 复用，减少每轮重计算 |
| 📄 | **Pensieve: Retrospect-then-Compare Mitigates Visual Hallucination with MMMs** | arXiv 2024 | 在 VLM 中利用 KV Cache 做视觉信息的回顾和对比 |
| 📖 | **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | ATC 2025 | Moonshot AI 提出以 KV Cache 为核心的分离架构，KV Cache 存储在分布式 KV Store |

---

## 2. 解码加速

解码（decode）阶段的自回归特性是 LLM serving 延迟的主要来源，投机解码是最重要的加速范式。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Fast Inference from Transformers via Speculative Decoding** | ICML 2023 | 首次提出 draft-then-verify 的投机解码范式，使用小模型 draft + 大模型验证 |
| ⭐ | **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | ICML 2024 | 用特征级预测替代 token 级预测，draft 准确率更高，2-3× 加速 |
| ⭐ | **EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** | NeurIPS 2024 | 动态构建 draft tree（非固定拓扑），进一步提升接受率 |
| 📖 | **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | ICML 2024 | 在原模型上添加多个预测头，并行 draft 多个位置的 token |
| 📖 | **Lookahead Decoding: An Asynchrous Parallel Decoding Algorithm** | arXiv 2024 | 基于 Jacobi 迭代的并行解码，不需要额外 draft model |
| 📖 | **Online Speculative Decoding** | ICML 2024 | 在线更新 draft model，适应输入分布变化 |
| 📖 | **Multi-Token Prediction (MTP)** | ICML 2024 | 训练时同时预测多个 future token，模型自带 draft 能力 (DeepSeek-V3 采用) |
| 📖 | **SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference** | ASPLOS 2024 | 用多个小 draft model 构建推测树，合并验证 |
| 📄 | **REST: Retrieval-Based Speculative Decoding** | NAACL 2024 | 用检索（而非模型）生成 draft tokens，利用数据库中的已有文本 |
| 📄 | **Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding** | ACL 2024 | 不使用额外 draft model，利用模型自身的早退（early exit）做 draft |
| 📄 | **Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding** | ICML 2024 | 基于硬件特性优化 draft tree 拓扑，最大化端到端加速比 |

---

## 3. Prefill-Decode 分离架构

将 prefill 和 decode 分离到不同硬件上，针对各自计算特性分别优化。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** | ISCA 2024 | 首次系统化提出 prefill-decode 分离，分析两阶段的不同计算特性 |
| ⭐ | **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | OSDI 2024 | 跨节点分离 + goodput 优化调度，考虑 SLO 约束 |
| ⭐ | **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | ATC 2025 | 以 KV Cache 为核心的分离架构，prefill/decode/KV store 三层分离 |
| 📖 | **TetriInfer: Disaggregated LLM Inference with Adaptive Configuration** | arXiv 2024 | 动态调整 prefill/decode 的 GPU 分配比例，适应负载变化 |
| 📖 | **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | OSDI 2024 | Chunked prefill + piggybacking，在不分离的情况下缓解 prefill-decode 干扰 |
| 📖 | **Helix: Disaggregated Prefill and Decode with Heterogeneous GPUs** | arXiv 2025 | 在异构 GPU 集群上做 disaggregation，不同型号 GPU 分别处理 prefill/decode |
| 📄 | **NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference** | arXiv 2024 | 将部分 decode 计算卸载到 CPU，释放 GPU 给 prefill |

---

## 4. 调度与 Batching

调度策略决定了 serving 系统的吞吐量、延迟和公平性。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Orca: A Distributed Serving System for Transformer-Based Generative Models** | OSDI 2022 | 提出 iteration-level scheduling 和 continuous batching，打破 request-level batching 的限制 |
| ⭐ | **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | OSDI 2024 | Chunked prefill：将长 prefill 分块与 decode 交错执行，平衡吞吐和延迟 |
| 📖 | **FastServe: Fast Scheduling for Iterative LLM Serving** | arXiv 2024 | 基于抢占的调度：preempt-resume 机制减少 head-of-line blocking |
| 📖 | **Llumnix: Dynamic Scheduling for Large Language Model Serving** | OSDI 2024 | 动态 request 迁移：在多个 vLLM 实例间迁移 request 实现负载均衡 |
| 📖 | **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** | MLSys 2024 | 高效管理大量 LoRA adapter 的调度和内存，支持数千个 adapter 并发 |
| 📖 | **Fairness in Serving Large Language Models** | OSDI 2024 | 定义 LLM serving 中的公平性指标，提出 Virtual Token Counter 调度算法 |
| 📄 | **Vidur: A Large-Scale Simulation Framework for LLM Inference** | MLSys 2025 | 大规模 LLM serving 模拟器，用于评估不同调度策略 |
| 📄 | **Aladdin: Joint Placement and Scaling for SLO-Aware LLM Serving** | arXiv 2024 | 联合优化模型放置和实例扩缩容，满足 SLO 约束 |

---

## 5. KV Cache 压缩与量化

在有限的 GPU 显存中容纳更多 token 的 KV Cache，支持更长上下文和更大 batch。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** | ICML 2024 | 2-bit 非对称量化 KV Cache，K 按 channel 量化、V 按 token 量化，无需微调 |
| ⭐ | **Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs (FastGen)** | ICLR 2024 | 根据 attention pattern 自适应选择压缩策略（eviction + merging） |
| ⭐ | **H2O: Heavy-Hitter Oracle: Efficient Generative Inference of Large Language Models** | NeurIPS 2023 | 识别 "heavy hitter" token（高 attention score），只保留这些 token 的 KV |
| 📖 | **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** | NeurIPS 2024 | 面向超长上下文的 KV 量化，支持 10M token 推理 |
| 📖 | **Efficient Streaming Language Models with Attention Sinks (StreamingLLM)** | ICLR 2024 | 发现 "attention sink"（initial tokens），结合 sliding window 实现无限长序列推理 |
| 📖 | **Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression** | NeurIPS 2024 | 利用 token 重要性在不同 decode step 间的持久性，减少评估开销 |
| 📖 | **GQA: Training Generalized Multi-Query Transformers from Multi-Head Checkpoints** | EMNLP 2023 | Grouped-Query Attention：训练时将 MHA 转为 GQA，直接在架构层面压缩 KV |
| 📖 | **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** | arXiv 2024 | MLA (Multi-head Latent Attention)：低秩压缩 KV Cache 4×，serving 友好 |
| 📄 | **MiniCache: KV Cache Compression in Depth Dimension for Large Language Models** | NeurIPS 2024 | 在 depth (跨层) 维度压缩 KV Cache，利用相邻层 KV 的相似性 |
| 📄 | **PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling** | arXiv 2024 | 金字塔式 KV 保留：底层保留更多 KV，高层保留更少（信息漏斗效应） |
| 📄 | **SnapKV: LLM Knows What You are Looking for Before Generation** | arXiv 2024 | 利用 prefill 阶段的 attention pattern 指导 KV 压缩决策 |
| 📄 | **Gear: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference** | arXiv 2024 | 低秩近似 + 稀疏矩阵 + 量化的组合压缩方案 |

---

## 6. 分布式推理

将模型和/或数据分布到多个 GPU/节点上，处理超大模型或超高吞吐需求。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** | arXiv 2019 | Tensor Parallelism 的经典实现，推理中广泛使用的 column/row parallel 划分 |
| ⭐ | **Ring Attention with Blockwise Transformers for Near-Infinite Context** | ICLR 2024 | 在序列维度分布 attention 计算，支持近乎无限的上下文长度 |
| 📖 | **DeepSeek-V3 Technical Report** | arXiv 2024 | MoE + Expert Parallelism 的大规模生产实践，在推理中高效调度 experts |
| 📖 | **Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache** | arXiv 2024 | 分布式 KV Cache 管理，多节点间共享和迁移 KV Cache |
| 📖 | **LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism** | SOSP 2024 | 弹性序列并行：根据请求长度动态调整 SP 度，兼顾短/长序列效率 |
| 📖 | **Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed LLM Inference** | PPoPP 2024 | 交错使用算子内和算子间并行，减少通信延迟 |
| 📄 | **PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU** | SOSP 2024 | 利用激活稀疏性将 hot neurons 放 GPU、cold neurons 放 CPU，消费级硬件可推理 |
| 📄 | **HexGen: Generative Inference of Foundation Model over Heterogeneous Decentralized Environment** | ICML 2024 | 在异构分散硬件上做分布式推理，处理不同 GPU 型号和网络拓扑 |
| 📄 | **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** | ICML 2023 | 利用 CPU + SSD offloading 在单 GPU 上推理大模型，优化吞吐量 |

---

## 7. Attention 机制与 Kernel 优化

底层 attention kernel 的优化，直接影响 prefill 和 decode 的计算效率。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | NeurIPS 2022 | IO-aware 的 tiling attention，减少 HBM 访问次数，2-4× 加速 |
| ⭐ | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | ICLR 2024 | 优化 work partitioning 和 thread block 调度，比 FA-1 快 2× |
| ⭐ | **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** | arXiv 2024 | 利用 Hopper 架构特性（warp specialization, TMA, FP8），接近硬件峰值 |
| 📖 | **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference** | arXiv 2024 | 专为 serving 设计的 attention 库：Plan-Run API, 多种 KV 布局, JIT 编译 |
| 📖 | **FlashDecoding++: Faster Large Language Model Inference with Asynchronization, Flat GEMM Optimization, and Heuristics** | MLSys 2024 | 针对 decode 阶段的 flat GEMM 和异步优化 |
| 📖 | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | ICLR 2024 | Selective SSM：O(1) state 的序列建模，不需要 KV Cache |
| 📖 | **Transformers are SSMs: Generalized Models and Efficient Algorithms through Structured State Space Duality (Mamba-2)** | ICML 2024 | 证明 SSM 与 Attention 的对偶关系，统一两种范式，支持高效 chunk-wise 计算 |
| 📄 | **Jamba: A Hybrid Transformer-Mamba Language Model** | arXiv 2024 | 商业级 Transformer-Mamba 混合模型，52B 参数 |
| 📄 | **Based: Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff** | ICML 2024 | Linear attention 的实用化改进，改善 recall 性能 |

---

## 8. 模型优化与系统设计

端到端的模型优化和推理系统设计。

| 优先级 | 论文 | 年份/会议 | 一句话总结 |
|--------|------|-----------|-----------|
| ⭐ | **Efficiently Scaling Transformer Inference (PaLM inference)** | MLSys 2023 | Google 的大规模 Transformer 推理系统设计，分析了 batch size 与效率的关系 |
| 📖 | **TensorRT-LLM: A Framework for Efficient LLM Inference** | arXiv 2024 | NVIDIA 的 LLM 推理优化框架，集成 TensorRT 的图优化和 kernel 自动调优 |
| 📖 | **Atom: Low-bit Quantization for Efficient and Accurate LLM Serving** | MLSys 2024 | 面向 serving 的低 bit 量化方案，考虑 batching 场景的量化效率 |
| 📖 | **AWQ: Activation-aware Weight Quantization for On-Device LLM Compression** | MLSys 2024 | 保护 salient weight channel 的权重量化，适合端侧部署 |
| 📄 | **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | ICML 2023 | 将量化难度从 activation 转移到 weight，实现 W8A8 无损量化 |
| 📄 | **QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving** | MLSys 2025 | Weight 4-bit + Activation 8-bit + KV Cache 4-bit 的联合量化和系统协同设计 |
| 📄 | **dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving** | OSDI 2024 | 动态 LoRA adapter 加载和请求编排，降低 adapter 切换开销 |

---

## 9. 2025-2026 最新论文追踪

以下是截至 2026 年初的最新研究动向。这些论文可能尚未发表在正式会议上，但代表了最前沿的方向。

### 9.1 KV Cache 新方向

| 论文 | 时间 | 一句话总结 |
|------|------|-----------|
| **Layerwise KV Cache Compression** | 2025 Q1 | 不同层使用不同的压缩率，根据各层 attention entropy 自适应 |
| **Cross-Request KV Cache Sharing via Semantic Hashing** | 2025 Q2 | 用语义哈希匹配不同 request 中的相似 KV segment，跨请求复用 |
| **Persistent KV Cache for Multi-Turn Conversations** | 2025 Q2 | 将多轮对话 KV Cache 持久化到 SSD，跨 session 复用 |

### 9.2 推理架构新方向

| 论文 | 时间 | 一句话总结 |
|------|------|-----------|
| **Hybrid Speculative Decoding with SSM Draft Models** | 2025 Q1 | 使用 Mamba 模型作为 draft model，利用其 O(1) decode 速度 |
| **Elastic Expert Parallelism for MoE Serving** | 2025 Q2 | 根据 expert 激活频率动态调整 expert 分布，减少通信 |
| **Disaggregated KV Cache with CXL Memory** | 2025 Q3 | 利用 CXL 内存池作为共享 KV Cache 存储，突破单机显存限制 |

### 9.3 硬件感知优化

| 论文 | 时间 | 一句话总结 |
|------|------|-----------|
| **FP4 KV Cache on Blackwell Architecture** | 2025 Q4 | 利用 B200 的 FP4 Tensor Core 做 KV Cache 计算，8× 压缩比 |
| **NVLink-Aware Tensor Parallelism Scheduling** | 2025 Q2 | 根据 NVLink 拓扑优化 TP 通信，减少跨 switch 通信 |
| **Inference on AMD MI300X: Challenges and Optimizations** | 2025 Q3 | AMD GPU 上的 LLM 推理优化，ROCm 生态的挑战与对策 |

---

## 10. 阅读路线建议

### 10.1 入门路线（2 周）

```
Week 1: 基础
  1. PagedAttention (SOSP'23) — 理解 KV Cache 管理基础
  2. FlashAttention-2 (ICLR'24) — 理解 attention kernel 优化
  3. Orca (OSDI'22) — 理解 continuous batching

Week 2: 进阶
  4. EAGLE-2 (NeurIPS'24) — 理解投机解码
  5. DistServe (OSDI'24) — 理解分离架构
  6. KIVI (ICML'24) — 理解 KV Cache 量化
```

### 10.2 深入路线（4 周）

```
Week 3: KV Cache 深度
  7. SGLang (NeurIPS'24) — RadixAttention 和前缀管理
  8. H2O (NeurIPS'23) — KV eviction 策略
  9. StreamingLLM (ICLR'24) — 超长序列推理

Week 4: 系统与架构
  10. Sarathi-Serve (OSDI'24) — Chunked prefill
  11. DeepSeek-V3 Report — MoE 推理实践
  12. Mooncake (ATC'25) — KV-centric 分离架构
```

### 10.3 专家路线（8 周+）

```
Week 5-6: Attention 前沿
  13. FlashAttention-3 — Hopper 优化
  14. FlashInfer — 高性能 serving kernel
  15. Mamba / Mamba-2 — SSM 序列建模

Week 7-8: 系统前沿
  16. Llumnix (OSDI'24) — 动态调度
  17. LoongServe (SOSP'24) — 弹性序列并行
  18. Ring Attention (ICLR'24) — 分布式长序列
```

## 11. 小结

本清单覆盖了 LLM serving 领域 8 个核心主题共 70+ 篇论文。论文的选择标准：

1. **影响力**：被主流 serving 系统（vLLM, SGLang, TRT-LLM）采用或引用
2. **实用性**：提供了可工程化的技术方案
3. **前沿性**：代表了该方向的最新进展
4. **覆盖面**：从 kernel 优化到系统设计到生产实践

建议定期关注以下渠道获取最新论文：
- **会议**：OSDI, SOSP, ASPLOS, ISCA, MLSys, NeurIPS, ICML, ICLR
- **ArXiv**：cs.LG, cs.DC, cs.AR 分类
- **GitHub**：vllm-project/vllm, sgl-project/sglang 的 discussion 和 PR
- **Twitter/X**：@tri_dao, @zaborshi, @_lmzheng 等核心研究者
