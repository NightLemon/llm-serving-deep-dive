# Ch05: Prefill-Decode 分离架构

> 前置知识：Ch01 KV Cache 深度剖析、Ch04 PagedAttention

## 🎯 学习目标

- 理解 Prefill 和 Decode 阶段的计算特征差异及其对硬件利用率的影响
- 掌握 Disaggregated Prefill-Decode 的核心架构设计
- 了解 KV Cache 传输协议（NIXL、P2P NCCL、Mooncake）
- 走读 vLLM disaggregated prefill 源码
- 能够判断何时该用分离架构、何时不该用

## 📑 内容大纲

### 1. 为什么要分离？（01-motivation.md）

**Prefill vs Decode 的根本矛盾：**
- Prefill：compute-bound，GPU 计算单元满载，显存带宽有余
- Decode：memory-bound，显存带宽满载，GPU 计算单元闲置
- 混合在一起 → 两边都无法达到最优利用率

**Chunked Prefill 的折中：**
- 将长 prefill 切成小 chunk，与 decode 交替执行
- 减少 TTFT 但不能完全解决利用率问题
- vLLM 中的实现：`--enable-chunked-prefill`

**完全分离的优势：**
- Prefill 节点：可用更少的 GPU（计算密集，batch 大）
- Decode 节点：每 GPU 服务更多请求（显存密集）
- 独立扩缩容：prefill 和 decode 节点独立按需扩展

### 2. 架构设计（02-architecture.md）

**系统组件：**
- Router / Load Balancer：将请求分发到 prefill 节点
- Prefill Worker：执行 prompt 的前向传播，生成 KV Cache
- KV Transfer Layer：将 KV Cache 从 prefill 节点传输到 decode 节点
- Decode Worker：接收 KV Cache，执行 autoregressive decode
- Metadata Service：管理节点状态、KV Cache 位置信息

**关键挑战：**
- KV Cache 传输延迟：如何不让传输成为瓶颈？
- 节点间同步：prefill 完成后如何快速通知 decode 节点？
- 故障恢复：prefill 节点挂了，decode 节点怎么办？
- 负载均衡：如何让 prefill 和 decode 节点都保持高利用率？

### 3. KV Transfer 协议（03-kv-transfer.md）

**NIXL (NVIDIA Inference Xfer Library)：**
- NVIDIA 提供的高性能数据传输库
- 支持 GPU-GPU、GPU-CPU、跨节点传输
- vLLM NixlConnector 源码分析

**P2P NCCL：**
- 基于 NCCL 的 GPU 直接通信
- 适合同机多 GPU 场景
- 延迟低但跨节点需要 NVLink/IB

**Mooncake：**
- 月之暗面开源的 KV Transfer 方案
- 利用 RDMA 实现跨节点 KV Cache 传输
- vLLM Mooncake Connector

**对比分析：**
| 方案 | 适用场景 | 带宽 | 延迟 | 复杂度 |
|------|---------|------|------|--------|
| NIXL | 通用 | 高 | 低 | 中 |
| P2P NCCL | 同机 | 很高 | 很低 | 低 |
| Mooncake | 跨机 RDMA | 高 | 中 | 高 |

### 4. vLLM Disaggregated Prefill 源码分析（04-vllm-disagg.md）

**源码走读：**
- `vllm/distributed/kv_transfer/` — KV 传输框架
- `vllm/distributed/kv_transfer/kv_connector/v1/` — V1 connector 实现
  - `nixl_connector.py`
  - `lmcache_connector.py`
  - `p2p/p2p_nccl_connector.py`
- `vllm/entrypoints/serve/disagg/` — 分离服务入口
- 配置方式：`--kv-transfer-config`

**部署拓扑示例：**
```
Prefill Node (A100 x 2, TP=2)
    ↓ KV Transfer (NIXL over InfiniBand)
Decode Node 1 (A100 x 2, TP=2) ← requests batch 1
Decode Node 2 (A100 x 2, TP=2) ← requests batch 2
```

### 5. 何时该用 / 不该用（05-when-to-use.md）

**适合分离架构的场景：**
- 长 prompt + 短生成（如 RAG、文档分析）
- 高并发、需要独立扩缩容
- 混合工作负载（部分请求 prefill 重、部分 decode 重）

**不适合分离架构的场景：**
- 短 prompt + 长生成（prefill 开销小，分离的传输成本不划算）
- 单机部署（传输延迟抵消收益）
- 低并发场景

**决策公式：**
- 分离收益 ≈ (prefill 节省的 GPU 时间) - (KV 传输延迟 + 系统复杂度成本)
- 当 prompt_length / output_length > 某个阈值时，分离才有意义

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Splitwise: Efficient Generative LLM Inference via Phase Splitting](https://arxiv.org/abs/2311.18677) | 2023 | Prefill-Decode 分离的系统设计 |
| [DistServe: Disaggregating Prefill and Decoding for LLM Serving](https://arxiv.org/abs/2401.09670) | 2024 | 跨节点分离的优化 |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | 2024 | KV-centric 架构 |
| [TetriInfer: Disaggregated Inference with Intelligent KV Cache Management](https://arxiv.org/abs/2401.11181) | 2024 | 智能 KV 调度 |

## 📁 文件清单

- [x] `01-motivation.md` — 为什么要分离
- [x] `02-architecture.md` — 架构设计
- [x] `03-kv-transfer.md` — KV Transfer 协议
- [x] `04-vllm-disagg.md` — vLLM 源码分析
- [x] `05-when-to-use.md` — 决策指南
- [x] `exercises.md` — 动手练习
