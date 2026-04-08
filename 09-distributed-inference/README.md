# Ch09: 分布式推理

> 前置知识：Ch04 PagedAttention、Ch08 调度与批处理、[gpu-ai-systems-learning/06-distributed-training](../../gpu-ai-systems-learning/06-distributed-training/)

## 🎯 学习目标

- 理解推理场景下 Tensor Parallel / Pipeline Parallel 的实现差异（与训练时的区别）
- 掌握 Expert Parallel (EP) 在 MoE 模型推理中的应用
- 理解 Data Parallel 推理的适用场景与实现
- 了解 Context Parallel 在超长上下文推理中的作用
- 走读 vLLM 的分布式推理架构

## 📑 内容大纲

### 1. Tensor Parallel 推理（01-tensor-parallel.md）

**与训练时 TP 的差异：**
- 训练时：前向 + 反向 + 梯度同步
- 推理时：只有前向，通信模式更简单
- 推理中 TP 的通信瓶颈：AllReduce / AllGather 在每层之后
- 何时 TP 是瓶颈？—— 小 batch size 时通信占比高

**vLLM 中的 TP 实现：**
- `--tensor-parallel-size` 配置
- 权重切分方式：Column Parallel + Row Parallel
- 通信后端选择：NCCL vs 自定义 AllReduce（vLLM custom_all_reduce）
- NVLink vs PCIe 对 TP 性能的影响

**调优建议：**
- A100 80GB：TP=2 用 NVLink，TP=4 需要 NVSwitch
- H100：TP=8 通过 NVSwitch 全连接
- 跨机 TP 通常不划算（延迟太高）

### 2. Pipeline Parallel 推理（02-pipeline-parallel.md）

**推理时 PP 的特点：**
- 没有反向传播，只需要前向 pipeline
- Micro-batching 策略不同于训练
- Pipeline bubble 的影响

**何时使用 PP？**
- 模型太大，无法放入单机所有 GPU（即使用了 TP）
- 与 TP 的组合：TP within node, PP across nodes

**vLLM PP 源码：**
- `--pipeline-parallel-size` 配置
- Pipeline stage 间的通信

### 3. Expert Parallel (EP) — MoE 推理（03-expert-parallel.md）

**MoE 推理的挑战：**
- 模型参数总量大（如 DeepSeek-V3: 671B 参数）
- 但每个 token 只激活部分 expert（如 8/256）
- Expert 放置策略：不同 expert 放在不同 GPU 上
- All-to-All 通信：token 路由到对应 expert 所在的 GPU

**Expert Parallel 方案：**
- Static EP：每个 GPU 持有固定数量的 expert
- Elastic EP：动态调整 expert 放置（vLLM elastic_ep）
- EPLB (Expert Parallel Load Balancing)：负载均衡策略

**源码走读：**
- `vllm/distributed/elastic_ep/` — 弹性 EP
- `vllm/distributed/eplb/` — EP 负载均衡
- `vllm/model_executor/layers/fused_moe/` — MoE layer 实现

### 4. Data Parallel 推理（04-data-parallel.md）

**何时需要 DP 推理？**
- 模型足够小，单 GPU 即可容纳
- 需要提高吞吐量 → 多个 replica 并行处理不同请求
- 与 TP 的组合：DP across replicas, TP within replica

**vLLM DP 实现：**
- `--data-parallel-size` 配置
- 数据并行的请求分发策略
- 多实例 vs 单实例多 DP

**Load Balancing 挑战：**
- 请求长度不均匀 → 某些 replica 不均衡
- Cache 亲和性：相似 prompt 的请求路由到同一 replica 以提高 cache hit

### 5. Context Parallel（05-context-parallel.md）

**超长上下文推理（>128K tokens）的挑战：**
- KV Cache 显存占用与序列长度线性增长
- Attention 计算复杂度 $O(n^2)$（或 $O(n)$ with linear attention）
- 单 GPU 可能无法存放完整的 KV Cache

**Context Parallel 方案：**
- 将序列按 context 维度切分到多个 GPU
- 每个 GPU 只处理部分 context
- 需要 Ring Attention 等通信模式

**vLLM 支持：**
- `--context-parallel-deployment` 文档
- 适用场景分析

### 6. 多维并行组合（06-hybrid-parallelism.md）

**实际部署中的并行策略组合：**
- 小模型（7B-13B）：TP=1-2, DP=N
- 中模型（70B）：TP=4-8, DP=1-2
- 大模型 MoE（600B+）：TP=8, EP=N, PP=可选
- 超长上下文：TP + CP

**决策框架：**
```
if model_fits_single_gpu:
    use DP for throughput scaling
elif model_fits_single_node:
    use TP within node + DP across nodes
elif model_is_moe:
    use TP within node + EP across nodes
else:
    use TP within node + PP across nodes
```

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Megatron-LM: Efficient Large-Scale LM Training](https://arxiv.org/abs/1909.08053) | 2019 | TP/PP 框架 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | MoE + EP 实践 |
| [Ring Attention](https://arxiv.org/abs/2310.01889) | 2023 | 序列级并行 |
| [Infinite-LLM: Efficient LLM Service for Long Context](https://arxiv.org/abs/2401.02669) | 2024 | 分布式 KV Cache 管理 |

## 📁 文件清单

- [ ] `01-tensor-parallel.md` — Tensor Parallel 推理
- [ ] `02-pipeline-parallel.md` — Pipeline Parallel 推理
- [ ] `03-expert-parallel.md` — Expert Parallel (MoE)
- [ ] `04-data-parallel.md` — Data Parallel 推理
- [ ] `05-context-parallel.md` — Context Parallel
- [ ] `06-hybrid-parallelism.md` — 多维并行组合决策
- [ ] `exercises.md` — 动手练习
