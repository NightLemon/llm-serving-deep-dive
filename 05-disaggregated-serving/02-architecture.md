# 分离架构设计

> 本节深入 Prefill-Decode 分离系统的架构设计，解读 Splitwise 和 DistServe 两篇奠基性论文。

## 1. 系统组件总览

一个完整的 Disaggregated Serving 系统由以下五大组件构成：

```
                    ┌──────────────────────────────────────┐
                    │           Metadata Service           │
                    │  (节点注册, KV Cache 位置, 健康检查)   │
                    └──────────┬────────────┬──────────────┘
                               │            │
       ┌───────────────────────┴───┐    ┌───┴───────────────────────┐
       │                           │    │                           │
  ┌────┴────┐                      │    │                     ┌─────┴────┐
  │ Client  │───► ┌────────────┐   │    │  ┌──────────────┐   │  Client  │
  │ Request │     │   Router   │───┤    ├──│ Decode Worker │◄──│ Response │
  └─────────┘     │   / LB     │   │    │  │    Pool       │   └──────────┘
                  └─────┬──────┘   │    │  └──────▲───────┘
                        │          │    │         │
                        ▼          │    │         │
                 ┌──────────────┐  │    │  ┌──────┴───────┐
                 │Prefill Worker│  │    │  │ KV Transfer  │
                 │    Pool      │──┼────┼─►│    Layer      │
                 └──────────────┘  │    │  └──────────────┘
                                   │    │
                    ┌──────────────┴────┴──────────────┐
                    │     KV Transfer Layer            │
                    │  (NIXL / NCCL P2P / Mooncake)    │
                    └──────────────────────────────────┘
```

### 1.1 Router / Load Balancer

**职责：**
- 接收客户端请求，将 prefill 任务分发到合适的 prefill worker
- 感知各 worker 的负载状态，做智能路由
- 在 prefill 完成后，将 decode 任务路由到合适的 decode worker

**路由策略：**

```python
# 简化的路由逻辑
class DisaggRouter:
    def route_prefill(self, request: Request) -> PrefillWorker:
        """选择 prefill worker"""
        # 策略1: 最少负载
        # 策略2: 亲和性路由（相似 prefix 的请求路由到同一节点，利用 prefix cache）
        # 策略3: prompt 长度感知（长 prompt 分配给空闲节点）
        candidates = self.prefill_pool.get_available()
        return min(candidates, key=lambda w: w.pending_tokens)
    
    def route_decode(self, kv_metadata: KVMeta) -> DecodeWorker:
        """选择 decode worker"""
        # 策略1: 最少活跃序列
        # 策略2: KV Cache 就近（选择与 prefill 节点网络距离近的 decode 节点）
        candidates = self.decode_pool.get_available()
        return min(candidates, key=lambda w: w.active_sequences)
```

### 1.2 Prefill Worker

**职责：**
- 加载模型权重
- 接收 prompt，执行完整的 prefill 前向传播
- 生成 KV Cache，通过 KV Transfer Layer 传输到 decode worker
- 传输完成后释放本地 KV Cache 空间

**关键特征：**
- 优化目标是 **最大化 prefill 吞吐（tokens/s）**
- 通常使用较大的 batch size
- KV Cache 空间需求相对较小（用完即释放）
- 可以使用更高的 tensor parallelism degree 来加速单次 prefill

### 1.3 Decode Worker

**职责：**
- 加载模型权重
- 接收从 prefill worker 传来的 KV Cache
- 执行 autoregressive decode，逐 token 生成
- 管理大量并发 decode 序列的 KV Cache

**关键特征：**
- 优化目标是 **最大化并发序列数** 和 **稳定的 TPOT**
- KV Cache 占据大量显存
- 不执行 prefill，所以没有 TPOT spike

### 1.4 KV Transfer Layer

**职责：**
- 将 prefill worker 生成的 KV Cache 高效传输到 decode worker
- 支持多种传输协议：NIXL、P2P NCCL、Mooncake 等
- 处理传输失败和重试

**关键挑战：**

KV Cache 的数据量可能非常大。以 Llama-3-70B 为例：

```
每层 KV Cache 大小:
  K: [num_kv_heads × head_dim × seq_len] × dtype_size
   = [8 × 128 × seq_len] × 2 bytes (FP16)
   = 2048 × seq_len bytes per layer

  K + V = 4096 × seq_len bytes per layer

总计 (80 layers):
  KV Cache = 80 × 4096 × seq_len bytes
           = 327,680 × seq_len bytes

对于 seq_len=4096:
  KV Cache ≈ 1.28 GB

对于 seq_len=32768:
  KV Cache ≈ 10.24 GB
```

以 200 Gbps InfiniBand 连接为例，传输 1.28 GB 需要约 **51 ms**，传输 10.24 GB 需要约 **410 ms**。这个传输延迟会直接加到 TTFT 上，是分离架构最核心的开销。

### 1.5 Metadata Service

**职责：**
- Worker 注册和健康检查
- KV Cache 位置索引（哪个 KV Cache 在哪个节点上）
- 协调 prefill → decode 的切换流程
- 集群拓扑管理

## 2. 关键挑战

### 2.1 KV Cache 传输延迟

这是分离架构面临的**最核心挑战**。传输延迟直接叠加到 TTFT 上：

$$
\text{TTFT}_{disagg} = T_{prefill} + T_{kv\_transfer} + T_{decode\_first\_token}
$$

而混合部署的 TTFT 仅为：

$$
\text{TTFT}_{mixed} = T_{prefill} + T_{decode\_first\_token}
$$

优化手段：
1. **流水线传输**：prefill 一边计算一边传输已完成层的 KV Cache，而非等全部计算完再传
2. **压缩传输**：对 KV Cache 做量化压缩后再传输（如 FP16→INT4）
3. **高速互联**：使用 NVLink、InfiniBand、RoCE 等高带宽低延迟网络

```
无流水线：
  Prefill: ████████████████
  Transfer:                 ████████████████
  Decode:                                   █
  TTFT = T_prefill + T_transfer

有流水线（逐层传输）：
  Prefill: ████████████████
  Transfer:   ████████████████
  Decode:                     █
  TTFT = T_prefill + T_transfer_last_layers  (显著减少！)
```

### 2.2 节点间同步

Prefill 完成后需要通知 decode worker 开始工作。同步机制的延迟直接影响 TTFT：

- **Pull 模式**：Decode worker 轮询 metadata service 是否有新的 KV Cache 可用
- **Push 模式**：Prefill worker 主动通知 decode worker（更低延迟）
- **混合模式**：通过共享消息队列（如 Redis、NATS）实现异步通知

### 2.3 故障恢复

在分离架构中，故障场景更加复杂：

| 故障类型 | 影响 | 恢复策略 |
|---------|------|---------|
| Prefill worker 故障 | 正在 prefill 的请求失败 | Router 将请求重新路由到其他 prefill worker |
| Decode worker 故障 | 所有正在 decode 的序列丢失 | 需要重新 prefill 或从 checkpoint 恢复 |
| KV Transfer 失败 | Prefill 完成但 KV Cache 未到达 | 重试传输或在 decode 端重做 prefill |
| Metadata Service 故障 | 路由信息不可用 | 使用本地缓存的路由表 + 快速恢复 |

### 2.4 负载均衡

分离架构需要更精细的负载均衡：

```
理想状态：Prefill 和 Decode 节点利用率都在 70-90%

实际挑战：
  - 请求到达是随机的，prefill 和 decode 负载波动大
  - 长 prompt 请求会导致 prefill 节点负载不均
  - Decode 序列长度不同，完成时间不确定

解决方案：
  - 动态调整 Prefill/Decode 节点比例
  - 使用「混合节点」做缓冲（既能 prefill 也能 decode）
  - 基于预测模型估算 decode 时长
```

## 3. Splitwise 论文解读

> Splitwise: Efficient Generative LLM Inference via Phase Splitting (Patel et al., 2023)

### 3.1 核心思想

Splitwise 是最早系统性提出 Prefill-Decode 分离的论文之一。核心观察：

> "Prompt processing and token generation phases have fundamentally different computational characteristics, and treating them as separate workloads enables better resource utilization."

### 3.2 关键设计

**Phase-level Scheduling：**

Splitwise 将推理过程显式拆分为两个阶段，每个阶段由专门的机器池执行：

```
┌───────────────┐     ┌──────────────┐     ┌───────────────┐
│  Prompt Pool  │────►│ KV Transfer  │────►│ Generation    │
│  (Prefill)    │     │              │     │ Pool (Decode) │
│               │     │ PCIe / IB /  │     │               │
│ High compute  │     │ NVLink       │     │ High mem-BW   │
└───────────────┘     └──────────────┘     └───────────────┘
```

**Machine Allocation：**

Splitwise 提出了一个配比优化公式，根据工作负载特征决定 prefill 和 decode 的机器比例：

$$
R_{opt} = \frac{N_{prefill}}{N_{decode}} = \frac{\bar{T}_{prefill} \times \lambda}{\bar{T}_{decode} \times \bar{L}_{output}}
$$

其中：
- $\lambda$：请求到达速率
- $\bar{T}_{prefill}$：平均 prefill 时间
- $\bar{T}_{decode}$：平均每 token decode 时间
- $\bar{L}_{output}$：平均输出长度

**异构硬件支持：**

Splitwise 指出 prefill 和 decode 可以使用不同的 GPU 型号：
- Prefill 节点：选择 FLOPS/$ 最优的 GPU
- Decode 节点：选择 Memory-BW/$ 最优的 GPU

### 3.3 实验结果

在 Splitwise 的实验中（LLaMA-2-70B，A100 集群）：
- 对比混合部署，**TTFT 降低 20-40%**
- **TPOT P99 降低 50-70%**
- 在相同 SLO 约束下，**吞吐提升 1.4x**
- GPU 利用率从 ~40% 提升到 ~65%

### 3.4 局限

- 论文主要考虑了 **同构集群**内的分离，异构集群的调度更复杂
- **KV Transfer 开销**在长序列场景下可能抵消分离收益
- 没有深入讨论**故障恢复**机制

## 4. DistServe 论文解读

> DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Zhong et al., 2024)

### 4.1 核心思想

DistServe 在 Splitwise 的基础上进一步优化，核心贡献是 **Goodput-driven placement**——不仅分离 prefill 和 decode，还要为每个阶段选择最优的并行策略。

### 4.2 Goodput 定义

DistServe 提出了 **Goodput** 的概念，即在满足 SLO 约束下的有效吞吐：

$$
\text{Goodput} = \lambda \times \Pr[\text{TTFT} < \text{SLO}_{TTFT} \wedge \text{TPOT} < \text{SLO}_{TPOT}]
$$

Goodput = 请求到达率 × 满足 SLO 的请求比例。目标是**最大化 Goodput**。

### 4.3 关键设计

**阶段级并行策略优化：**

DistServe 为 prefill 和 decode 阶段独立选择最优的 parallelism 配置：

```
Prefill 节点:
  - 高 TP degree (e.g., TP=4) → 加速单次 prefill
  - 较小 PP degree → 减少 pipeline 延迟
  - 目标：最小化 TTFT

Decode 节点:
  - 低 TP degree (e.g., TP=2) → 每组 GPU 服务更多序列
  - 可用 PP → 增加总 batch size
  - 目标：最小化 TPOT，最大化并发
```

**Placement Algorithm：**

DistServe 使用搜索算法在可能的（TP_p, PP_p, TP_d, PP_d, N_p, N_d）配置空间中找到 Goodput 最优的方案：

```python
# 简化的配置搜索逻辑
def find_optimal_placement(total_gpus, model, workload, slo):
    best_goodput = 0
    best_config = None
    
    for tp_p in [1, 2, 4, 8]:           # Prefill TP degree
        for pp_p in [1, 2, 4]:          # Prefill PP degree
            for tp_d in [1, 2, 4, 8]:   # Decode TP degree
                for pp_d in [1, 2, 4]:  # Decode PP degree
                    gpus_per_prefill = tp_p * pp_p
                    gpus_per_decode = tp_d * pp_d
                    
                    for n_p in range(1, total_gpus // gpus_per_prefill + 1):
                        n_d = (total_gpus - n_p * gpus_per_prefill) // gpus_per_decode
                        if n_d <= 0:
                            continue
                        
                        goodput = simulate_goodput(
                            model, workload, slo,
                            tp_p, pp_p, n_p,
                            tp_d, pp_d, n_d
                        )
                        
                        if goodput > best_goodput:
                            best_goodput = goodput
                            best_config = (tp_p, pp_p, n_p, tp_d, pp_d, n_d)
    
    return best_config
```

**流水线 KV Transfer：**

DistServe 实现了逐层 KV Cache 流水线传输：

```
Layer 0 prefill │████│
Layer 0 transfer│    │██│
Layer 1 prefill │    │████│
Layer 1 transfer│    │    │██│
...
Decode start    │    │    │    │...│█
                                    ↑ 比等全部层完成再传输更早开始 decode
```

### 4.4 实验结果

DistServe 在多种模型和工作负载上的结果：

| 模型 | 工作负载 | Goodput 提升 (vs vLLM) | Goodput 提升 (vs Splitwise) |
|------|---------|----------------------|---------------------------|
| LLaMA-2-70B | 聊天 | 4.48× | 1.32× |
| LLaMA-2-70B | 代码 | 3.2× | 1.18× |
| OPT-66B | 摘要 | 2.18× | 1.05× |

DistServe 相比 Splitwise 的额外提升主要来自：
1. 阶段级并行策略优化
2. 更精细的 Goodput-driven 配置搜索
3. 流水线 KV Transfer

### 4.5 DistServe vs Splitwise 对比

| 维度 | Splitwise | DistServe |
|------|-----------|-----------|
| 分离粒度 | 机器级分离 | 机器级 + 并行策略级分离 |
| 优化目标 | 降低延迟、提升利用率 | 最大化 Goodput（SLO-aware） |
| 并行策略 | Prefill/Decode 使用相同 TP/PP | 各自独立选择最优 TP/PP |
| KV Transfer | 全量传输 | 流水线逐层传输 |
| 配置搜索 | 手动/启发式 | 自动搜索最优配置 |

## 5. 部署拓扑示例

### 5.1 单机多卡分离

```
┌─────────────── 单机 (8× A100 SXM) ──────────────┐
│                                                    │
│  Prefill Group (GPU 0-1, TP=2)                     │
│  ┌─────────┐  ┌─────────┐                          │
│  │ GPU 0   │──│ GPU 1   │  NVLink                  │
│  │ A100    │  │ A100    │                          │
│  └────┬────┘  └────┬────┘                          │
│       │            │                               │
│       └─────┬──────┘                               │
│             │ NVLink (KV Transfer, ~600 GB/s)      │
│             ▼                                      │
│  Decode Group (GPU 2-7, 3×TP=2)                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ... │
│  │ GPU 2  │─│ GPU 3  │ │ GPU 4  │─│ GPU 5  │     │
│  └────────┘ └────────┘ └────────┘ └────────┘     │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 5.2 多机跨节点分离

```
┌────── Prefill Node ──────┐          ┌────── Decode Node 1 ─────┐
│  A100 ×4, TP=4            │          │  A100 ×4, TP=2 ×2 groups │
│  高 FLOPS 利用率           │          │  高显存利用率              │
│                            │          │                           │
│  Prefill Worker            │   IB     │  Decode Worker ×2         │
│  batch_size=64             │──200G───►│  active_seq=256 per group │
└────────────────────────────┘          └───────────────────────────┘
                                                    
                              IB       ┌────── Decode Node 2 ─────┐
                             200G      │  A100 ×4, TP=2 ×2 groups │
                            ─────────►│  Decode Worker ×2         │
                                       └───────────────────────────┘
```

### 5.3 混合弹性部署

实际生产中，可以保留部分节点作为**混合节点**，根据负载动态切换角色：

```
固定 Prefill 节点:   P1, P2        (始终做 prefill)
固定 Decode 节点:    D1, D2, D3    (始终做 decode)
混合节点:            H1, H2        (根据负载切换)

负载变化：
  高 Prefill 压力 → H1, H2 切换为 Prefill
  高 Decode 压力  → H1, H2 切换为 Decode
  均衡负载        → H1 做 Prefill, H2 做 Decode
```

## 6. 小结

| 组件 | 职责 | 关键设计决策 |
|------|------|-------------|
| Router | 请求路由 | 负载感知、亲和性路由 |
| Prefill Worker | 执行 prefill | 大 batch、高 TP |
| KV Transfer | 传输 KV Cache | 流水线传输、高速互联 |
| Decode Worker | 执行 decode | 高并发、稳定 TPOT |
| Metadata Service | 协调管理 | 健康检查、KV 位置索引 |

**论文启示：**
- Splitwise 提出了分离的基本框架和配比公式
- DistServe 进一步优化了并行策略选择和 Goodput 指标
- 两篇论文都验证了分离架构在高负载场景下的显著收益

> **下一节**：[03-kv-transfer.md](03-kv-transfer.md) — 深入 KV Cache 传输协议：NIXL、P2P NCCL、Mooncake。
