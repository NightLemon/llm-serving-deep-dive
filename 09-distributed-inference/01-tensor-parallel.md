# Tensor Parallel 推理

> 本节深入分析推理场景下 Tensor Parallelism (TP) 的实现原理、通信开销、vLLM 源码实现以及调优策略。

## 1. 推理 TP 与训练 TP 的核心差异

Tensor Parallelism 最早由 Megatron-LM 提出，将单个 Transformer 层的权重切分到多个 GPU 上。训练时 TP 需要处理前向、反向和梯度同步三个阶段，而推理时的 TP 有本质性简化。

### 1.1 训练时 TP 的通信

```
训练 TP 通信模式（每层）:
  Forward:   AllReduce (or Reduce-Scatter + AllGather)
  Backward:  AllReduce (梯度同步)
  Optimizer: 每个 rank 独立更新自己持有的权重分片

每层通信次数: 2 次 AllReduce（前向 + 反向）
```

### 1.2 推理时 TP 的通信

```
推理 TP 通信模式（每层）:
  Forward:   AllReduce (or AllGather)
  Backward:  无
  Optimizer: 无

每层通信次数: 1-2 次集合通信（仅前向）
```

关键差异总结：

| 维度 | 训练 TP | 推理 TP |
|------|---------|---------|
| 计算阶段 | 前向 + 反向 | 仅前向 |
| 每层通信次数 | 2x AllReduce | 1-2x AllReduce/AllGather |
| 权重更新 | 每步更新 | 权重只读 |
| 内存压力 | 权重 + 梯度 + 优化器状态 | 仅权重 + KV Cache |
| Batch 特征 | 固定 batch size | 动态 batch（continuous batching） |
| 延迟敏感度 | 看吞吐 | 看延迟（尤其 decode 阶段） |

推理 TP 最重要的特性是**只有前向传播**，这意味着通信量减半，且不需要保存中间激活值用于反向传播，显存可以全部用于 KV Cache。

## 2. 推理中 TP 的通信瓶颈

### 2.1 通信模式详解

在标准 Transformer 层中，推理 TP 的通信发生在两个位置：

```
Input
  │
  ▼
┌─────────────────┐
│  QKV Projection  │  ← Column Parallel（无通信）
│  (切分 head 维度) │
└────────┬────────┘
         │
┌────────▼────────┐
│   Attention      │  ← 各 rank 独立计算自己的 head
└────────┬────────┘
         │
┌────────▼────────┐
│  Output Proj     │  ← Row Parallel
└────────┬────────┘
         │
    AllReduce ①     ← 第一次通信：聚合 attention 输出
         │
┌────────▼────────┐
│  MLP Gate/Up     │  ← Column Parallel（无通信）
│  (切分 hidden)   │
└────────┬────────┘
         │
┌────────▼────────┐
│  MLP Down Proj   │  ← Row Parallel
└────────┬────────┘
         │
    AllReduce ②     ← 第二次通信：聚合 MLP 输出
         │
Output
```

**每个 Transformer 层需要 2 次 AllReduce 操作**。对于一个 L 层的模型，一次完整的前向传播需要 `2L` 次 AllReduce。

### 2.2 AllReduce 通信量

单次 AllReduce 的数据量：

```
AllReduce 数据量 = 2 * (tp_world_size - 1) / tp_world_size * batch_tokens * hidden_size * sizeof(dtype)
```

以 Llama-3-70B（hidden_size=8192, 80 层, bf16）为例：

```
每次 AllReduce: 2 * (7/8) * batch_tokens * 8192 * 2 bytes
             ≈ 28672 * batch_tokens bytes

每层 2 次: 57344 * batch_tokens bytes
80 层总计: 4,587,520 * batch_tokens bytes

当 batch_tokens = 1（单 token decode）:
  总通信量 ≈ 4.4 MB

当 batch_tokens = 256:
  总通信量 ≈ 1.1 GB
```

### 2.3 通信延迟分析

通信延迟由两部分组成：

```
T_comm = T_latency + T_bandwidth
       = α + (data_size / bandwidth)

其中:
  α = 发起通信的固定延迟（kernel launch, synchronization 等）
  bandwidth = 有效互联带宽
```

**关键观察：Decode 阶段是延迟瓶颈。**

在 decode 阶段，每次只生成 1 个 token（每个请求），数据量很小，但 AllReduce 的固定延迟 α 无法消除。假设 NVLink 的 α ≈ 5-10 μs：

```
Decode 阶段每层通信:
  数据量 = batch_size * hidden_size * 2 bytes（很小）
  延迟 ≈ 2 * α（两次 AllReduce 的固定开销）
       ≈ 10-20 μs

80 层总延迟: 800-1600 μs = 0.8-1.6 ms
```

这 0.8-1.6ms 的通信延迟对于 decode 阶段来说是非常显著的，因为 decode 的单步计算本身可能只需要 2-5ms（小 batch 场景）。这就是为什么**小 batch 场景下 TP 并行效率会显著降低**。

### 2.4 Prefill vs Decode 的通信占比

```
Prefill 阶段:
  - 计算量大（处理整个 prompt）
  - 通信量相对计算量小
  - 通信/计算比 低 → TP 效率高

Decode 阶段:
  - 计算量小（每次 1 token per request）
  - 通信延迟固定（α 不可忽略）
  - 通信/计算比 高 → TP 效率低
  - batch_size 越小，效率越低
```

## 3. vLLM 中的 TP 实现

### 3.1 基本配置

```bash
# 启动 4 路 TP
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4

# vLLM 会自动:
# 1. 将模型权重切分到 4 个 GPU
# 2. 在每个 GPU 上创建一个 worker
# 3. 使用 NCCL 作为默认通信后端
```

### 3.2 Column Parallel + Row Parallel

vLLM 遵循 Megatron-LM 的经典切分方式。核心实现在 `vllm/model_executor/layers/linear.py`：

**Column Parallel Linear**（按列切分权重，无通信）：

```python
# 权重 W 的形状: [hidden_size, out_features]
# 切分方式: 沿 out_features 维度切分
# Rank i 持有: W[:, i*(out_features//tp): (i+1)*(out_features//tp)]

class ColumnParallelLinear:
    """
    应用场景:
    - QKV projection: 按 head 维度切分
    - MLP gate_proj / up_proj: 按 intermediate_size 切分
    
    切分后:
    - 每个 rank 输入相同的 x
    - 每个 rank 输出不同的 y_i = x @ W_i
    - 无需通信
    """
    def forward(self, input_):
        # input_ 在所有 rank 上是相同的
        output = F.linear(input_, self.weight, self.bias)
        # output 在各 rank 上是不同的分片
        return output
```

**Row Parallel Linear**（按行切分权重，需要 AllReduce）：

```python
# 权重 W 的形状: [in_features, hidden_size]
# 切分方式: 沿 in_features 维度切分
# Rank i 持有: W[i*(in_features//tp): (i+1)*(in_features//tp), :]

class RowParallelLinear:
    """
    应用场景:
    - Attention output projection (o_proj)
    - MLP down_proj
    
    切分后:
    - 每个 rank 输入不同的 x_i（来自 ColumnParallel 的输出）
    - 每个 rank 计算 y_i = x_i @ W_i（局部结果）
    - 需要 AllReduce 聚合: y = sum(y_i)
    """
    def forward(self, input_):
        # input_ 在各 rank 上是不同的分片
        output = F.linear(input_, self.weight)
        # 需要 AllReduce 得到完整输出
        output = tensor_model_parallel_all_reduce(output)
        return output
```

**Column + Row 配合的妙处**：Column Parallel 的输出直接作为 Row Parallel 的输入，中间不需要通信。AllReduce 只发生在 Row Parallel 的输出处。

### 3.3 Attention Head 的切分

```python
# 以 Llama-3-70B 为例:
# num_heads = 64, num_kv_heads = 8
# TP = 8

# Attention head 切分:
#   每个 rank: 64/8 = 8 个 query head
#              8/8  = 1 个 kv head
#
# GQA 场景下，KV head 数量必须能被 TP size 整除
# 这限制了 TP size 的选择

# 如果 num_kv_heads < tp_size:
#   方案 1: 降低 TP size
#   方案 2: KV head 复制（某些 rank 共享 KV head）
```

### 3.4 权重加载与分发

```python
# vLLM 权重加载流程:
# 1. 每个 worker 独立加载属于自己 rank 的权重分片
# 2. 使用 weight_loader 函数做切分映射

# 关键函数: vllm/model_executor/model_loader/weight_utils.py
def _get_weight_shard(
    param: torch.Tensor,
    shard_id: int,         # 当前 rank
    num_shards: int,       # TP size
    shard_dim: int = 0,    # 切分维度
) -> torch.Tensor:
    """从完整权重中提取当前 rank 的分片"""
    shard_size = param.shape[shard_dim] // num_shards
    start = shard_id * shard_size
    end = start + shard_size
    return param.narrow(shard_dim, start, shard_size)
```

## 4. 通信后端：NCCL vs custom_all_reduce

### 4.1 NCCL

NCCL (NVIDIA Collective Communications Library) 是 vLLM 的默认通信后端：

```python
# vLLM 使用 PyTorch 的 distributed 包，底层调用 NCCL
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=tp_group)
```

NCCL 的特点：
- 通用性强，支持各种拓扑
- 自动选择最优的通信算法（Ring, Tree, etc.）
- 对大消息做了深度优化
- **缺点：小消息场景下 kernel launch overhead 较大**

### 4.2 vLLM Custom AllReduce

vLLM 实现了自定义 AllReduce，专门优化小消息场景（decode 阶段）。源码位于 `vllm/distributed/device_communicators/custom_all_reduce.py`：

```python
# Custom AllReduce 的核心思路:
# 1. 使用 CUDA IPC (Inter-Process Communication) 共享 GPU 显存
# 2. 通过 shared memory 进行数据交换，避免 NCCL 的 overhead
# 3. 实现 one-shot / two-shot AllReduce

class CustomAllreduce:
    """
    适用条件:
    - 同一节点内的 GPU（需要 P2P access）
    - 数据量较小（< 数 MB）
    - NVLink 连接
    
    不适用:
    - 跨节点通信
    - 大数据量通信（此时 NCCL 更优）
    """
    
    # One-shot AllReduce（数据量 < 阈值时使用）:
    # 1. 每个 rank 将数据写入共享 buffer
    # 2. barrier 同步
    # 3. 每个 rank 读取所有其他 rank 的数据并求和
    # 延迟: 1 次 barrier + 1 次读取

    # Two-shot AllReduce（数据量 > 阈值时使用）:
    # 类似 Reduce-Scatter + AllGather
    # 1. Reduce-Scatter: 每个 rank 计算一部分的 reduce
    # 2. AllGather: 广播 reduce 结果
    # 延迟: 2 次 barrier + 2 次读写
```

### 4.3 何时使用哪个后端

```python
# vLLM 的自动选择逻辑（简化版）:
def all_reduce(tensor, group):
    if can_use_custom_allreduce(tensor, group):
        # 条件: 同节点 + NVLink + 数据量小 + GPU 支持 P2P
        return custom_all_reduce(tensor, group)
    else:
        # fallback 到 NCCL
        return torch.distributed.all_reduce(tensor, group=group)
```

在实际部署中：
- **Decode 阶段**：数据量小，custom AllReduce 通常胜出（延迟降低 30-50%）
- **Prefill 阶段**：数据量大，NCCL 的带宽优化更重要

## 5. NVLink vs PCIe 对 TP 性能的影响

### 5.1 互联带宽对比

```
互联技术          单向带宽            双向带宽
───────────────────────────────────────────────
PCIe 4.0 x16      32 GB/s             64 GB/s
PCIe 5.0 x16      64 GB/s            128 GB/s
NVLink 3 (A100)   300 GB/s/link      600 GB/s (总 12 links)
NVLink 4 (H100)   450 GB/s/link      900 GB/s (总 18 links)
NVLink 5 (B200)   900 GB/s/link     1800 GB/s (总 18 links)
```

### 5.2 拓扑对 TP 效率的影响

**A100 8-GPU 节点（DGX A100）拓扑：**

```
GPU 0 ←──NVLink──→ GPU 1   (直连)
GPU 0 ←──NVLink──→ GPU 2   (直连)
GPU 0 ←──NVLink──→ GPU 3   (直连)
...
所有 8 GPU 通过 NVSwitch 全连接
NVSwitch 提供 any-to-any 全带宽通信
```

**非 NVSwitch 拓扑（如 4x A100 PCIe）：**

```
GPU 0 ←──PCIe──→ CPU ←──PCIe──→ GPU 1
                            ↕
GPU 2 ←──PCIe──→ CPU ←──PCIe──→ GPU 3

PCIe 带宽远低于 NVLink → TP 效率大幅下降
```

### 5.3 NVLink vs PCIe 性能实测经验

以 Llama-3-70B, TP=4 为例，典型的延迟差异：

```
场景                NVLink (A100)    PCIe 4.0
─────────────────────────────────────────────
Decode latency       3.2 ms/token     5.8 ms/token
Prefill (2K tokens)  45 ms            78 ms
TP 效率 (decode)     ~85%             ~55%
TP 效率 (prefill)    ~92%             ~70%
```

**核心结论**：PCIe 连接下 TP > 2 通常不划算。如果只有 PCIe 互联，优先考虑 DP 或 PP 替代高度 TP。

## 6. 调优建议

### 6.1 A100 80GB 配置建议

```
模型          推荐 TP    互联要求       备注
────────────────────────────────────────────────────────
7B-13B        1          无             单卡即可
34B           2          NVLink         
70B           4          NVSwitch       TP=4 需要 NVSwitch 保证全带宽
70B           8          NVSwitch       可释放更多 KV Cache 空间
405B          8 + PP     NVSwitch       单节点 8 卡装不下，需多节点
```

### 6.2 H100 80GB 配置建议

```
模型          推荐 TP    互联要求       备注
────────────────────────────────────────────────────────
7B-13B        1          无             H100 单卡性能强，TP 不必要
70B           4-8        NVSwitch       TP=4 延迟最优, TP=8 KV Cache 更大
405B          8          NVSwitch       一个节点刚好放下
MoE 600B+     8          NVSwitch       配合 EP 使用
```

### 6.3 跨节点 TP：为什么通常不推荐

```
跨节点通信延迟:
  InfiniBand HDR (200 Gbps): ~1-2 μs 延迟, 25 GB/s 带宽
  InfiniBand NDR (400 Gbps): ~1 μs 延迟, 50 GB/s 带宽

vs 节点内 NVLink:
  NVLink 4: ~0.5 μs 延迟, 450 GB/s 带宽

跨节点 TP 的问题:
  1. 带宽差 9-18x → prefill 阶段通信成为瓶颈
  2. 延迟差 2-4x → decode 阶段每层通信延迟增加
  3. 80 层模型 × 2 次/层 = 160 次跨节点通信/step
  4. 总额外延迟: 160 × ~2μs = 320μs（仅固定延迟部分）
```

**推荐策略**：节点内用 TP，节点间用 PP 或 EP。

### 6.4 TP 与 Continuous Batching 的配合

```python
# TP 场景下 continuous batching 的注意事项:

# 1. 所有 TP rank 必须处理相同的 batch
#    - scheduler 在 rank 0 上运行
#    - batch 信息通过 broadcast 同步到其他 rank

# 2. 所有 rank 的 KV Cache 管理必须同步
#    - block table 在所有 rank 上保持一致
#    - 每个 rank 存储自己负责的 head 的 KV Cache

# 3. Sampling 在 rank 0 上进行
#    - logits 在 AllGather 后只在 rank 0 上做 sampling
#    - 结果 broadcast 回其他 rank
```

## 7. 性能分析框架

### 7.1 Roofline 模型应用于 TP

```python
def tp_decode_time_estimate(
    model_params_B: float,    # 模型参数量 (billions)
    tp_size: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    dtype_bytes: int = 2,     # bf16
    mem_bw_GBs: float = 2000, # A100 HBM bandwidth
    link_bw_GBs: float = 300, # NVLink bandwidth per direction
    alpha_us: float = 5,      # AllReduce launch latency (μs)
):
    """估算 TP decode 单步延迟"""
    
    # 计算时间（memory-bound, decode 阶段）
    weight_bytes = model_params_B * 1e9 * dtype_bytes / tp_size
    compute_time_ms = weight_bytes / (mem_bw_GBs * 1e9) * 1000
    
    # 通信时间
    msg_size = batch_size * hidden_size * dtype_bytes  # 每次 AllReduce 的数据量
    comm_per_layer = 2 * (msg_size / (link_bw_GBs * 1e9) * 1000 + alpha_us / 1000)
    total_comm_ms = comm_per_layer * num_layers
    
    # 总延迟（通信与计算部分重叠，但 decode 阶段重叠有限）
    total_time_ms = compute_time_ms + total_comm_ms * 0.8  # 假设 20% 重叠
    
    return {
        "compute_ms": compute_time_ms,
        "comm_ms": total_comm_ms,
        "total_ms": total_time_ms,
        "comm_ratio": total_comm_ms / (compute_time_ms + total_comm_ms),
    }

# 示例: Llama-3-70B, TP=8, batch=32
result = tp_decode_time_estimate(
    model_params_B=70, tp_size=8, batch_size=32,
    hidden_size=8192, num_layers=80
)
# compute_ms ≈ 8.75 ms (70B * 2 / 8 / 2000 GB/s)
# comm_ms ≈ 0.8 ms
# total_ms ≈ 9.39 ms
# comm_ratio ≈ 8.4%
```

### 7.2 TP 效率度量

```
TP 效率 = T_single / (T_tp * tp_size)

其中:
  T_single = 单卡执行相同计算的时间（假设能放下）
  T_tp     = TP 后的实际执行时间

理想效率 = 100%（完美并行，零通信开销）
实际效率:
  TP=2 NVLink:  90-95%
  TP=4 NVLink:  85-92%
  TP=8 NVLink:  75-88%
  TP=2 PCIe:    70-80%
  TP=4 PCIe:    50-65%
```

## 8. 总结

| 要点 | 说明 |
|------|------|
| 推理 TP 更简单 | 无反向传播，通信减半 |
| 核心瓶颈 | Decode 阶段的 AllReduce 延迟（固定开销 α） |
| vLLM 实现 | Column Parallel + Row Parallel，2 次 AllReduce/层 |
| 通信后端 | NCCL（大消息） + custom AllReduce（小消息） |
| 互联选择 | NVLink/NVSwitch 是 TP 的基础，PCIe 下 TP > 2 不推荐 |
| 跨节点 TP | 通常不推荐，改用 PP/EP |

---

> **下一节**：[Pipeline Parallel 推理](02-pipeline-parallel.md) —— 当模型太大无法放入单节点时的分层策略
