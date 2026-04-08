# Pipeline Parallel 推理

> 本节分析推理场景下 Pipeline Parallelism (PP) 的设计、bubble overhead、延迟影响，以及与 TP 的组合策略。

## 1. 推理 PP 与训练 PP 的根本区别

### 1.1 训练时的 PP

训练时 PP 将模型按层分组，每组放在一个 stage（GPU）上。核心挑战是**pipeline bubble**——由于前向和反向传播的依赖关系，GPU 不可避免地有空闲时间。

```
训练 PP (GPipe 风格, 4 stages, 4 micro-batches):

时间 →
Stage 0: [F0][F1][F2][F3][  ][  ][  ][B3][B2][B1][B0]
Stage 1: [  ][F0][F1][F2][F3][  ][B3][B2][B1][B0][  ]
Stage 2: [  ][  ][F0][F1][F2][F3][B3][B2][B1][B0][  ][  ]
Stage 3: [  ][  ][  ][F0][F1][F2][F3][B3][B2][B1][B0][  ][  ][  ]

F = Forward, B = Backward
[  ] = Bubble (空闲)

Bubble 比例 = (pp_size - 1) / (num_microbatches + pp_size - 1)
```

### 1.2 推理时的 PP

推理只有前向传播，PP 的行为完全不同：

```
推理 PP (4 stages, continuous batching):

时间 →
Stage 0: [F_batch0][F_batch1][F_batch2][F_batch3][F_batch4]...
Stage 1:    [F_batch0][F_batch1][F_batch2][F_batch3][F_batch4]...
Stage 2:       [F_batch0][F_batch1][F_batch2][F_batch3][F_batch4]...
Stage 3:          [F_batch0][F_batch1][F_batch2][F_batch3][F_batch4]...

特点:
1. 没有反向传播 → 无 F/B 依赖造成的 bubble
2. Continuous batching → pipeline 可以持续流动
3. 主要开销: stage 间的通信延迟（inter-stage latency）
```

### 1.3 差异对比

| 维度 | 训练 PP | 推理 PP |
|------|---------|---------|
| Pipeline 方向 | 前向 + 反向（双向） | 仅前向（单向） |
| Bubble 来源 | F/B 依赖 | 启动延迟（warmup） |
| Micro-batching | 必须切分以减少 bubble | 由 continuous batching 天然提供 |
| 通信内容 | 激活值 + 梯度 | 仅激活值（hidden states） |
| 显存占用 | 需缓存激活值用于反向 | 无需缓存激活值 |
| 稳态行为 | 周期性 bubble | 持续流水 |

## 2. 推理 PP 的 Pipeline Bubble

虽然推理 PP 没有训练那样严重的 bubble，但仍有以下空闲时间。

### 2.1 Warmup Bubble

```
启动阶段 (warmup):

第 1 步: 只有 Stage 0 在计算
第 2 步: Stage 0 + Stage 1 在计算
第 3 步: Stage 0 + Stage 1 + Stage 2 在计算
第 pp_size 步: 所有 stage 满载

Warmup 时间 = (pp_size - 1) × per_stage_latency
```

这个 warmup bubble 只发生一次（服务启动时或重新开始 serving 时），对于持续运行的推理服务影响很小。

### 2.2 Drain Bubble

当需要清空 pipeline 时（例如所有请求都完成了），最后一个 batch 需要等待流过所有 stage：

```
Drain 阶段:

Stage 0: [...][F_last][  ][  ][  ]
Stage 1: [...][  ][F_last][  ][  ]
Stage 2: [...][  ][  ][F_last][  ]
Stage 3: [...][  ][  ][  ][F_last]

Drain 时间 = (pp_size - 1) × per_stage_latency
```

在高负载下 drain 很少发生，因为 continuous batching 持续有新请求进入。

### 2.3 Decode 阶段的真正 Bubble 问题

推理 PP 最关键的 bubble 问题出现在 **decode 阶段的逐 token 生成**：

```
Decode 一个 token 的流程 (PP=4):

时间 →
Stage 0: [计算]──────────────→ send hidden
Stage 1:        [等待] [计算]──────────────→ send hidden
Stage 2:               [等待] [计算]──────────────→ send hidden
Stage 3:                      [等待] [计算] → 输出 token

单 token 延迟 = pp_size × per_stage_compute + (pp_size - 1) × inter_stage_comm
```

**关键问题**：decode 阶段每步只生成 1 个 token（对于每个请求），pipeline 的各 stage 串行执行。如果有多个 batch 在同时 decode，可以通过 **interleaved scheduling** 隐藏部分延迟。

### 2.4 Interleaved Micro-batch Scheduling

```
没有 interleave (naive):

Stage 0: [Batch0]     [Batch1]     [Batch2]
Stage 1:    [Batch0]     [Batch1]     [Batch2]
Stage 2:       [Batch0]     [Batch1]     [Batch2]
Stage 3:          [Batch0]     [Batch1]     [Batch2]

有 interleave (1F1B 风格):

Stage 0: [B0][B1][B2][B3][B4][B5]...
Stage 1:   [B0][B1][B2][B3][B4][B5]...
Stage 2:     [B0][B1][B2][B3][B4][B5]...
Stage 3:       [B0][B1][B2][B3][B4][B5]...

Interleave 使得每个 stage 在等待下游返回时可以处理其他 batch
但 per-request 延迟不变（甚至略增），只是吞吐提升
```

## 3. PP 的延迟 Overhead 分析

### 3.1 Inter-stage 通信开销

PP 的 stage 间通信内容是**激活值**（hidden states），而非 TP 中的 AllReduce 结果。

```python
# Inter-stage 通信量:
# 每步传输的数据 = batch_tokens × hidden_size × dtype_bytes
#
# 以 Llama-3-70B (hidden_size=8192, bf16) 为例:
#   Decode (batch=64):  64 × 8192 × 2 = 1 MB
#   Prefill (2K tokens): 2048 × 8192 × 2 = 32 MB

# 通信延迟 (InfiniBand NDR, 400 Gbps = 50 GB/s):
#   Decode:  1MB / 50 GB/s ≈ 0.02 ms + α ≈ 0.03 ms
#   Prefill: 32MB / 50 GB/s ≈ 0.64 ms + α ≈ 0.65 ms

# 通信延迟 (NVLink 4, 450 GB/s):
#   Decode:  1MB / 450 GB/s ≈ 0.002 ms + α ≈ 0.007 ms
#   Prefill: 32MB / 450 GB/s ≈ 0.07 ms + α ≈ 0.08 ms
```

### 3.2 PP 对 TTFT 和 TPOT 的影响

```
TTFT (Time to First Token):
  无 PP: T_prefill
  有 PP: T_prefill + (pp_size - 1) × inter_stage_comm
  
  对于跨节点 PP (IB), pp_size=2:
    额外延迟 ≈ 0.65 ms（可接受）

TPOT (Time Per Output Token):
  无 PP: T_decode_step
  有 PP: T_decode_step + (pp_size - 1) × inter_stage_comm
  
  但如果使用 interleaved scheduling:
    TPOT 不变（通信被隐藏），吞吐提升
    per-request 延迟: 增加 (pp_size - 1) × inter_stage_comm
```

### 3.3 延迟 vs 吞吐的权衡

```
PP 延迟分析:

                  PP=1    PP=2     PP=4
─────────────────────────────────────────
每 stage 层数      80      40       20
(Llama-3-70B)
计算时间/step     10ms    5ms      2.5ms
通信开销          0       0.03ms   0.09ms
端到端延迟        10ms    5.03ms   2.59ms
Pipeline 延迟     0       5.03ms   7.77ms
                          (×1)     (×3)

结论:
- PP 降低了单 stage 计算时间
- 但串行的 pipeline 延迟会累积
- 实际 end-to-end 延迟取决于 interleave 效果
```

## 4. 何时使用 PP

### 4.1 模型太大，单节点放不下

```
场景: Llama-3-405B, bf16 ≈ 810 GB 权重

方案 1: TP=8 (单节点, 8×H100 80GB = 640GB)
  → 640GB < 810GB, 放不下!

方案 2: TP=8, PP=2 (双节点)
  → 每个节点分到 405GB / 2 ≈ 405GB 权重
  → 每张卡: 405GB / 8 ≈ 50.6GB → 可以放下
  → 剩余 ~30GB/卡 用于 KV Cache

方案 3: TP=8, PP=4 (四节点)
  → 每张卡: ~25.3GB → 更多显存给 KV Cache
```

### 4.2 PP vs TP 的选择原则

```
优先用 TP（节点内）的理由:
  1. NVLink 带宽高，TP 通信开销小
  2. TP 无 pipeline 延迟
  3. 所有 GPU 同时计算同一个 token → 延迟最低

优先用 PP（跨节点）的理由:
  1. 跨节点带宽低，TP AllReduce 开销大（每层 2 次）
  2. PP 跨节点只需 stage 边界处的 P2P 通信（比 TP 少得多）
  3. PP 通信量 = 1 × hidden_states/step
     TP 通信量 = 2L × AllReduce/step
```

### 4.3 不推荐使用 PP 的场景

```
1. 模型可以放入单节点
   → 用 TP 更好（延迟更低）

2. 请求量极少，pipeline 无法饱和
   → PP 的 GPU 利用率低
   → 不如用单节点 TP + 更大 batch

3. 极端延迟敏感场景
   → PP 的 pipeline latency 是不可消除的
   → 即使用 interleave 也只能隐藏吞吐，不能减少单请求延迟
```

## 5. vLLM PP 实现

### 5.1 配置

```bash
# PP=2, TP=4 (2 节点，每节点 4 卡)
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2

# vLLM 会自动:
# 1. 将 80 层模型分为 2 个 stage，每个 40 层
# 2. Stage 0: 节点 0 的 4 张卡 (TP=4)
# 3. Stage 1: 节点 1 的 4 张卡 (TP=4)
# 4. 使用 NCCL P2P 进行 stage 间通信
```

### 5.2 层分配策略

```python
# vLLM 的层分配逻辑 (简化)
# 位置: vllm/distributed/parallel_state.py

def get_pp_layer_range(
    num_layers: int,       # 总层数
    pp_rank: int,          # 当前 PP rank
    pp_size: int,          # PP size
) -> tuple[int, int]:
    """计算当前 PP rank 负责的层范围"""
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size
    
    # 尽量均匀分配，余数给前面的 stage
    if pp_rank < remainder:
        start = pp_rank * (layers_per_stage + 1)
        end = start + layers_per_stage + 1
    else:
        start = pp_rank * layers_per_stage + remainder
        end = start + layers_per_stage
    
    return start, end

# 示例: 80 层, PP=3
# Stage 0: 层 0-26  (27 层)
# Stage 1: 层 27-53 (27 层)
# Stage 2: 层 54-79 (26 层)
```

### 5.3 Stage 间通信

```python
# PP stage 间的数据传输
# 位置: vllm/distributed/communication_op.py

def send_to_next_pp_stage(tensor: torch.Tensor):
    """将激活值发送到下一个 PP stage"""
    next_rank = get_next_pp_rank()
    torch.distributed.send(tensor, dst=next_rank, group=pp_group)

def recv_from_prev_pp_stage(shape, dtype) -> torch.Tensor:
    """从前一个 PP stage 接收激活值"""
    tensor = torch.empty(shape, dtype=dtype, device="cuda")
    prev_rank = get_prev_pp_rank()
    torch.distributed.recv(tensor, src=prev_rank, group=pp_group)
    return tensor
```

### 5.4 PP + Continuous Batching 的调度

```python
# PP 与 continuous batching 的交互:

# 1. Scheduler 在 rank 0 (Stage 0 的 driver) 上运行
# 2. Scheduler 决定哪些请求在当前步执行
# 3. 调度信息 broadcast 到所有 rank

# PP 调度的挑战:
# - 不同 stage 可能在处理不同的 micro-batch
# - Stage 0 发出 batch A 时，Stage 1 还在处理 batch (A-1)
# - 需要保证所有 stage 的 KV Cache 管理一致

# vLLM 的做法:
# - 同步模式: 所有 stage 处理相同的 batch（简单但有 bubble）
# - 异步模式: 不同 stage 可以处理不同 batch（复杂但吞吐高）
```

## 6. PP 的显存分析

### 6.1 权重显存

```
PP 的显存优势: 每个 stage 只加载一部分层的权重

以 Llama-3-70B (bf16) 为例:
  全量权重: ~140 GB

  PP=1, TP=8: 每卡 140/8 = 17.5 GB 权重
  PP=2, TP=4: 每卡 140/(2×4) = 17.5 GB 权重  (相同)
  PP=4, TP=2: 每卡 140/(4×2) = 17.5 GB 权重  (相同)

PP 和 TP 在权重显存上的效果相同: weight / (pp_size × tp_size)
```

### 6.2 KV Cache 显存

```
KV Cache 的分布:
  TP: 每个 rank 存储 num_kv_heads/tp_size 个 head 的 KV Cache
  PP: 每个 stage 只存储自己负责的层的 KV Cache

以 Llama-3-70B, max_seq_len=8192 为例:
  每层 KV Cache = 2 × num_kv_heads × head_dim × seq_len × dtype
               = 2 × 8 × 128 × 8192 × 2 = 32 MB/seq

  PP=1: 每卡存 80 层 KV → 80 × 32MB/8 = 320 MB/seq
  PP=2: 每卡存 40 层 KV → 40 × 32MB/4 = 320 MB/seq
  PP=4: 每卡存 20 层 KV → 20 × 32MB/2 = 320 MB/seq

每卡 KV Cache 大小 = num_layers/pp_size × per_layer_kv / tp_size
```

### 6.3 额外的激活值显存

```
PP 的额外显存开销:
  - Stage 间传输的 buffer: batch_tokens × hidden_size × dtype
  - 通常很小 (< 100 MB)
  - 不是瓶颈
```

## 7. PP 的实际部署案例

### 7.1 案例: Llama-3-405B 部署

```
硬件: 4 × DGX H100 (每节点 8 × H100 80GB)
网络: 节点间 InfiniBand NDR 400 Gbps

方案 A: TP=8, PP=4 (使用所有 32 卡)
  - 每卡权重: 810GB / 32 ≈ 25.3 GB
  - 每卡剩余: ~55 GB 给 KV Cache
  - 跨节点通信: PP P2P (每步 1 次, 数据量小)
  - 优势: KV Cache 空间大，支持长序列和大 batch

方案 B: TP=8, PP=2 (使用 16 卡)
  - 每卡权重: 810GB / 16 ≈ 50.6 GB
  - 每卡剩余: ~29 GB 给 KV Cache
  - 优势: 更少节点，pipeline 延迟更小
  - 劣势: KV Cache 较小，最大 batch 受限

选择依据:
  - 延迟优先 → 方案 B (PP=2, 更短 pipeline)
  - 吞吐优先 → 方案 A (PP=4, 更大 KV Cache → 更大 batch)
```

### 7.2 案例: 延迟敏感型服务

```
场景: 实时对话应用，要求 TPOT < 30ms
模型: 70B, bf16

硬件选择:
  方案 1: 1 × DGX H100, TP=8, PP=1
    decode latency ≈ 8-12 ms → 满足要求
    
  方案 2: 2 × 4-GPU 节点, TP=4, PP=2
    decode latency ≈ 10-15 ms + pipeline overhead
    pipeline overhead ≈ 2-5 ms
    total ≈ 12-20 ms → 勉强满足

结论: 延迟敏感场景尽量避免 PP，用 TP 替代
```

## 8. PP 优化技巧

### 8.1 Layer 分配不均匀优化

```python
# 问题: 第一个和最后一个 stage 有额外计算
# Stage 0: embedding layer + 前 N 层
# Stage last: 后 M 层 + lm_head + sampling

# 优化: 给中间 stage 分配更多层
# 例如 80 层, PP=4:
# Stage 0: embedding + 层 0-17  (18 层)  → 计算包含 embedding
# Stage 1: 层 18-38              (21 层)
# Stage 2: 层 39-59              (21 层)
# Stage 3: 层 60-79 + lm_head   (20 层)  → 计算包含 lm_head
```

### 8.2 通信与计算重叠

```python
# 异步 P2P 通信:
# 在发送当前 batch 结果的同时，开始处理下一个 batch

# Stage i 的执行流程:
async def stage_loop():
    while True:
        # 接收上一个 stage 的输出
        hidden = recv_from_prev()
        
        # 异步发送上一个 batch 的结果（如果有的话）
        if prev_result is not None:
            send_future = async_send_to_next(prev_result)
        
        # 计算当前 batch
        result = compute_layers(hidden)
        
        # 等待发送完成
        if send_future:
            send_future.wait()
        
        prev_result = result
```

## 9. 总结

| 要点 | 说明 |
|------|------|
| 推理 PP 更轻量 | 无反向传播，无需缓存中间激活值 |
| Bubble 来源 | 主要是 warmup/drain 和 decode 串行化 |
| 通信开销 | 小于 TP（只有 stage 边界的 P2P，而非每层 AllReduce） |
| 适用场景 | 模型跨节点部署时作为 TP 的补充 |
| 延迟影响 | 增加 (pp_size-1) × inter_stage_comm 的 pipeline 延迟 |
| 不推荐场景 | 模型能放单节点、延迟极度敏感 |

**PP 使用决策树：**

```
模型能放入单节点？
  ├── 是 → 不用 PP，用 TP
  └── 否 → 用 PP + TP
        ├── 延迟优先 → PP 尽量小 (PP=2)
        └── 吞吐优先 → PP 可以大些（更多 KV Cache 空间）
```

---

> **下一节**：[Expert Parallel (MoE) 推理](03-expert-parallel.md) —— MoE 架构下的 Expert 分布与通信策略
