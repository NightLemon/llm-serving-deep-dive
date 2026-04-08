# Expert Parallel (EP) — MoE 推理

> 本节深入分析 Mixture-of-Experts (MoE) 模型在推理场景下的 Expert Parallel 策略，涵盖 expert 放置、All-to-All 通信、弹性 EP、负载均衡，以及 vLLM 源码实现。

## 1. MoE 推理的独特挑战

### 1.1 MoE 架构回顾

MoE (Mixture-of-Experts) 模型用稀疏激活实现大参数量的同时保持较低的计算成本。

```
标准 Transformer 层:
  Input → Attention → MLP → Output
  每个 token 经过完整的 MLP (dense)

MoE Transformer 层:
  Input → Attention → Router → Expert_i (MLP) → Output
  每个 token 经过 Router 选择 top-K 个 expert
  只有被选中的 expert 参与计算

典型参数:
  DeepSeek-V3:  256 个 routed experts + 1 shared expert
                每 token 激活 8 个 experts
                总参数量 671B, 激活参数量 ~37B
  
  Mixtral 8x7B: 8 个 experts
                每 token 激活 2 个 experts
                总参数量 47B, 激活参数量 ~13B
```

### 1.2 MoE 推理 vs Dense 推理

```
显存挑战:
  Dense 70B (bf16): ~140 GB 权重
  MoE 671B (bf16):  ~1342 GB 权重（但计算量只相当于 ~37B）

计算特征:
  Dense: 每个 token 使用所有参数 → 计算密度高
  MoE:   每个 token 只用部分参数 → 计算密度低, memory-bound 更严重

通信模式:
  Dense + TP: AllReduce (每个 token 到相同的 GPU)
  MoE + EP:   All-to-All (不同 token 路由到不同 expert 所在的 GPU)
```

### 1.3 核心矛盾

```
MoE 推理的核心矛盾:

1. 参数量太大 → 需要多 GPU 存放 expert 权重
2. 每 token 激活量小 → 每个 expert 的计算量很小
3. Router 动态决定路由 → 通信模式不规则（All-to-All）
4. Expert 负载不均 → 部分 GPU 成为热点

结果: MoE 推理很容易变成通信 bound 而非计算 bound
```

## 2. Expert 放置策略

### 2.1 基本放置方案

```
方案 1: Expert Replication (复制)
  每个 GPU 持有所有 expert 的完整副本
  优点: 无需 All-to-All 通信
  缺点: 显存浪费严重, 大模型不可行
  适用: 小型 MoE (如 Mixtral 8x7B 在 8×H100 上)

方案 2: Expert Partition (分区, 即 EP)
  每个 GPU 持有一部分 expert
  优点: 显存高效
  缺点: 需要 All-to-All 通信
  适用: 大型 MoE (如 DeepSeek-V3)

方案 3: Hybrid (混合)
  部分热门 expert 复制，其余 expert 分区
  优点: 平衡通信和显存
  缺点: 实现复杂
```

### 2.2 Expert Partition 细节

```
以 DeepSeek-V3 (256 experts, EP=32) 为例:

GPU 0:  Expert 0-7    (8 experts)
GPU 1:  Expert 8-15   (8 experts)
GPU 2:  Expert 16-23  (8 experts)
...
GPU 31: Expert 248-255 (8 experts)

每个 GPU 的 expert 权重:
  每个 expert: ~5.2 GB (bf16)
  每卡: 8 × 5.2 GB ≈ 41.6 GB expert 权重
  加上共享层 (attention, shared expert): ~10 GB
  总计: ~52 GB / 卡

剩余 ~28 GB 用于 KV Cache (H100 80GB)
```

### 2.3 EP 与 TP 的组合

```
TP + EP 组合 (DeepSeek-V3 典型配置):

节点内 (8 GPU): TP=8 (切分 attention 和 shared expert)
节点间 (N 节点): EP=N (分配 routed experts)

每个 GPU 看到的计算:
  Attention: 处理所有 token, 只有自己负责的 head
  Shared Expert: 处理所有 token, 切分 hidden 维度
  Routed Expert: 只处理路由到自己的 token, 完整的 expert 权重

通信模式:
  TP: AllReduce (节点内, NVLink)
  EP: All-to-All (节点间, InfiniBand)
```

## 3. All-to-All 通信

### 3.1 All-to-All 基本概念

```
All-to-All 操作:
  每个 rank 向每个其他 rank 发送不同的数据
  不同于 AllReduce (所有 rank 发送相同维度的数据)

EP 中的 All-to-All:
  Step 1 (Dispatch): 每个 GPU 将 token 发送到对应 expert 所在的 GPU
  Step 2 (Compute):  各 GPU 用本地 expert 计算收到的 token
  Step 3 (Combine):  各 GPU 将计算结果发回 token 原始所在的 GPU
```

### 3.2 All-to-All 通信模式图

```
All-to-All Dispatch (4 GPUs, 4 experts/GPU):

Token 路由结果:
  GPU 0 的 token: [E2, E5, E1, E12]  → 需发送到 GPU 0,1,0,3
  GPU 1 的 token: [E0, E3, E7, E9]   → 需发送到 GPU 0,0,1,2
  GPU 2 的 token: [E4, E11, E2, E6]  → 需发送到 GPU 1,2,0,1
  GPU 3 的 token: [E8, E1, E14, E3]  → 需发送到 GPU 2,0,3,0

通信矩阵 (发送量):
          To GPU0  To GPU1  To GPU2  To GPU3
GPU 0:      2        1        0        1
GPU 1:      2        1        1        0
GPU 2:      1        2        1        0
GPU 3:      1        0        1        1

特点: 通信矩阵每次都不同, 由 Router 动态决定
```

### 3.3 All-to-All 通信开销

```python
# All-to-All 通信量分析:

# 假设:
#   N = EP size (参与的 GPU 数)
#   T = 每个 GPU 上的 token 数
#   K = top-K experts per token
#   H = expert hidden size
#   dtype = bf16 (2 bytes)

# Dispatch 阶段:
#   每个 GPU 发送 T × K 个 token 的 hidden states
#   总发送量 ≈ T × K × H × 2 bytes (per GPU)
#
#   理想均匀分布下, 每对 GPU 间:
#   T × K / N × H × 2 bytes

# Combine 阶段 (相同量):
#   每个 GPU 接收计算结果并发回
#   总量与 Dispatch 相同

# 以 DeepSeek-V3 (EP=32, K=8, H=2048, batch_tokens=4096) 为例:
#   Dispatch per GPU: 4096 × 8 × 2048 × 2 ≈ 128 MB
#   均分到 32 路: 每路 ~4 MB
#   IB NDR (50 GB/s): 128 MB / 50 GB/s ≈ 2.56 ms
#   实际(考虑网络拥塞): 3-5 ms
```

### 3.4 All-to-All 优化

```
1. Token Dropping (丢弃过载 token):
   当某个 expert 收到的 token 超过容量上限时, 丢弃多余的 token
   减少通信不均衡, 但影响输出质量

2. Expert Capacity Factor:
   capacity = ceil(tokens_per_gpu × K / N × capacity_factor)
   capacity_factor = 1.0 → 恰好均匀
   capacity_factor = 1.25 → 允许 25% 不均衡

3. Overlapping Communication and Computation:
   将 All-to-All 拆分为多个小消息
   在发送/接收的同时计算已到达的 token
   
4. Hierarchical All-to-All:
   节点内先做 local All-to-All (NVLink)
   然后节点间做 global All-to-All (IB)
   减少跨节点通信量
```

## 4. Static EP vs Elastic EP vs EPLB

### 4.1 Static EP

```
Static EP: 固定分配 expert 到 GPU

优点:
  - 实现简单
  - 通信模式可预测
  
缺点:
  - Expert 负载不均 → 某些 GPU 过载
  - 无法适应动态负载变化
  - 热门 expert 成为瓶颈

示例负载不均问题:
  假设 Expert 0 被 30% 的 token 选中
  而 Expert 100 只被 0.1% 的 token 选中
  → 持有 Expert 0 的 GPU 计算量是 Expert 100 的 300 倍
  → 所有 GPU 必须等待最慢的那个 GPU
```

### 4.2 Elastic EP (弹性 EP)

vLLM 实现了 Elastic EP，允许动态调整 expert 在 GPU 间的分配。

```
Elastic EP 核心思想:

1. 监控每个 expert 的实际负载
2. 热门 expert 复制到多个 GPU (replicate)
3. 冷门 expert 合并到更少的 GPU (consolidate)
4. 周期性重新平衡

示例:
  初始: GPU 0 = [E0, E1], GPU 1 = [E2, E3]
  
  观测: E0 负载 = 40%, E1 = 10%, E2 = 35%, E3 = 15%
  
  调整后: GPU 0 = [E0, E3], GPU 1 = [E0_replica, E2]
  (E0 复制到 GPU 1, E1 和 E3 合并到 GPU 0)
```

vLLM 中的 Elastic EP 实现位于 `vllm/distributed/elastic_ep/`:

```python
# Elastic EP 核心组件:
# 
# 1. ExpertLoadMonitor: 追踪每个 expert 的 token 处理量
# 2. ExpertPlacementPolicy: 决策 expert 重新放置策略
# 3. ExpertMigrator: 执行 expert 权重的迁移
#
# 工作流程:
# ┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
# │ LoadMonitor  │────→│ PlacementPolicy  │────→│  Migrator    │
# │ (统计负载)    │     │ (决策新分配)      │     │ (迁移权重)    │
# └──────────────┘     └──────────────────┘     └──────────────┘
#       ↑                                              │
#       └──────────────────────────────────────────────┘
#                      (周期性循环)
```

### 4.3 EPLB (Expert Parallel Load Balancing)

EPLB 是 vLLM 中更高级的负载均衡策略，位于 `vllm/distributed/eplb/`。

```
EPLB 的核心方法:

1. 基于历史统计的负载预测:
   - 收集过去 N 个 step 的 expert 选择分布
   - 预测未来的负载模式
   - 据此调整 expert 放置

2. Redundant Expert 策略:
   - 为高负载 expert 创建冗余副本
   - 冗余副本可以分散到不同 GPU 上
   - Router 将 token 同时路由到原始和冗余副本

3. Expert Rebalancing:
   - 定期检查负载均衡度
   - 如果不均衡度超过阈值, 触发 rebalance
   - Rebalance 过程中需要迁移权重 + 更新路由表
```

```python
# EPLB 的关键数据结构 (简化):

class ExpertLoadBalancer:
    """Expert Parallel Load Balancer"""
    
    def __init__(self, num_experts, ep_size):
        self.num_experts = num_experts
        self.ep_size = ep_size
        # expert_load[i] = 最近 N 步中 expert i 处理的 token 数
        self.expert_load = torch.zeros(num_experts)
        # expert_placement[i] = expert i 所在的 GPU rank
        self.expert_placement = self._init_placement()
    
    def update_load(self, expert_indices: torch.Tensor):
        """更新 expert 负载统计"""
        for idx in expert_indices.unique():
            count = (expert_indices == idx).sum()
            self.expert_load[idx] += count
    
    def should_rebalance(self) -> bool:
        """判断是否需要重新平衡"""
        loads_per_gpu = self._get_per_gpu_load()
        imbalance = loads_per_gpu.max() / loads_per_gpu.mean()
        return imbalance > self.threshold  # 例如 threshold = 1.3
    
    def compute_new_placement(self) -> dict:
        """计算新的 expert 放置方案"""
        # 贪心算法: 将高负载 expert 分散到低负载 GPU
        # 可能涉及 expert 复制 (redundancy)
        ...
```

### 4.4 三种方案对比

```
                Static EP    Elastic EP    EPLB
──────────────────────────────────────────────────
实现复杂度        低           中            高
负载均衡          差           好            最好
运行时开销        无           中 (监控)      高 (监控+迁移)
显存效率          高           中 (复制)      中 (冗余)
适用场景          均匀路由      动态路由       高度不均衡路由
vLLM 支持        是           是            是
```

## 5. vLLM MoE 源码走读

### 5.1 Fused MoE Layer

核心实现在 `vllm/model_executor/layers/fused_moe/`:

```python
# Fused MoE 的核心流程:

class FusedMoE:
    """
    融合的 MoE 层实现
    将 routing + dispatch + expert compute + combine 融合为高效 kernel
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,   # [num_tokens, hidden_size]
        router_logits: torch.Tensor,   # [num_tokens, num_experts]
    ) -> torch.Tensor:
        
        # Step 1: Routing
        # 计算每个 token 选择哪些 expert
        routing_weights, selected_experts = self.router(router_logits)
        # routing_weights: [num_tokens, top_k]  (softmax 权重)
        # selected_experts: [num_tokens, top_k] (expert 编号)
        
        # Step 2: Token dispatch
        # 将 token 按 expert 分组, 准备发送到对应 GPU
        dispatched = self.dispatch(hidden_states, selected_experts)
        
        # Step 3: All-to-All (如果使用 EP)
        if self.ep_size > 1:
            received = all_to_all(dispatched, ep_group)
        else:
            received = dispatched
        
        # Step 4: Expert computation
        # 每个 GPU 用本地 expert 计算收到的 token
        expert_output = self.expert_compute(received)
        
        # Step 5: All-to-All (结果返回)
        if self.ep_size > 1:
            combined = all_to_all(expert_output, ep_group)
        else:
            combined = expert_output
        
        # Step 6: Combine
        # 加权聚合各 expert 的输出
        output = self.combine(combined, routing_weights)
        
        return output
```

### 5.2 Fused MoE Kernel

```python
# vLLM 使用 Triton kernel 实现高效的 MoE 计算
# 位置: vllm/model_executor/layers/fused_moe/fused_moe.py

# 关键优化:
# 1. Token 重排: 将同一 expert 的 token 连续排列
#    避免 gather/scatter 的随机访问开销
#
# 2. Fused Gate+Up+Down: 
#    将 gate_proj, up_proj, down_proj 融合为一个 kernel
#    减少显存读写次数
#
# 3. Group GEMM:
#    将多个小 expert GEMM 合并为一个 group GEMM
#    提高 GPU 利用率

@triton.jit
def fused_moe_kernel(
    # token hidden states
    hidden_states_ptr,
    # expert weights
    gate_up_weights_ptr,
    down_weights_ptr,
    # routing info
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_per_expert_ptr,
    # output
    output_ptr,
    ...
):
    """
    Triton kernel for fused MoE computation
    每个 block 处理一个 (expert, token_group) 对
    """
    ...
```

### 5.3 EP 通信实现

```python
# EP All-to-All 通信的实现
# 位置: vllm/distributed/parallel_state.py

def expert_parallel_all_to_all(
    input_tensor: torch.Tensor,
    input_splits: List[int],     # 每个 rank 发送的 token 数
    output_splits: List[int],    # 每个 rank 接收的 token 数
    ep_group: ProcessGroup,
) -> torch.Tensor:
    """
    EP All-to-All 通信
    
    与标准 All-to-All 的区别:
    - 发送/接收大小不对称 (由 routing 决定)
    - 需要先交换 split 信息
    """
    # 1. 交换 split 信息 (All-to-All metadata)
    output_splits = all_to_all_single(
        torch.tensor(input_splits), group=ep_group
    )
    
    # 2. 实际数据的 All-to-All
    output = torch.empty(sum(output_splits), *input_tensor.shape[1:])
    torch.distributed.all_to_all_single(
        output, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=ep_group,
    )
    
    return output
```

## 6. DeepSeek-V3 的 EP 实践

### 6.1 架构概述

```
DeepSeek-V3 MoE 配置:
  - 61 层 MoE (每层有 MoE block)
  - 256 routed experts + 1 shared expert per layer
  - Top-8 routing (每 token 激活 8 个 expert)
  - Expert hidden size: 2048
  - Total params: 671B, Active params: ~37B
  
DeepSeek-V3 的 MLA (Multi-head Latent Attention):
  - 压缩 KV Cache, 减少显存占用
  - latent dimension = 512 (远小于标准 KV head dim)
```

### 6.2 推理部署配置

```
DeepSeek-V3 典型推理部署:

方案 1: FP8 量化 + EP
  硬件: 4 × 8-GPU H100 节点 (32 GPUs)
  TP=8 (节点内), EP=4 (跨节点, 每节点一组 experts)
  每节点存放 256/4 = 64 experts
  每卡 expert 权重: ~64 × 2.6GB(FP8) ≈ 21GB
  
方案 2: BF16 + EP
  硬件: 8 × 8-GPU H100 节点 (64 GPUs)
  TP=8 (节点内), EP=8 (跨节点)
  每节点存放 256/8 = 32 experts
  每卡 expert 权重: ~32 × 5.2GB(BF16) ≈ 21GB
```

### 6.3 DeepSeek-V3 的辅助损失和负载均衡

```
DeepSeek-V3 训练时使用 auxiliary-loss-free 负载均衡:
  - 不使用传统的 load balancing loss
  - 使用 bias term 动态调整 expert 选择概率
  - 推理时这些 bias 是固定的

推理时的挑战:
  - 训练好的 routing 不一定在所有 prompt 分布上均衡
  - 不同领域的文本 → 不同的 expert 热点
  - 代码 → 偏向某些 expert
  - 数学 → 偏向另一些 expert
  
应对策略:
  - Redundant expert: 热门 expert 复制到多个 GPU
  - Dynamic routing adjustment: 运行时微调 routing bias
  - Expert prefetch: 预取可能需要的 expert 权重
```

### 6.4 性能优化

```
DeepSeek-V3 推理优化技术:

1. All-to-All 通信优化:
   - 使用 NVLink 做节点内 All-to-All
   - 使用 InfiniBand RDMA 做节点间 All-to-All
   - Hierarchical All-to-All: 先本地聚合, 再跨节点通信
   
2. Computation-Communication Overlap:
   - Shared expert 计算与 All-to-All 通信重叠
   - Shared expert 处理所有 token, 计算时间稳定
   - 在 shared expert 计算期间, 完成 routed expert 的 All-to-All

   Timeline:
   ┌──────────────┬──────────────┐
   │ Shared Expert│ Routed Expert│ ← Compute
   │ Compute      │ Compute      │
   ├──────────────┤              │
   │ All-to-All   │              │ ← Communication
   │ (dispatch)   │              │   (与 shared expert 重叠)
   └──────────────┴──────────────┘

3. Expert Grouping:
   - 将常常被同时选中的 expert 放在同一个 GPU
   - 减少 All-to-All 通信量
   - 通过分析训练数据的 routing 统计来决定分组
```

## 7. EP 的性能分析

### 7.1 通信/计算比

```python
def ep_comm_compute_ratio(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    expert_intermediate_size: int,
    ep_size: int,
    dtype_bytes: int = 2,
    network_bw_GBs: float = 50,  # IB NDR
    gpu_flops_Tflops: float = 990,  # H100 bf16
):
    """分析 EP 的通信/计算比"""
    
    # All-to-All 通信量 (dispatch + combine)
    comm_bytes = 2 * num_tokens * top_k * hidden_size * dtype_bytes
    # 理想情况: 均匀分布, 每对 GPU 通信量 = total / ep_size
    # 但跨节点部分 ≈ comm_bytes * (1 - 1/ep_size)
    cross_node_bytes = comm_bytes * (1 - 1/ep_size)
    comm_time = cross_node_bytes / (network_bw_GBs * 1e9)
    
    # Expert 计算量
    # 每个 expert MLP: 3 × num_tokens × hidden × intermediate (gate+up+down)
    flops_per_token = 3 * 2 * hidden_size * expert_intermediate_size  # ×2 for mul+add
    total_flops = num_tokens * top_k * flops_per_token
    compute_per_gpu = total_flops / ep_size
    compute_time = compute_per_gpu / (gpu_flops_Tflops * 1e12)
    
    return {
        "comm_time_ms": comm_time * 1000,
        "compute_time_ms": compute_time * 1000,
        "ratio": comm_time / compute_time,
    }

# DeepSeek-V3 示例:
result = ep_comm_compute_ratio(
    num_tokens=4096, top_k=8, hidden_size=2048,
    expert_intermediate_size=2048, ep_size=32,
    network_bw_GBs=50, gpu_flops_Tflops=990,
)
# comm_time ≈ 2.0 ms
# compute_time ≈ 0.8 ms
# ratio ≈ 2.5 → 通信 bound!
```

### 7.2 优化方向

```
当 EP 通信 bound 时的优化方向:

1. 增大 batch size
   → 提高每个 expert 的计算量
   → 计算时间增长快于通信时间
   → 改善通信/计算比

2. 减少 EP size
   → 每 GPU 持有更多 expert
   → 更多 token 本地处理, 减少跨节点通信
   → 代价: 需要更多显存

3. Expert 复制 (冗余)
   → 热门 expert 在多个 GPU 上有副本
   → 更多 token 可以本地处理
   → 代价: 显存, 一致性管理

4. 通信-计算重叠
   → 在发送/接收时同时计算
   → 需要精心设计执行流水线

5. 更快的网络
   → NVLink > IB
   → 尽量将 EP 限制在节点内
```

## 8. EP 与其他并行方式的关系

```
EP 与 TP 的关系:
  TP: 切分每层的所有权重 (包括 attention + MLP)
  EP: 只切分 MoE 层的 expert 权重

  组合时:
  - Attention 用 TP (AllReduce)
  - Shared Expert 用 TP (AllReduce)
  - Routed Experts 用 EP (All-to-All)
  
  TP 和 EP 可以使用不同的并行度

EP 与 PP 的关系:
  PP: 按层切分
  EP: 按 expert 切分 (同一层内)
  
  两者正交, 可以组合:
  - PP stage 0: 层 0-30
  - PP stage 1: 层 31-60
  - 每个 stage 内部使用 EP 分配 expert

EP 与 DP 的关系:
  DP: 多个完整 replica 处理不同请求
  EP: 一个 replica 内部的 expert 分布
  
  组合: 多个 EP group, 每个 group 是一个 DP replica
```

## 9. 总结

| 要点 | 说明 |
|------|------|
| MoE 核心挑战 | 参数量大但稀疏激活，通信模式不规则 |
| EP 通信 | All-to-All，由 Router 动态决定，不可预测 |
| 负载不均 | MoE 推理的关键瓶颈，需要 EPLB 或 Elastic EP |
| 通信优化 | Hierarchical All-to-All + Compute-Comm Overlap |
| DeepSeek-V3 | 256 expert + EP 的工业级实践标杆 |
| vLLM 支持 | Fused MoE kernel + Elastic EP + EPLB |

---

> **下一节**：[Data Parallel 推理](04-data-parallel.md) —— 通过复制模型实例提升推理吞吐
