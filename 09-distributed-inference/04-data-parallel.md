# Data Parallel 推理

> 本节分析推理场景下 Data Parallelism (DP) 的适用场景、与 TP 的组合策略、请求分发与负载均衡，以及 vLLM 的 DP 实现。

## 1. 推理中的 DP 与训练中的 DP

### 1.1 训练 DP 回顾

```
训练 DP:
  - 每个 GPU 持有完整模型副本
  - 每个 GPU 处理不同的 mini-batch
  - 前向 + 反向后, AllReduce 同步梯度
  - 所有 GPU 保持权重一致

通信: 每步一次 AllReduce（梯度同步）
```

### 1.2 推理 DP 的特点

```
推理 DP:
  - 每个 replica 持有完整模型副本
  - 每个 replica 独立处理不同的请求
  - 无需梯度同步（权重只读）
  - replica 之间完全独立

通信: 无!（replica 间无通信需求）
```

这是推理 DP 最大的优势：**零通信开销**。每个 replica 是完全独立的推理实例，可以部署在不同节点上，甚至不同数据中心。

### 1.3 关键区别

| 维度 | 训练 DP | 推理 DP |
|------|---------|---------|
| 通信 | AllReduce 梯度同步 | 无（完全独立） |
| 一致性 | 所有 replica 权重必须一致 | 天然一致（只读） |
| Batch 形成 | 固定 batch size | 动态 continuous batching |
| 扩展方式 | 增加 GPU → 增大 global batch | 增加 replica → 增大吞吐 |
| 瓶颈 | 通信（梯度同步） | 负载均衡（请求分发） |

## 2. 何时需要 DP 推理

### 2.1 基本判断

```
DP 适用条件:
  1. 模型能放入单 GPU（或单节点用 TP）
  2. 需要提高吞吐量（QPS）
  3. 单个 replica 的吞吐无法满足需求

不适用:
  1. 模型太大, 单个 replica 就需要多节点
     → 先用 TP/PP/EP, 然后再考虑 DP
  2. 延迟是唯一目标
     → DP 不降低单请求延迟, 只提升吞吐
```

### 2.2 DP 的吞吐扩展

```
理论吞吐:
  Single replica: T tokens/s
  DP = N replicas: N × T tokens/s (理想线性扩展)

实际吞吐:
  受限于:
  1. 负载均衡效果 → 最慢的 replica 决定整体效率
  2. 前端 router 的分发开销 → 额外延迟
  3. GPU 利用率 → 低负载时 GPU 空闲

示例:
  单 replica (Llama-3-8B, 1×H100): ~3000 tokens/s
  DP=8 (8×H100): ~22000 tokens/s (91% 扩展效率)
  DP=32 (32×H100): ~85000 tokens/s (89% 扩展效率)
```

### 2.3 DP vs 多实例部署

```
方式 1: vLLM DP (--data-parallel-size)
  - 单个 vLLM 进程管理多个 replica
  - 内置请求分发
  - 共享 tokenizer, scheduler 逻辑
  - 统一的 KV Cache 管理视角

方式 2: 独立多实例 + 外部负载均衡器
  - 每个实例是独立的 vLLM 进程
  - 使用 Nginx/Envoy/HAProxy 做负载均衡
  - 各实例完全隔离
  - 更灵活（可以不同模型/配置）

对比:
                    vLLM DP          独立多实例
─────────────────────────────────────────────────
部署复杂度            低                中
负载均衡            内置, 智能         外部, 简单
Cache 亲和性         可实现             难实现
故障隔离              差               好
异构配置              不支持            支持
Scale up/down        需重启            独立扩缩
```

## 3. DP 与 TP 的组合

### 3.1 基本组合模式

```
DP + TP 组合 (最常见的生产配置):

  TP 用于单 replica 内部: 切分模型到节点内的多个 GPU
  DP 用于多 replica 之间: 复制模型到多个节点

示例: 4 节点, 每节点 8 GPU, 模型 70B

方案 1: TP=8, DP=4
  节点 0: [GPU 0-7] = Replica 0 (TP=8)
  节点 1: [GPU 0-7] = Replica 1 (TP=8)
  节点 2: [GPU 0-7] = Replica 2 (TP=8)
  节点 3: [GPU 0-7] = Replica 3 (TP=8)
  
  吞吐: 4 × 单 replica 吞吐

方案 2: TP=4, DP=8
  节点 0: [GPU 0-3] = Replica 0, [GPU 4-7] = Replica 1
  节点 1: [GPU 0-3] = Replica 2, [GPU 4-7] = Replica 3
  节点 2: [GPU 0-3] = Replica 4, [GPU 4-7] = Replica 5
  节点 3: [GPU 0-3] = Replica 6, [GPU 4-7] = Replica 7
  
  吞吐: 8 × 单 replica 吞吐（但每 replica 的 KV Cache 更小）
```

### 3.2 TP vs DP 的 GPU 分配权衡

```
给定 N 个 GPU, 模型大小 M:

TP 越大:
  + 单 replica 延迟越低 (并行计算)
  + 单 replica KV Cache 容量越大 (显存分摊)
  - replica 数越少 (DP = N/TP)
  - TP 通信开销越大
  - 吞吐扩展受限

DP 越大:
  + replica 数越多, 总吞吐越高
  + 零通信开销
  - 单 replica 延迟不变 (各 replica 独立)
  - 单 replica KV Cache 有限

最优配置取决于:
  1. SLA 要求 (延迟上限)
  2. 目标吞吐 (QPS 要求)
  3. 请求特征 (prompt 长度分布, 生成长度分布)
```

### 3.3 决策示例

```python
def choose_tp_dp(
    num_gpus: int,
    model_size_GB: float,
    gpu_memory_GB: float = 80,
    target_latency_ms: float = 50,    # TPOT
    kv_cache_per_token_MB: float = 1, # KV Cache per token per GPU
):
    """选择 TP 和 DP 配置"""
    
    results = []
    for tp in [1, 2, 4, 8]:
        if num_gpus % tp != 0:
            continue
        dp = num_gpus // tp
        
        # 检查权重是否放得下
        weight_per_gpu = model_size_GB / tp
        if weight_per_gpu > gpu_memory_GB * 0.6:  # 留 40% 给 KV Cache
            continue
        
        # 估算 KV Cache 容量
        kv_memory_per_gpu = gpu_memory_GB - weight_per_gpu - 2  # 2GB for overhead
        max_tokens_per_replica = kv_memory_per_gpu * 1024 / kv_cache_per_token_MB
        total_max_tokens = max_tokens_per_replica * dp
        
        # 估算延迟 (简化)
        # TP 越大, decode 延迟越低 (但有通信开销)
        base_latency = model_size_GB / tp * 0.5  # 简化的延迟模型
        comm_overhead = tp * 0.5  # ms
        estimated_latency = base_latency + comm_overhead
        
        if estimated_latency <= target_latency_ms:
            results.append({
                "tp": tp, "dp": dp,
                "latency_ms": estimated_latency,
                "total_kv_tokens": total_max_tokens,
            })
    
    # 选择吞吐最高的配置 (total_kv_tokens × dp)
    return max(results, key=lambda x: x["total_kv_tokens"])
```

## 4. vLLM DP 配置与实现

### 4.1 配置方式

```bash
# vLLM Data Parallel 配置
vllm serve meta-llama/Llama-3-8B-Instruct \
    --data-parallel-size 4 \
    --tensor-parallel-size 1

# TP + DP 组合
vllm serve meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --data-parallel-size 2
# 总共需要 4 × 2 = 8 GPU
```

### 4.2 vLLM DP 架构

```
vLLM DP 内部架构:

                    ┌─────────────┐
                    │   API Server │
                    │  (FastAPI)   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Router /   │
                    │  Dispatcher  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Replica 0│ │ Replica 1│ │ Replica 2│
        │ (TP=N)   │ │ (TP=N)   │ │ (TP=N)   │
        │Scheduler │ │Scheduler │ │Scheduler │
        │ Workers  │ │ Workers  │ │ Workers  │
        │ KV Cache │ │ KV Cache │ │ KV Cache │
        └──────────┘ └──────────┘ └──────────┘

每个 replica:
  - 独立的 Scheduler
  - 独立的 Worker(s) (如果 TP>1 则多个 worker)
  - 独立的 KV Cache Pool
  - 独立的 continuous batching
```

### 4.3 请求分发流程

```python
# vLLM DP 请求分发 (简化):

class DPRouter:
    """DP 请求分发器"""
    
    def __init__(self, dp_size: int, policy: str = "round_robin"):
        self.dp_size = dp_size
        self.policy = policy
        self.replica_loads = [0] * dp_size  # 各 replica 当前负载
    
    def route(self, request: Request) -> int:
        """选择目标 replica"""
        if self.policy == "round_robin":
            return self._round_robin(request)
        elif self.policy == "least_load":
            return self._least_load(request)
        elif self.policy == "cache_aware":
            return self._cache_aware(request)
    
    def _round_robin(self, request):
        """轮询分发"""
        self.counter += 1
        return self.counter % self.dp_size
    
    def _least_load(self, request):
        """最小负载分发"""
        # 负载 = 当前处理的 token 总数
        return min(range(self.dp_size), 
                   key=lambda i: self.replica_loads[i])
    
    def _cache_aware(self, request):
        """Cache 感知分发"""
        # 将相似 prompt 的请求路由到同一 replica
        # 提高 prefix cache hit rate
        prefix_hash = hash(request.prompt[:1024])  # hash 前 1024 字符
        return prefix_hash % self.dp_size
```

## 5. 请求分发策略深度分析

### 5.1 Round Robin

```
Round Robin (轮询):

请求序列:  R0 R1 R2 R3 R4 R5 R6 R7 ...
Replica 0:  R0      R3      R6
Replica 1:     R1      R4      R7
Replica 2:        R2      R5      ...

优点:
  - 实现最简单
  - 无状态, 无需收集负载信息
  
缺点:
  - 不考虑请求长度差异
  - 可能导致严重不均衡

问题示例:
  R0: prompt=100 tokens, max_output=10
  R1: prompt=10000 tokens, max_output=2000
  R2: prompt=50 tokens, max_output=5
  
  Replica 0 很快完成, Replica 1 长时间忙碌
  → GPU 利用率不均衡
```

### 5.2 Least Load (最小负载)

```
Least Load:

负载度量选择:
  1. 当前处理的请求数 → 简单但不准确
  2. 当前处理的 token 总数 → 更准确
  3. 预估剩余计算量 → 最准确但难以预测
  4. GPU 利用率 → 直接但有延迟

实现挑战:
  - 负载信息有延迟 (需要从各 replica 收集)
  - 长请求的预估不准确
  - 高并发时多个请求可能同时选择同一 replica

改进: Weighted Least Load
  load_score = active_tokens + pending_tokens * 0.5
  选择 load_score 最小的 replica
```

### 5.3 Cache-Aware Routing

```
Cache-Aware Routing (缓存感知路由):

核心思想:
  如果两个请求有相同的 prefix (如 system prompt),
  路由到同一个 replica 可以复用 prefix cache,
  减少重复的 prefill 计算。

实现:
  1. 计算请求 prompt 的 prefix hash
  2. 使用 consistent hashing 将 hash 映射到 replica
  3. 相同 prefix 的请求自动路由到同一 replica

示例:
  系统提示 A: "你是一个编程助手..."  → hash = 0x3F → Replica 0
  系统提示 B: "你是一个翻译..."     → hash = 0x7A → Replica 1
  
  所有用系统提示 A 的请求 → Replica 0 (cache hit!)
  所有用系统提示 B 的请求 → Replica 1 (cache hit!)

挑战:
  - Prefix 分布不均 → 某些 replica 过载
  - 需要 fallback: 如果目标 replica 过载, 降级到其他 replica
  - Cache 命中率 vs 负载均衡 的 trade-off
```

### 5.4 Hybrid Routing

```python
# 混合路由策略 (推荐生产使用):

class HybridRouter:
    def route(self, request: Request) -> int:
        # 1. 优先考虑 cache 亲和性
        preferred_replica = self._cache_preferred(request)
        
        # 2. 检查 preferred replica 的负载
        if self._is_overloaded(preferred_replica):
            # 3. 过载时降级到 least-load
            return self._least_load(request)
        
        return preferred_replica
    
    def _is_overloaded(self, replica_id: int) -> bool:
        avg_load = sum(self.replica_loads) / self.dp_size
        return self.replica_loads[replica_id] > avg_load * 1.5
```

## 6. Load Balancing 挑战

### 6.1 请求长度不均匀

```
实际请求长度分布（典型对话场景）:

Prompt 长度:
  P10 = 50 tokens
  P50 = 200 tokens
  P90 = 2000 tokens
  P99 = 8000 tokens
  
Output 长度:
  P10 = 10 tokens
  P50 = 100 tokens
  P90 = 500 tokens
  P99 = 2000 tokens

问题:
  一个 P99 请求 (8000+2000 tokens) 的计算量
  ≈ 一个 P10 请求 (50+10 tokens) 的 167 倍
  
  如果 P99 请求集中在某个 replica → 严重不均衡
```

### 6.2 KV Cache 碎片化

```
DP 中的 KV Cache 管理:

每个 replica 独立管理自己的 KV Cache pool

问题场景:
  Replica 0: KV Cache 使用率 95% (快满了)
  Replica 1: KV Cache 使用率 30% (大量空闲)
  
  新请求路由到 Replica 0 → 可能因 KV Cache 不足而排队
  新请求路由到 Replica 1 → 可以立即处理, 但可能错过 cache hit

解决方案:
  1. 将 KV Cache 使用率纳入路由决策
  2. 当 replica KV Cache 接近满时, 降低其被选择的概率
  3. 支持请求迁移: 将请求从高负载 replica 移到低负载 replica
     (需要迁移 KV Cache, 开销较大)
```

### 6.3 长尾延迟

```
DP 场景下的长尾延迟:

原因:
  1. 请求长度差异 → 某些 replica 处理时间长
  2. Cache miss → 某些请求需要完整 prefill
  3. GPU 频率波动 → 热节流导致性能下降
  4. 内存压力 → KV Cache eviction 导致重算

缓解策略:
  1. 请求预估: 根据 prompt 长度和 max_tokens 预估计算量
  2. Preemption: 允许高优先级请求抢占低优先级请求
  3. Request hedging: 将同一请求发给两个 replica, 取先完成的
  4. 动态 batch 调整: 高负载时限制每 replica 的并发请求数
```

## 7. DP 的扩展性分析

### 7.1 线性扩展的条件

```
DP 理论上可以线性扩展吞吐, 条件:

1. 负载完全均衡 (每个 replica 处理量相同)
2. 前端 router 不成为瓶颈
3. 每个 replica 有足够的 GPU 利用率
4. 网络带宽不是瓶颈 (DP replica 间无通信)

实际限制:
1. 负载不均衡 → 效率 < 100%
2. 共享资源竞争 → 网络 I/O, CPU tokenization
3. 长尾请求 → 部分 replica 卡在长请求上
```

### 7.2 DP 扩展效率

```python
def dp_scaling_efficiency(
    dp_size: int,
    request_distribution: str = "uniform",
    cache_hit_rate: float = 0.3,
):
    """估算 DP 扩展效率"""
    
    # 基础效率 (负载均衡)
    if request_distribution == "uniform":
        lb_efficiency = 0.98  # 几乎完美
    elif request_distribution == "skewed":
        lb_efficiency = 0.85  # 长度差异导致不均衡
    elif request_distribution == "highly_skewed":
        lb_efficiency = 0.70  # 极端不均衡
    
    # Cache 效率损失
    # DP 越大, 每个 replica 收到的相似请求越少, cache hit rate 下降
    effective_cache_rate = cache_hit_rate * (1 / dp_size ** 0.3)
    cache_penalty = 1 - (cache_hit_rate - effective_cache_rate) * 0.2
    
    # 总效率
    efficiency = lb_efficiency * cache_penalty
    
    return {
        "dp_size": dp_size,
        "lb_efficiency": lb_efficiency,
        "cache_penalty": cache_penalty,
        "total_efficiency": efficiency,
        "effective_throughput_ratio": dp_size * efficiency,
    }

# 示例:
# DP=4, skewed: efficiency=83%, effective=3.32x (vs 理想 4x)
# DP=8, skewed: efficiency=80%, effective=6.40x (vs 理想 8x)
# DP=16, skewed: efficiency=77%, effective=12.3x (vs 理想 16x)
```

## 8. DP 与 Disaggregated Serving

### 8.1 DP 在 PD 分离架构中的应用

```
Prefill-Decode 分离 + DP:

Prefill 集群:
  N 个 prefill replica (DP)
  每个 replica 专注于 prefill 计算
  可以用更大的 TP (延迟容忍度更高)
  
Decode 集群:
  M 个 decode replica (DP)
  每个 replica 专注于 decode 计算
  需要更小的 TP (延迟敏感)

请求流程:
  1. Router 选择 prefill replica (负载均衡)
  2. Prefill replica 计算完成, 发送 KV Cache 到 decode replica
  3. Router 选择 decode replica (负载均衡)
  4. Decode replica 生成 tokens

DP 在 PD 分离中的优势:
  - Prefill 和 decode 可以独立扩展
  - Prefill DP 和 Decode DP 可以不同
  - 更精细的资源分配
```

### 8.2 全局 Cache 管理

```
多 DP replica 的全局 cache 管理:

传统 DP:
  每个 replica 独立管理 cache → cache 不共享
  相同 prefix 在多个 replica 上各存一份 → 浪费

优化方案:
  1. Global Cache Directory:
     维护一个全局的 cache 索引
     知道每个 prefix 在哪个 replica 上有 cache
     → 将请求路由到有 cache 的 replica

  2. Shared Cache Pool (需要 RDMA):
     多个 replica 共享分布式 KV Cache
     cache hit 时通过 RDMA 远程读取
     → 减少重复存储, 提高 cache hit rate

  3. Cache Broadcast:
     当一个 replica 计算了新的 prefix cache
     广播给其他 replica (后台异步)
     → 所有 replica 都有热门 prefix 的 cache
```

## 9. 生产环境 DP 最佳实践

```
1. 选择合适的 TP×DP 组合:
   - 满足延迟 SLA 的前提下, 最大化 DP
   - 例: 70B 模型, 32 GPU
     TP=4 DP=8 (推荐) vs TP=8 DP=4

2. 使用混合路由策略:
   - Cache-aware + 负载感知
   - 过载时降级到 least-load

3. 监控与告警:
   - 每 replica 的 QPS, 延迟 P50/P90/P99
   - KV Cache 使用率
   - 队列深度
   - GPU 利用率

4. 弹性扩缩:
   - 根据 QPS 动态增减 DP replica
   - 低峰期减少 replica 省成本
   - 高峰期增加 replica 保 SLA

5. 健康检查:
   - 定期检查每个 replica 的响应时间
   - 自动摘除不健康的 replica
   - 自动替换故障 replica
```

## 10. 总结

| 要点 | 说明 |
|------|------|
| DP 核心优势 | 零通信开销，线性扩展吞吐 |
| 适用场景 | 模型单节点可放下，需要提升 QPS |
| 与 TP 组合 | TP 在节点内降延迟，DP 在节点间提吞吐 |
| 核心挑战 | 负载均衡（请求长度不均、cache 亲和性） |
| 路由策略 | Round Robin → Least Load → Cache Aware → Hybrid |
| 生产建议 | 混合路由 + 弹性扩缩 + 全局 Cache 管理 |

---

> **下一节**：[Context Parallel](05-context-parallel.md) —— 超长上下文推理的序列级并行策略
