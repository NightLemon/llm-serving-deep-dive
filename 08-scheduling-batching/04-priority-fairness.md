# 请求优先级与公平性

> 当多个请求竞争有限的 GPU 资源时，如何决定谁先执行、如何避免饥饿、如何满足 SLA

## 1. 为什么需要优先级调度？

在生产环境中，LLM serving 系统面对的请求并非同质的：

```
场景 1：多租户 API 服务
  - 付费用户（高优先级）：需要低延迟、稳定响应
  - 免费用户（低优先级）：可以容忍较高延迟
  - 内部批量任务（最低优先级）：不关心延迟，只关心最终完成

场景 2：混合工作负载
  - 交互式聊天（高优先级）：TTFT < 500ms, TBT < 50ms
  - 后台摘要生成（低优先级）：TTFT < 5s, TBT 不敏感
  - 数据标注（批量，最低优先级）：无 SLA

场景 3：流量突增
  - 突然涌入大量请求，超过系统容量
  - 需要选择性地降级低优先级请求
  - 保证高优先级请求的 SLA 不受影响
```

默认的 FCFS（先来先服务）调度策略在这些场景下无法满足需求——一个到达较早的低优先级长请求可能长时间占据 GPU 资源，导致后来的高优先级请求排队等待。

## 2. vLLM 的优先级机制

### 2.1 Priority 参数

vLLM 在 API 层面支持请求优先级：

```python
# OpenAI 兼容 API 调用
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1")

# 高优先级请求（数值越小优先级越高）
response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[{"role": "user", "content": "紧急：解释量子计算"}],
    extra_body={"priority": 0}  # 最高优先级
)

# 低优先级请求
response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[{"role": "user", "content": "总结这篇文章..."}],
    extra_body={"priority": 10}  # 较低优先级
)
```

启用优先级调度：

```bash
vllm serve model_name --scheduling-policy priority
```

### 2.2 优先级队列实现

当配置为 `priority` 调度策略时，waiting queue 使用优先级队列：

```python
class PriorityRequestQueue(RequestQueue):
    """
    优先级请求队列。
    优先级数值越小 → 优先级越高 → 越先被调度。
    相同优先级按到达时间排序（FCFS）。
    """

    def __init__(self):
        self._queue: list[Request] = []

    def add(self, request: Request) -> None:
        # 使用 (priority, arrival_time) 作为排序键
        heapq.heappush(self._queue, request)

    def __iter__(self) -> Iterator[Request]:
        # 按 priority 从小到大（最高优先级优先）
        return iter(sorted(
            self._queue,
            key=lambda r: (r.priority, r.arrival_time)
        ))
```

`Request` 对象实现了比较操作：

```python
class Request:
    priority: int = 0
    arrival_time: float = 0.0

    def __lt__(self, other: "Request") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority  # 数值小 → 优先级高
        return self.arrival_time < other.arrival_time  # 同优先级 → FCFS
```

### 2.3 优先级抢占（Priority Preemption）

优先级调度的核心机制是 **抢占**：当高优先级请求到达时，如果 GPU 显存不足，可以抢占正在运行的低优先级请求的 KV Cache。

```python
# 调度器中的抢占逻辑（简化）
def _try_schedule_high_priority(self, new_req: Request, budget):
    """尝试调度一个高优先级请求，必要时抢占低优先级请求"""

    # 1. 先尝试正常分配
    status = self.kv_cache_manager.allocate_slots(
        new_req, new_req.num_prompt_tokens
    )
    if status == AllocationStatus.OK:
        return True  # 无需抢占

    # 2. 显存不足，查找可抢占的低优先级请求
    # 按优先级从低到高排序 running 请求
    preempt_candidates = sorted(
        self.running_requests.values(),
        key=lambda r: (-r.priority, -r.arrival_time)
        # 优先级最低（数值最大）的排在前面
    )

    freed_blocks = 0
    needed_blocks = self._estimate_needed_blocks(new_req)
    preempted = []

    for victim in preempt_candidates:
        # 只有优先级严格低于新请求才能被抢占
        if victim.priority <= new_req.priority:
            break  # 不能抢占同级或更高优先级的请求

        # 抢占 victim
        freed = self.kv_cache_manager.free(victim)
        freed_blocks += freed
        preempted.append(victim)

        if freed_blocks >= needed_blocks:
            break  # 释放够了

    if freed_blocks >= needed_blocks:
        # 抢占成功，被抢占的请求放回 waiting queue
        for victim in preempted:
            del self.running_requests[victim.request_id]
            victim.status = RequestStatus.PREEMPTED
            self.waiting_queue.add(victim)

        # 为新请求分配
        self.kv_cache_manager.allocate_slots(
            new_req, new_req.num_prompt_tokens
        )
        return True
    else:
        # 即使抢占所有低优先级请求也不够
        # 恢复被释放的 blocks（或让它们保持释放状态等待下轮）
        return False
```

**抢占的代价：**

被抢占的请求失去了已计算的 KV Cache。当它重新被调度时，需要重新执行 prefill：

```
请求 A（低优先级）：已经 decode 到第 50 个 token
  → 被抢占
  → KV Cache 全部释放
  → 重新进入 waiting queue
  → 下次调度时需要重新 prefill 整个 prompt + 已生成的 50 个 token
  → 浪费了之前的计算
```

这就是优先级调度的**效率代价**——为了满足高优先级请求的低延迟，系统整体吞吐量会有所下降。

## 3. 避免饥饿：Aging 机制

### 3.1 饥饿问题

在纯优先级调度下，低优先级请求可能"永远"得不到执行：

```
时间 →
高优先级请求:  [H1][H2][H3][H4][H5][H6]...  （持续到达）
低优先级请求:  [L1 等待中...][L1 等待中...][L1 等待中...]...  （一直被延后）
```

如果高优先级请求的到达速率超过系统处理能力，低优先级请求可能等待无限长时间。这在生产系统中是不可接受的——即使是免费用户，也应该在"合理时间"内获得响应。

### 3.2 Aging 机制的设计

Aging（老化）是解决饥饿的经典方法：随着等待时间增长，请求的有效优先级逐渐提升。

```python
class Request:
    priority: int               # 原始优先级
    arrival_time: float         # 到达时间

    def effective_priority(self, current_time: float) -> float:
        """
        计算有效优先级。
        等待越久，有效优先级越高（数值越小）。
        """
        wait_time = current_time - self.arrival_time
        # aging_factor 控制老化速度
        # 例如 aging_factor = 0.1 表示每等 10 秒，有效优先级提升 1
        aging_bonus = wait_time * self.aging_factor
        return self.priority - aging_bonus
```

**Aging 策略的参数选择：**

| 参数 | 含义 | 选择建议 |
|------|------|---------|
| `aging_factor` | 每秒优先级提升量 | 0.01 ~ 1.0，取决于优先级范围 |
| `max_aging` | 最大 aging 值（防止低优先级完全覆盖高优先级） | 通常设为最高和最低优先级差值的 50-80% |
| `aging_start_delay` | 开始 aging 前的等待时间 | 0 ~ 30 秒 |

**示例配置：**

```
优先级范围: 0（最高）到 10（最低）
aging_factor: 0.5 /秒
max_aging: 8

请求 A (priority=10) 到达后:
  t=0s:  effective_priority = 10 - 0 = 10
  t=5s:  effective_priority = 10 - 2.5 = 7.5
  t=10s: effective_priority = 10 - 5.0 = 5.0
  t=16s: effective_priority = 10 - 8.0 = 2.0  (达到 max_aging)
  t=20s: effective_priority = 10 - 8.0 = 2.0  (不再继续提升)

→ 低优先级请求最多等待约 16 秒就能获得接近高优先级的调度权重
```

### 3.3 饥饿保护的替代方案

除了 Aging，还有其他防止饥饿的方法：

**1. 最大等待时间保证**

```python
def schedule(self):
    # 先检查是否有等待超时的请求
    for req in self.waiting_queue:
        if req.wait_time() > MAX_WAIT_TIME:
            # 强制调度，无论优先级
            self._force_schedule(req)
```

**2. 保留容量（Reserved Capacity）**

```python
# 为每个优先级保留一定比例的 GPU 容量
capacity_reservation = {
    0: 0.5,   # 50% 容量保留给最高优先级
    5: 0.3,   # 30% 保留给中等优先级
    10: 0.2,  # 20% 保留给最低优先级
}
```

**3. 权重公平队列（Weighted Fair Queuing）**

```python
# 类似网络调度中的 WFQ
# 高优先级请求获得更多"虚拟时间"配额
virtual_time_weight = {
    0: 5.0,   # 高优先级获得 5x 调度权重
    5: 2.0,   # 中等优先级获得 2x
    10: 1.0,  # 低优先级获得 1x
}
```

## 4. Token 级公平 vs 请求级公平

### 4.1 请求级公平（Request-Level Fairness）

请求级公平的目标是**每个请求获得大致相同的等待时间**。

```
公平性指标: max_wait / min_wait → 越接近 1 越公平

FCFS 调度: 天然的请求级公平（先来先服务）
优先级调度: 请求级不公平（高优先级永远优先）
```

**问题**：请求级公平忽略了请求的大小差异。一个 10K prompt 的请求消耗的资源远大于一个 100 token 的请求，但在请求级公平下它们获得相同的调度优先权。

### 4.2 Token 级公平（Token-Level Fairness）

Token 级公平的目标是**每个请求消耗的 GPU 计算时间与其 token 数成正比**。

```python
# Token 级公平调度
def fair_priority(req):
    """
    每个请求的公平优先级取决于它已消耗的"服务量"。
    服务量越少的请求优先级越高。
    """
    service_received = req.num_computed_tokens * cost_per_token
    return service_received  # 服务量越少 → 优先级越高
```

**Deficit Round Robin (DRR) 风格调度：**

```python
class DRRScheduler:
    """
    基于 Deficit Round Robin 的 token 级公平调度。
    每个请求有一个 deficit counter（欠债计数器）。
    """
    def __init__(self, quantum: int = 512):
        self.quantum = quantum  # 每轮给每个请求的 token 配额

    def schedule(self, requests):
        scheduled = []
        for req in requests:
            req.deficit += self.quantum
            tokens_to_schedule = min(
                req.deficit,
                req.remaining_tokens
            )
            if tokens_to_schedule > 0:
                scheduled.append((req, tokens_to_schedule))
                req.deficit -= tokens_to_schedule
        return scheduled
```

### 4.3 混合公平策略

实际系统通常结合优先级和公平性：

```
层级 1: 按优先级分组（高 / 中 / 低）
层级 2: 每个优先级组内，按 Token 级公平调度
层级 3: 组间按加权公平共享 GPU 资源

示例:
  高优先级组（权重 60%）：请求 A, B, C → 组内 FCFS
  中优先级组（权重 30%）：请求 D, E → 组内 FCFS
  低优先级组（权重 10%）：请求 F, G, H → 组内 FCFS
```

## 5. SLA-Aware 调度

### 5.1 LLM Serving 的 SLA 指标

LLM serving 有两个关键延迟指标：

| 指标 | 全称 | 含义 | 典型 SLA |
|------|------|------|---------|
| TTFT | Time To First Token | 从请求到达到第一个 token 生成的时间 | P99 < 500ms ~ 2s |
| TBT | Time Between Tokens | 两个连续 token 之间的间隔 | P99 < 50ms ~ 200ms |

还有衍生指标：

| 指标 | 含义 | 计算 |
|------|------|------|
| TPOT | Time Per Output Token | 总生成时间 / 输出 token 数 |
| E2E Latency | 端到端延迟 | TTFT + output_tokens * TBT |
| Normalized Latency | 归一化延迟 | E2E / output_tokens |

### 5.2 TTFT SLA 感知调度

**目标**：控制等待队列中请求的最大等待时间，使 TTFT 不超过 SLA。

```python
def ttft_aware_schedule(self, budget):
    """
    TTFT 感知调度：优先调度即将违反 TTFT SLA 的请求。
    """
    current_time = time.time()
    urgent_requests = []
    normal_requests = []

    for req in self.waiting_queue:
        wait_time = current_time - req.arrival_time
        ttft_slack = req.ttft_sla - wait_time
        # ttft_slack: 距离违反 SLA 还剩多少时间

        if ttft_slack < URGENT_THRESHOLD:
            urgent_requests.append((ttft_slack, req))
        else:
            normal_requests.append(req)

    # 优先调度紧急请求
    urgent_requests.sort(key=lambda x: x[0])  # 最紧急的优先

    scheduled = []
    for _, req in urgent_requests:
        if self._try_schedule(req, budget):
            scheduled.append(req)

    # 然后调度正常请求
    for req in normal_requests:
        if not budget.has_remaining():
            break
        if self._try_schedule(req, budget):
            scheduled.append(req)

    return scheduled
```

### 5.3 TBT SLA 感知调度

**目标**：控制每个 iteration 的执行时间，使 decode 请求的 TBT 不超过 SLA。

```python
def tbt_aware_budget(self, tbt_sla_ms: float) -> int:
    """
    基于 TBT SLA 计算每轮最大 token 预算。

    核心思想：iteration 耗时 ≈ f(num_tokens)
    要保证 iteration 耗时 < TBT SLA
    → 反推出 max_num_batched_tokens
    """
    # 通过 profiling 或建模得到 iteration 耗时函数
    # t(n) = a * n + b (线性近似)
    # 解方程：a * n + b < tbt_sla_ms
    max_tokens = int((tbt_sla_ms - self.b) / self.a)
    return max(max_tokens, MIN_BUDGET)
```

**动态 Budget 调整：**

```python
class AdaptiveBudgetController:
    """
    根据实际 TBT 观测值动态调整 budget。
    采用 AIMD（Additive Increase Multiplicative Decrease）策略。
    """

    def __init__(self, tbt_sla_ms: float, initial_budget: int = 2048):
        self.tbt_sla = tbt_sla_ms
        self.budget = initial_budget

    def update(self, observed_tbt_ms: float):
        if observed_tbt_ms > self.tbt_sla * 0.9:
            # 接近 SLA 边界，减少 budget（multiplicative decrease）
            self.budget = int(self.budget * 0.8)
        elif observed_tbt_ms < self.tbt_sla * 0.5:
            # 远低于 SLA，增加 budget（additive increase）
            self.budget = min(self.budget + 128, MAX_BUDGET)
```

## 6. 吞吐量与延迟的权衡

### 6.1 基本权衡关系

吞吐量和延迟之间存在根本性的矛盾：

```
吞吐量优化方向:
  → 增大 batch size
  → 增大 max_num_batched_tokens
  → 减少调度频率
  → 结果: 每个 iteration 更多 token → iteration 耗时更长 → 延迟升高

延迟优化方向:
  → 减小 batch size
  → 减小 max_num_batched_tokens
  → 优先调度已在执行的请求
  → 结果: 每个 iteration 更少 token → GPU 利用率下降 → 吞吐量降低
```

### 6.2 Pareto 最优曲线

```
延迟
  ↑
  │   *  (小 batch，高延迟保证但低吞吐)
  │
  │     *
  │       *  ← Pareto 最优边界
  │         *
  │           *
  │              *  (大 batch，高吞吐但延迟波动)
  └──────────────────→ 吞吐量
```

不同业务场景在这条曲线上选择不同的工作点：

| 业务场景 | 工作点 | 配置示例 |
|---------|--------|---------|
| 实时聊天 | 低延迟端 | max_seqs=64, budget=1024 |
| API 服务 | 平衡点 | max_seqs=128, budget=2048 |
| 批处理 | 高吞吐端 | max_seqs=512, budget=8192 |

### 6.3 实际调优建议

```bash
# 场景 1: 延迟敏感（聊天应用）
vllm serve model_name \
    --max-num-seqs 64 \
    --max-num-batched-tokens 1024 \
    --enable-chunked-prefill \
    --scheduling-policy priority

# 场景 2: 吞吐量优先（批量处理）
vllm serve model_name \
    --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill

# 场景 3: 混合工作负载（API 服务）
vllm serve model_name \
    --max-num-seqs 256 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --scheduling-policy priority
```

## 7. 前沿研究方向

### 7.1 预测感知调度（Prediction-Aware Scheduling）

利用 output length 预测来优化调度：

```python
def prediction_aware_schedule(self, req):
    """
    预测请求的输出长度，据此做调度决策。
    短输出请求优先：它们很快释放资源。
    """
    predicted_output_len = self.length_predictor.predict(req)

    # 短请求优先（SRPT: Shortest Remaining Processing Time）
    req.scheduling_key = predicted_output_len
```

相关工作：
- [S3 (2023)](https://arxiv.org/abs/2306.06000) 使用输出长度预测来优化调度
- [Andes (2024)](https://arxiv.org/abs/2404.16283) 基于 SLA 和预测长度的质量感知调度

### 7.2 多模型调度

当多个模型部署在同一 GPU 集群上时，调度变得更加复杂：

```
GPU 0: Model A (7B) + Model B (7B)  ← 共享显存
GPU 1: Model C (13B)                ← 独占

调度器需要考虑：
  - 模型切换的开销（加载权重）
  - KV Cache 在不同模型间的分配
  - 跨模型的优先级管理
```

### 7.3 能耗感知调度

在大规模部署中，能耗成为重要考量：

```
低负载时段: 降低 batch size，允许 GPU 降频 → 省电
高负载时段: 最大化 batch size，GPU 满频运行 → 保 SLA
```

## 8. 参考资料

- [FastServe: Fast Preemption-based Serving for Large Language Models (2023)](https://arxiv.org/abs/2305.05920)
- [Andes: Quality-Aware Serving for LLM Inference (2024)](https://arxiv.org/abs/2404.16283)
- [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput (2023)](https://arxiv.org/abs/2306.06000)
- [vLLM Priority Scheduling 文档](https://docs.vllm.ai/en/latest/features/scheduling.html)
- [Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180)
