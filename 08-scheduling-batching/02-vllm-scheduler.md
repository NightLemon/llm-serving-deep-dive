# vLLM Scheduler 源码走读

> 基于 vLLM v1 架构（2025 年 Q3 起成为默认架构），深入解析 `vllm/v1/core/sched/` 下的调度器实现

## 1. 概述：vLLM v1 调度器的定位

vLLM v1 的调度器是整个 serving 系统的"大脑"，它在每个 iteration 决定：

- 哪些请求可以继续执行（running 队列）
- 哪些新请求可以加入执行（waiting 队列）
- 哪些请求需要被暂停/换出（preemption）
- 总共需要处理多少 token

调度器的核心文件结构：

```
vllm/v1/core/sched/
├── scheduler.py          # 主调度器 Scheduler 类
├── request_queue.py       # 请求队列管理（FCFS、优先级）
├── output.py             # SchedulerOutput 数据结构
└── async_scheduler.py    # 异步调度器（DBO，见 05-dbo.md）
```

## 2. 核心数据结构

### 2.1 请求的生命周期

在 vLLM v1 中，一个请求从到达到完成经历以下状态：

```
                ┌──────────┐
    到达 ──────→│ WAITING  │
                └────┬─────┘
                     │ 分配到 KV blocks，加入执行
                     ↓
                ┌──────────┐
                │ RUNNING  │←──────────────┐
                └────┬─────┘               │
                     │                     │
              ┌──────┴──────┐              │
              ↓             ↓              │
        ┌──────────┐  ┌──────────┐         │
        │ FINISHED │  │ PREEMPTED│─────────┘
        └──────────┘  └──────────┘
                        （重新等待调度）
```

### 2.2 SchedulerOutput（output.py）

`SchedulerOutput` 是调度器的输出，传递给 Model Runner 执行。它包含了 GPU 执行一个 iteration 所需的全部信息：

```python
@dataclass
class SchedulerOutput:
    # 本轮需要执行的请求及其调度元数据
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: list[CachedRequestData]
    scheduled_resumed_reqs: list[ResumedRequestData]

    # 本轮需要处理的总 token 数
    num_scheduled_tokens: dict[str, int]  # req_id → num_tokens

    # Block 操作指令
    # 告诉 worker 哪些 block 需要从 GPU swap 到 CPU，或反向
    finished_req_ids: set[str]
    free_encoder_input_ids: list[tuple[str, int]]

    # Preemption 相关
    preempted_req_ids: set[str]

    # 调度元数据
    total_num_scheduled_tokens: int
    # 用于 grammar-guided decoding
    grammar_bitmask: Optional[torch.Tensor]
```

关键字段说明：
- `scheduled_new_reqs`: 本轮首次执行的请求（需要完整初始化）
- `scheduled_cached_reqs`: 已经在运行中的请求（继续 decode 或 chunked prefill 的后续 chunk）
- `num_scheduled_tokens`: 每个请求在本轮要处理的 token 数（decode 时为 1，prefill 或 chunked prefill 时可能为多个）

### 2.3 请求队列（request_queue.py）

vLLM v1 抽象了请求队列接口，支持不同的排序策略：

```python
class RequestQueue(abc.ABC):
    """请求队列抽象基类"""

    @abc.abstractmethod
    def add(self, request: Request) -> None:
        """添加请求到队列"""
        ...

    @abc.abstractmethod
    def remove(self, request_id: str) -> Optional[Request]:
        """移除指定请求"""
        ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """按调度优先级迭代请求"""
        ...

    @abc.abstractmethod
    def __len__(self) -> int:
        ...
```

**FCFS 队列（FIFORequestQueue）：**

```python
class FIFORequestQueue(RequestQueue):
    """先来先服务的请求队列"""

    def __init__(self):
        self._queue: deque[Request] = deque()

    def add(self, request: Request) -> None:
        self._queue.append(request)

    def __iter__(self) -> Iterator[Request]:
        # 按到达顺序迭代
        return iter(self._queue)
```

**优先级队列（PriorityRequestQueue）：**

```python
class PriorityRequestQueue(RequestQueue):
    """支持优先级的请求队列，优先级数值越小越优先"""

    def __init__(self):
        self._queue: list[Request] = []

    def add(self, request: Request) -> None:
        heapq.heappush(self._queue, request)

    def __iter__(self) -> Iterator[Request]:
        # 按优先级排序迭代
        return iter(sorted(self._queue))
```

队列的选择通过配置决定：当用户设置了 `--scheduling-policy priority` 时使用优先级队列，否则默认使用 FCFS。

## 3. Scheduler 主类详解（scheduler.py）

### 3.1 初始化

```python
class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ):
        # 核心配置
        self.max_num_seqs = scheduler_config.max_num_seqs
        self.max_num_batched_tokens = scheduler_config.max_num_batched_tokens

        # 请求队列
        self.waiting_queue = self._create_request_queue(
            scheduler_config.scheduling_policy
        )
        # running 请求集合
        self.running_requests: dict[str, Request] = {}

        # KV Cache 管理器
        self.kv_cache_manager = KVCacheManager(...)

        # Encoder cache 管理（用于 vision model 等）
        self.encoder_cache_manager = EncoderCacheManager(...)
```

**核心配置参数：**

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `max_num_seqs` | 每轮最大并发请求数 | 256 |
| `max_num_batched_tokens` | 每轮最大 token 总数 | 2048 / 8192 |
| `scheduling_policy` | 调度策略（fcfs / priority） | "fcfs" |
| `enable_chunked_prefill` | 是否启用分块预填充 | True（v1 默认） |

### 3.2 schedule() 方法：核心调度流程

`schedule()` 是调度器最核心的方法，在每个 iteration 调用一次。以下是完整流程的伪代码解析：

```python
def schedule(self) -> SchedulerOutput:
    # ===== 阶段 0: 预算初始化 =====
    budget = SchedulingBudget(
        token_budget=self.max_num_batched_tokens,
        max_num_requests=self.max_num_seqs,
    )

    # ===== 阶段 1: 处理新到达的请求 =====
    # 将 engine 传来的新请求加入 waiting_queue
    self._process_new_arrivals()

    # ===== 阶段 2: 处理 running 队列（decode 阶段的请求） =====
    # 遍历当前正在运行的请求
    requests_to_preempt = []
    scheduled_running = []

    for req in self.running_requests.values():
        # 检查请求是否已完成
        if req.is_finished():
            self._free_request(req)
            continue

        # 计算本轮需要的 token 数
        # 对于 decode: num_tokens = 1
        # 对于 chunked prefill 的后续 chunk: num_tokens = chunk_size
        num_tokens = req.num_tokens_to_schedule()

        # 检查预算是否允许
        if not budget.can_schedule(num_tokens):
            # 预算不足，需要 preempt
            requests_to_preempt.append(req)
            continue

        # 尝试为新 token 分配 KV Cache blocks
        alloc_status = self.kv_cache_manager.allocate_slots(
            req, num_tokens
        )

        if alloc_status == AllocationStatus.OK:
            budget.consume(num_tokens)
            scheduled_running.append(req)
        elif alloc_status == AllocationStatus.NO_FREE_BLOCKS:
            # 显存不足，触发 preemption
            requests_to_preempt.append(req)

    # 执行 preemption（按优先级从低到高 preempt）
    for req in reversed(sorted(requests_to_preempt, key=priority)):
        self._preempt_request(req)
        self.waiting_queue.add(req)  # 被抢占的请求重新进入等待队列

    # ===== 阶段 3: 处理 waiting 队列（新请求的 prefill） =====
    scheduled_new = []

    for req in self.waiting_queue:
        # 计算 prefill 需要的 token 数
        num_tokens = self._get_num_prefill_tokens(req)

        # chunked prefill: 如果 prompt 太长，只取一个 chunk
        if self.enable_chunked_prefill:
            num_tokens = min(
                num_tokens,
                budget.remaining_token_budget()
            )
            if num_tokens == 0:
                break  # 预算耗尽

        # 检查预算
        if not budget.can_schedule(num_tokens, num_new_seqs=1):
            if not self.enable_chunked_prefill:
                break  # 非 chunked 模式下，一个请求放不下就停止
            continue  # chunked 模式下，跳过这个太大的请求，尝试下一个

        # 尝试分配 KV Cache blocks
        alloc_status = self.kv_cache_manager.allocate_slots(
            req, num_tokens
        )

        if alloc_status == AllocationStatus.OK:
            budget.consume(num_tokens, num_new_seqs=1)
            self.waiting_queue.remove(req.request_id)
            self.running_requests[req.request_id] = req
            scheduled_new.append(req)
        else:
            break  # KV Cache 不足，停止调度新请求

    # ===== 阶段 4: 构建输出 =====
    return self._build_scheduler_output(
        scheduled_running=scheduled_running,
        scheduled_new=scheduled_new,
    )
```

### 3.3 调度决策流程图

```
schedule() 被调用
    │
    ▼
┌─────────────────────────┐
│ 初始化 budget            │
│ token_budget = max_tokens│
│ seq_budget = max_seqs    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 处理 running 请求        │
│ for each running req:   │
│   - 已完成？→ 释放       │
│   - budget 够？→ 加入    │
│   - 不够？→ preempt 候选 │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 执行 preemption          │
│ 按优先级从低到高 preempt │
│ 释放 KV blocks           │
│ 请求放回 waiting queue   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 处理 waiting 请求        │
│ for each waiting req:   │
│   - 计算 prefill tokens  │
│   - chunked? 限制 size   │
│   - budget 够？→ 分配    │
│   - 不够？→ 停止/跳过    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 构建 SchedulerOutput     │
│ 返回给 Model Runner      │
└─────────────────────────┘
```

## 4. Budget 控制机制

### 4.1 SchedulingBudget

Budget 是 vLLM 调度器避免超载的关键机制：

```python
class SchedulingBudget:
    def __init__(self, token_budget: int, max_num_requests: int):
        self.token_budget = token_budget
        self.max_num_requests = max_num_requests
        self._num_tokens_scheduled = 0
        self._num_requests_scheduled = 0

    def can_schedule(self, num_tokens: int, num_new_seqs: int = 0) -> bool:
        return (
            self._num_tokens_scheduled + num_tokens <= self.token_budget
            and self._num_requests_scheduled + num_new_seqs <= self.max_num_requests
        )

    def consume(self, num_tokens: int, num_new_seqs: int = 0):
        self._num_tokens_scheduled += num_tokens
        self._num_requests_scheduled += num_new_seqs

    def remaining_token_budget(self) -> int:
        return self.token_budget - self._num_tokens_scheduled
```

### 4.2 Budget 参数的选择

**`max_num_batched_tokens` 的影响：**

```
较小的值（如 2048）：
  + 每个 iteration 耗时短 → TBT 低
  + 内存使用可预测
  - 吞吐量受限（GPU 可能闲置）
  - prefill 需要更多 chunk → TTFT 增加

较大的值（如 8192 或 16384）：
  + 吞吐量更高（GPU 利用率高）
  + 长 prompt 可以一次性 prefill
  - 每个 iteration 耗时长 → TBT 波动
  - 显存峰值高
```

**`max_num_seqs` 的影响：**

```
较小的值（如 32）：
  + 调度开销低
  + 每个请求获得更多显存
  - 并发能力受限

较大的值（如 256 或 512）：
  + 高并发能力
  + 更高的 GPU 利用率（更多请求参与 batch）
  - 调度开销增加
  - 每个请求可用的 KV Cache 显存减少
```

## 5. 与 KVCacheManager 的交互

调度器和 KVCacheManager 之间存在紧密的协作关系：

### 5.1 交互接口

```python
class KVCacheManager:
    def allocate_slots(
        self, request: Request, num_tokens: int
    ) -> AllocationStatus:
        """
        为请求分配 KV Cache slots。
        返回分配状态：OK, NO_FREE_BLOCKS, NEVER_ALLOCATABLE
        """
        ...

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[list[int], int]:
        """
        检查是否有 prefix cache hit。
        返回 (cached_block_ids, num_cached_tokens)
        """
        ...

    def free(self, request: Request) -> None:
        """释放请求的所有 KV Cache blocks"""
        ...

    def get_num_free_blocks(self) -> int:
        """返回当前空闲 block 数量"""
        ...
```

### 5.2 调度器中的使用模式

```python
# 在调度 waiting 请求时：
def _schedule_waiting_request(self, req: Request, budget: SchedulingBudget):
    # 1. 先检查 prefix cache
    cached_blocks, num_cached_tokens = (
        self.kv_cache_manager.get_computed_blocks(req)
    )

    # 2. 计算需要 prefill 的 token 数（减去已缓存的部分）
    num_tokens_to_prefill = req.num_prompt_tokens - num_cached_tokens

    # 3. 尝试分配
    status = self.kv_cache_manager.allocate_slots(
        req, num_tokens_to_prefill
    )

    if status == AllocationStatus.OK:
        # 分配成功，加入 scheduled
        budget.consume(num_tokens_to_prefill, num_new_seqs=1)
        return True
    elif status == AllocationStatus.NEVER_ALLOCATABLE:
        # 请求需要的 blocks 超过物理总量，永远无法调度
        self._abort_request(req, "Request too large")
        return False
    else:
        # NO_FREE_BLOCKS：当前没有空闲 block
        return False
```

### 5.3 Preemption 触发

当 running 请求无法获得足够的 KV Cache blocks 时（例如 decode 阶段需要新 block 但已满），调度器需要 preempt 其他请求来释放显存：

```python
def _handle_preemption(self, victim_req: Request):
    """处理被抢占的请求"""
    # 1. 释放 victim 的 KV Cache blocks
    self.kv_cache_manager.free(victim_req)

    # 2. 将 victim 放回 waiting queue
    victim_req.status = RequestStatus.PREEMPTED
    self.waiting_queue.add(victim_req)

    # 注意：被 preempt 的请求下次调度时需要重新 prefill
    # 这就是 preemption 的代价——丢失了已计算的 KV Cache
```

## 6. 请求处理的完整生命周期

把所有环节串联起来，一个请求在 vLLM v1 中的完整生命周期：

```
1. API 请求到达
   └→ Engine 创建 Request 对象
       └→ 加入 Scheduler 的 waiting_queue

2. Scheduler.schedule() 被调用
   └→ 遍历 waiting_queue
       └→ 为请求分配 KV Cache blocks
           └→ 请求进入 running_requests
               └→ 纳入 SchedulerOutput

3. Model Runner 执行
   └→ 处理 SchedulerOutput 中的请求
       └→ prefill: 处理 prompt tokens (或 chunk)
       └→ decode: 生成下一个 token

4. 结果返回
   └→ 检查是否完成（EOS / max_length）
       ├→ 完成: Scheduler 标记 FINISHED，释放 KV blocks
       └→ 未完成: 继续留在 running_requests

5. 下一个 iteration
   └→ 回到步骤 2
```

## 7. v0 vs v1 调度器的关键差异

| 特性 | v0 Scheduler | v1 Scheduler |
|------|-------------|-------------|
| 架构 | 单文件 `scheduler.py` | 模块化 `sched/` 目录 |
| Swap 队列 | 有 swapped queue | 移除了 swapped queue |
| Chunked Prefill | 可选功能 | 默认启用 |
| Prefix Caching | 与调度松耦合 | 深度集成 |
| 异步调度 | 不支持 | 支持（DBO） |
| 请求队列 | 固定 FCFS | 可插拔策略 |
| 性能 | 调度本身有较高 CPU 开销 | 优化了调度路径 |

v1 移除 swapped queue 的原因：在实践中，swap 到 CPU 内存的方式开销较大（PCIe 带宽受限），recomputation 往往更高效。v1 简化了这个路径，改为直接 preempt + recompute。

## 8. 调试与观测

vLLM 提供了丰富的调度器指标，可以通过 Prometheus metrics 观测：

```python
# 关键指标
vllm:num_requests_running      # 当前 running 的请求数
vllm:num_requests_waiting      # 当前 waiting 的请求数
vllm:num_preemptions_total     # 累计 preemption 次数
vllm:gpu_cache_usage_perc      # GPU KV Cache 使用率
vllm:num_batched_tokens        # 每轮实际 batch 的 token 数
```

当你观察到以下现象时，说明调度器配置需要调优：

| 现象 | 可能原因 | 调优方向 |
|------|---------|---------|
| `num_requests_waiting` 持续很高 | batch 太小或 KV Cache 不足 | 增大 `max_num_seqs` 或减小 `max_model_len` |
| `num_preemptions_total` 快速增长 | 显存不足 | 减小 `max_num_seqs` 或减小 `max_model_len` |
| `gpu_cache_usage_perc` 长期 < 50% | batch 太小 | 增大 `max_num_seqs` 和 `max_num_batched_tokens` |
| `num_batched_tokens` 波动很大 | 混合长短请求 | 启用 chunked prefill，控制 chunk size |

## 9. 参考资料

- [vLLM v1 Source Code: vllm/v1/core/sched/](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched)
- [vLLM Architecture Documentation](https://docs.vllm.ai/en/latest/design/v1/v1_architecture.html)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
