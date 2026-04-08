# Dual Batch Overlap (DBO)

> 将 GPU 执行和 CPU 调度 overlap——消除调度的 CPU overhead，最大化 GPU 利用率

## 1. 问题：CPU 调度开销

### 1.1 同步调度的瓶颈

在传统的同步调度模式下，每个 iteration 的执行流程是：

```
时间 →
CPU: [schedule()]         [schedule()]         [schedule()]
GPU:              [exec()]            [exec()]            [exec()]
     ↑ CPU 调度    ↑ GPU 执行  ↑ CPU 调度    ↑ GPU 执行
```

CPU 和 GPU 是**串行**执行的：
1. CPU 执行 `schedule()` → 确定本轮要处理哪些请求
2. GPU 执行 model forward pass → 处理请求
3. 等 GPU 完成后，CPU 再执行下一轮 `schedule()`

**GPU 空闲时间 = CPU 调度耗时。**

### 1.2 调度开销的构成

`schedule()` 方法的 CPU 耗时由以下部分组成：

```
schedule() 总耗时分解:

1. 遍历 running 请求                    ~50-200μs
   - 检查每个请求是否完成
   - 标记已完成请求

2. KV Cache 管理                        ~100-500μs
   - allocate_slots() 分配新 blocks
   - get_computed_blocks() 检查 prefix cache
   - free() 释放已完成请求的 blocks
   - Block table 更新

3. 遍历 waiting 请求                    ~50-300μs
   - 按优先级排序
   - 检查预算约束
   - Chunked prefill 计算 chunk size

4. 构建 SchedulerOutput                 ~50-200μs
   - 序列化调度决策
   - 准备传给 model runner 的元数据

5. 请求状态更新                          ~20-100μs
   - 更新请求状态机
   - 触发回调通知

总计: ~300μs - 1.5ms (取决于请求数量和复杂度)
```

### 1.3 CPU 开销的相对影响

```
GPU iteration 执行时间 (典型值):
  Decode only (B=64, Llama-2-13B, A100):    ~15ms
  Mixed batch (B=128, budget=2048):          ~25ms
  Large prefill chunk (2048 tokens):         ~30ms

CPU 调度时间:
  少量请求 (B < 32):                        ~0.3ms → 2% overhead
  中等请求 (B = 128):                       ~0.8ms → 3-5% overhead
  大量请求 (B = 512):                       ~1.5ms → 5-10% overhead
  大量请求 + prefix cache 查询:              ~2.0ms → 7-13% overhead
```

当请求数量较大、启用 prefix caching 等复杂功能时，CPU 调度开销可以占到 iteration 总时间的 5-13%。这意味着 GPU 有 5-13% 的时间在等待 CPU 完成调度而处于空闲状态。

**关键洞察**：在高吞吐场景下，每 1% 的 GPU 利用率提升都意味着可观的成本节约（A100/H100 的租用成本为 $1-3/h/GPU）。

## 2. 解决方案：Dual Batch Overlap (DBO)

### 2.1 核心思想

DBO 的核心思想是将 **CPU 调度和 GPU 执行 overlap（重叠执行）**：

- 当 GPU 正在执行 iteration N 时
- CPU 同时计算 iteration N+1 的调度方案

```
Without DBO (同步调度):
CPU: [sched_1]         [sched_2]         [sched_3]
GPU:           [exec_1]         [exec_2]         [exec_3]
Total:   |←─ sched + exec ─→|←─ sched + exec ─→|

With DBO (异步调度):
CPU: [sched_1][sched_2]         [sched_3]         [sched_4]
GPU:          [exec_1]          [exec_2]          [exec_3]
               ↑ sched_2 与 exec_1 overlap
Total:   |←─ max(sched, exec) ─→|
```

**效果**：iteration 间隔从 `t_sched + t_exec` 降低为 `max(t_sched, t_exec)`。由于 `t_exec >> t_sched`（GPU 执行远慢于 CPU 调度），实际效果接近完全消除调度开销。

### 2.2 "Dual Batch" 的含义

"Dual Batch" 指的是系统中同时存在两个 batch：

```
Batch N  : 当前正在 GPU 上执行的 batch
Batch N+1: CPU 正在计算调度方案的下一个 batch

两个 batch 在时间上重叠（overlap）
→ GPU 几乎零空闲时间
```

这类似于 CPU 流水线中的 **指令预取（Instruction Prefetch）** 思想——在当前指令执行时，提前取出下一条指令。

## 3. 异步调度器源码走读

### 3.1 架构概览

vLLM v1 的异步调度器位于 `vllm/v1/core/sched/async_scheduler.py`：

```
┌────────────────────────────────────┐
│        AsyncScheduler              │
│                                    │
│  ┌──────────┐   ┌──────────────┐   │
│  │ Scheduler │   │ Background   │   │
│  │ (同步)    │   │ Thread       │   │
│  └──────────┘   └──────────────┘   │
│       ↓              ↓              │
│  ┌──────────────────────────────┐  │
│  │  Shared State (thread-safe)  │  │
│  │  - pending_output            │  │
│  │  - new_requests_queue        │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
```

### 3.2 核心实现

```python
class AsyncScheduler:
    """
    异步调度器：在后台线程中执行调度，
    与 GPU 执行 overlap。
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

        # 预计算的下一轮调度结果
        self._next_output: Optional[SchedulerOutput] = None

        # 用于通知后台线程的事件
        self._schedule_event = threading.Event()
        self._output_ready_event = threading.Event()

        # 新请求队列（线程安全）
        self._new_requests: queue.Queue[Request] = queue.Queue()

        # GPU 执行完成后的结果（用于更新调度器状态）
        self._execution_results: queue.Queue[ExecutionResult] = queue.Queue()

        # 启动后台调度线程
        self._bg_thread = threading.Thread(
            target=self._background_schedule_loop,
            daemon=True,
        )
        self._bg_thread.start()

    def _background_schedule_loop(self):
        """后台调度线程的主循环"""
        while True:
            # 等待信号：上一轮 GPU 执行开始后触发
            self._schedule_event.wait()
            self._schedule_event.clear()

            # 1. 处理上一轮 GPU 执行的结果
            #    （更新请求状态、释放完成请求的 KV blocks 等）
            while not self._execution_results.empty():
                result = self._execution_results.get_nowait()
                self.scheduler.process_execution_result(result)

            # 2. 处理新到达的请求
            while not self._new_requests.empty():
                req = self._new_requests.get_nowait()
                self.scheduler.add_request(req)

            # 3. 执行调度
            self._next_output = self.scheduler.schedule()

            # 4. 通知主线程：调度结果已就绪
            self._output_ready_event.set()

    def get_next_schedule(self) -> SchedulerOutput:
        """
        主线程调用：获取下一轮的调度结果。
        如果后台线程还没完成调度，阻塞等待。
        """
        self._output_ready_event.wait()
        self._output_ready_event.clear()
        output = self._next_output
        self._next_output = None
        return output

    def notify_execution_started(self, result: ExecutionResult = None):
        """
        主线程调用：通知后台线程 GPU 执行已开始，
        可以开始计算下一轮调度。
        """
        if result is not None:
            self._execution_results.put(result)
        self._schedule_event.set()

    def add_request(self, request: Request):
        """线程安全地添加新请求"""
        self._new_requests.put(request)
```

### 3.3 主循环中的使用

```python
class EngineCore:
    """引擎核心的主执行循环"""

    def run_engine_loop(self):
        # 第一轮特殊处理：同步调度
        scheduler_output = self.async_scheduler.get_next_schedule()

        while True:
            # 1. 将调度结果发送给 GPU 执行
            self.model_runner.execute_async(scheduler_output)

            # 2. 通知异步调度器：GPU 开始执行了
            #    后台线程立即开始计算下一轮调度
            self.async_scheduler.notify_execution_started(
                result=self.last_execution_result
            )

            # 3. 等待 GPU 执行完成
            execution_result = self.model_runner.wait_for_result()
            self.last_execution_result = execution_result

            # 4. 获取下一轮调度结果
            #    如果后台线程已完成 → 立即返回
            #    如果后台线程未完成 → 等待（极少发生，因为 GPU exec >> CPU sched）
            scheduler_output = self.async_scheduler.get_next_schedule()

            # 5. 处理输出（流式返回、完成通知等）
            self.process_outputs(execution_result)
```

**时序图：**

```
主线程:  [send_to_gpu][notify]      [wait_gpu] [get_schedule][send_to_gpu][notify]
后台线程:              [─ schedule ─]                          [─ schedule ─]
GPU:     [──── execute ────]                   [──── execute ────]

         ↑ schedule 和 execute overlap
```

## 4. DBO 的正确性保证

### 4.1 核心挑战：信息过时

DBO 面临一个关键问题：**后台线程在计算 iteration N+1 的调度时，iteration N 还在执行，iteration N 的结果尚未可知。**

这意味着：
- 不知道 iteration N 中哪些请求完成了（生成了 EOS）
- 不知道 iteration N 中哪些请求的 output token 是什么
- 不知道 KV Cache 的最新使用情况

### 4.2 解决方案：保守调度 + 事后修正

**保守调度策略：**

```python
def schedule_for_dbo(self):
    """
    DBO 模式下的调度策略：
    假设所有 running 请求在上一轮都没有完成。
    """
    # 假设所有 running 请求继续运行
    # 保守估计 KV Cache 使用量
    # 预留足够的 buffer

    # 如果某个请求实际上在 iteration N 中完成了，
    # iteration N+1 执行时会发现它已完成，跳过它
    # 释放的 KV blocks 在 iteration N+2 的调度中才能被使用
```

**事后修正：**

```python
def process_execution_result(self, result: ExecutionResult):
    """
    处理 GPU 执行结果，修正调度状态。
    """
    # 1. 标记已完成的请求
    for req_id in result.finished_request_ids:
        req = self.running_requests.pop(req_id)
        self.kv_cache_manager.free(req)

    # 2. 更新 token 计数
    for req_id, num_generated in result.generated_tokens.items():
        self.running_requests[req_id].num_computed_tokens += num_generated

    # 下一轮 schedule() 会基于修正后的状态执行
```

### 4.3 资源利用率的一个 iteration 延迟

DBO 的一个副作用是资源释放有一个 iteration 的延迟：

```
Without DBO:
  Iteration N:   Req A 完成 → 立即释放 KV blocks
  Iteration N+1: Req B 可以使用 Req A 释放的 blocks

With DBO:
  Iteration N:   Req A 完成
  Iteration N+1: 调度是在 iteration N 执行期间计算的，
                 不知道 Req A 已完成 → 无法使用其 blocks
  Iteration N+2: 才能使用 Req A 释放的 blocks
```

**影响**：在显存紧张的场景下，这一个 iteration 的延迟可能导致：
- 多一次 preemption（因为调度器高估了 KV Cache 使用量）
- 新请求延迟一个 iteration 才能被调度

**缓解措施**：
- 适当增加 KV Cache 的 over-provisioning（预留 5-10% buffer）
- 对于延迟极其敏感的场景，可以禁用 DBO

## 5. DBO 的性能收益分析

### 5.1 理论收益

```
同步调度的 iteration 间隔:
  T_sync = T_sched + T_exec

DBO 的 iteration 间隔:
  T_dbo = max(T_sched, T_exec)

收益:
  Speedup = T_sync / T_dbo
          = (T_sched + T_exec) / max(T_sched, T_exec)

当 T_exec >> T_sched 时:
  Speedup ≈ (T_sched + T_exec) / T_exec
          = 1 + T_sched / T_exec

典型值:
  T_sched = 0.8ms, T_exec = 15ms
  Speedup = 1 + 0.8/15 = 1.053 → 5.3% 提升

  T_sched = 1.5ms, T_exec = 15ms
  Speedup = 1 + 1.5/15 = 1.10 → 10% 提升

  T_sched = 2.0ms, T_exec = 25ms
  Speedup = 1 + 2.0/25 = 1.08 → 8% 提升
```

### 5.2 实际收益场景分析

```
场景 1: 小模型 + 高并发
  模型: Llama-3-8B on A100
  并发: 512 请求
  T_exec ≈ 12ms, T_sched ≈ 1.5ms
  DBO 收益: ~12.5%

场景 2: 大模型 + 中等并发
  模型: Llama-3-70B on 4xA100 (TP=4)
  并发: 128 请求
  T_exec ≈ 35ms, T_sched ≈ 0.8ms
  DBO 收益: ~2.3%

场景 3: 中等模型 + prefix caching
  模型: Llama-3-13B on A100
  并发: 256 请求, prefix caching 开启
  T_exec ≈ 20ms, T_sched ≈ 2.0ms (cache 查询耗时)
  DBO 收益: ~10%
```

**结论**：DBO 的收益与 `T_sched / T_exec` 比例成正比。以下场景收益最大：
- 小模型（T_exec 小 → ratio 大）
- 高并发（T_sched 大 → ratio 大）
- 启用复杂调度策略（prefix caching、priority scheduling）

### 5.3 DBO 的适用条件

| 条件 | 适合 DBO | 不适合 DBO |
|------|---------|-----------|
| 并发量 | > 128 请求 | < 32 请求 |
| 调度复杂度 | 高（prefix cache、priority） | 低（纯 FCFS） |
| 显存余量 | 充足（> 20% free） | 紧张（< 10% free） |
| 延迟要求 | 可容忍 1 iteration 延迟 | 极低延迟要求 |

## 6. 与其他 Overlap 优化的对比

### 6.1 CUDA Stream Overlap

GPU 内部的计算和通信 overlap，与 DBO 是不同层级的优化：

```
CUDA Stream Overlap (GPU 内部):
  Stream 1 (Compute):  [Layer 1 GEMM] [Layer 2 GEMM] ...
  Stream 2 (Transfer): [KV Cache copy] [Result copy] ...
  → GPU 内计算和数据传输 overlap

DBO (CPU-GPU):
  CPU:  [Scheduling]
  GPU:  [Model Execution]
  → CPU 和 GPU overlap
```

两者可以同时启用，互不冲突。

### 6.2 Pipeline Parallelism 的 Bubble

在 Pipeline Parallelism 中也存在类似的 overlap 问题：

```
PP without overlap:
  Stage 0: [micro1][idle]  [micro2][idle]
  Stage 1: [idle]  [micro1][idle]  [micro2]

PP with overlap:
  Stage 0: [micro1][micro2][micro3]...
  Stage 1: [micro1][micro2][micro3]...
            ↑ micro-batches fill the pipeline
```

DBO 和 PP overlap 可以叠加：

```
With both DBO + PP overlap:
  CPU:     [sched_1][sched_2]...
  GPU S0:  [micro1_1][micro1_2][micro2_1]...
  GPU S1:  [micro1_1][micro1_2][micro2_1]...
  → 调度和流水线执行同时 overlap
```

## 7. 实现注意事项

### 7.1 线程安全

DBO 引入了多线程，需要特别注意数据竞争：

```python
# 需要线程安全保护的共享状态:
# 1. 请求队列（新请求的添加）
# 2. KV Cache Manager 的状态（block 分配/释放）
# 3. 请求状态（running → finished）

# vLLM 的策略：
# - 新请求通过 thread-safe queue 传递
# - KV Cache 状态只在后台线程中修改（通过 execution_result 同步）
# - 调度器核心状态只在后台线程中访问
```

### 7.2 错误处理

```python
def _background_schedule_loop(self):
    while True:
        try:
            self._schedule_event.wait()
            self._schedule_event.clear()

            # ... 调度逻辑 ...

            self._output_ready_event.set()
        except Exception as e:
            # 调度异常不能让后台线程崩溃
            logger.error(f"Scheduling error: {e}")
            # 生成一个空的 SchedulerOutput 作为 fallback
            self._next_output = SchedulerOutput.empty()
            self._output_ready_event.set()
```

### 7.3 Graceful Shutdown

```python
def shutdown(self):
    """优雅关闭异步调度器"""
    self._shutdown_event.set()
    self._schedule_event.set()  # 唤醒后台线程
    self._bg_thread.join(timeout=5.0)
```

## 8. 配置与启用

```bash
# vLLM v1 中 DBO 的启用
# 在 v1 架构中，异步调度是默认行为的一部分

# 如果需要调试调度问题，可以回退到同步模式
vllm serve model_name --disable-async-output-proc

# 查看调度耗时指标
# Prometheus metrics:
#   vllm:scheduler_time_seconds  - 每轮调度耗时
#   vllm:gpu_execute_time_seconds - 每轮 GPU 执行耗时
```

## 9. 参考资料

- [vLLM v1 Architecture: Async Scheduling](https://github.com/vllm-project/vllm/tree/main/vllm/v1/core/sched)
- [vLLM Design Document: Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/v1/v1_architecture.html)
- [CUDA Streams and Concurrency](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
