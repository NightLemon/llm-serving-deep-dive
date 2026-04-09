# Ch08: 调度与批处理

> 前置知识：Ch04 PagedAttention、[gpu-ai-systems-learning/07/03-continuous-batching.md](https://github.com/NightLemon/gpu-ai-systems-learning/blob/master/07-inference-optimization/03-continuous-batching.md)

## 🎯 学习目标

- 深入理解 Continuous Batching 的实现细节，而不仅仅是概念
- 走读 vLLM v1 Scheduler 核心源码
- 理解 Chunked Prefill 的实现与权衡
- 掌握 SLA-aware 调度策略
- 能够根据工作负载特征选择和调优调度参数

## 📑 内容大纲

### 1. Continuous Batching 深入（01-continuous-batching.md）

**从概念到实现：**
- Iteration-level scheduling（Orca 论文）的核心思想
- 每个 decode step 都可以加入新请求 / 移除已完成请求
- 与 static batching 的吞吐量对比公式推导
- "padding 浪费" 的精确量化

**Micro-batching 策略：**
- 一个 iteration 内如何组织 prefill 和 decode 请求
- Prefill 请求的插入时机：是否与 decode 混合执行？
- piggybacking：将 prefill 的最后一个 token 视为 decode token

### 2. vLLM Scheduler 源码走读（02-vllm-scheduler.md）

**核心文件：**
- `vllm/v1/core/sched/scheduler.py` — 主调度器
  - `schedule()` 方法的完整流程
  - Running queue / Waiting queue / Swapped queue 管理
  - Budget 控制：每轮最大 token 数和最大请求数
  
- `vllm/v1/core/sched/request_queue.py` — 请求队列管理
  - FCFS（先来先服务）排序
  - 优先级队列支持

- `vllm/v1/core/sched/output.py` — 调度输出
  - `SchedulerOutput`：传递给 model runner 的指令

**调度决策流程：**
```
schedule() {
    1. 预算初始化（max_num_batched_tokens, max_num_seqs）
    2. 处理 running 队列中的请求（decode 阶段）
       - 检查每个请求是否有足够的 KV block
       - 显存不足 → 触发 preemption
    3. 处理 waiting 队列中的新请求（prefill 阶段）
       - 按优先级/到达时间排序
       - 分配 KV blocks
       - 预算耗尽 → 停止处理
    4. 返回 SchedulerOutput
}
```

### 3. Chunked Prefill（03-chunked-prefill.md）

**问题：长 prompt 的 prefill 会阻塞 decode 请求**

**解决方案：**
- 将长 prompt 切成固定大小的 chunk（如 512 tokens）
- 每次 iteration 只处理一个 chunk
- 其余 chunk 在后续 iteration 中处理
- 期间 decode 请求可以正常执行

**实现细节：**
- `--enable-chunked-prefill` / `--max-num-batched-tokens`
- Chunk 大小对 TTFT 和 TBT 的影响
- 源码：Scheduler 如何跟踪 partial prefill 的状态

**权衡：**
- Chunk 太大 → decode 延迟增加
- Chunk 太小 → prefill 效率降低（GPU 利用率下降）
- 最优 chunk size 取决于硬件和工作负载

### 4. 请求优先级与公平性（04-priority-fairness.md）

**优先级调度：**
- vLLM 的 `priority` 参数
- 如何实现：高优先级请求可以抢占低优先级请求的 KV Cache
- 使用场景：付费用户 vs 免费用户

**公平性保障：**
- 避免 "饥饿"：低优先级请求等待过久
- Aging 机制：等待时间越长，优先级逐步提升
- Token 级公平 vs 请求级公平

**SLA-aware 调度：**
- TTFT SLA：控制首 token 延迟
- TBT SLA：控制每 token 间隔
- 如何在吞吐量和延迟之间取得平衡

### 5. Dual Batch Overlap (DBO)（05-dbo.md）

**概念：**
- 论文/设计文档：vLLM DBO
- 将 GPU 执行和 CPU 调度 overlap
- 当前 batch 在 GPU 执行时，CPU 同时计算下一个 batch 的调度方案
- 减少调度的 CPU overhead 对吞吐量的影响

**源码走读：**
- `vllm/v1/core/sched/async_scheduler.py` — 异步调度器

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Orca: A Distributed Serving System](https://www.usenix.org/conference/osdi22/presentation/yu) | 2022 | Iteration-level scheduling |
| [Sarathi-Serve: Chunked Prefills for Efficient LLM Serving](https://arxiv.org/abs/2403.02310) | 2024 | Chunked prefill 系统化 |
| [FastServe: Fast Preemption-based Serving for Large Language Models](https://arxiv.org/abs/2305.05920) | 2023 | Preemption-based 调度 |
| [S3: Increasing GPU Utilization during Generative Inference for Higher Throughput](https://arxiv.org/abs/2306.06000) | 2023 | Split-Fuse 策略 |

## 📁 文件清单

- [x] `01-continuous-batching.md` — Continuous Batching 深入
- [x] `02-vllm-scheduler.md` — vLLM Scheduler 源码走读
- [x] `03-chunked-prefill.md` — Chunked Prefill
- [x] `04-priority-fairness.md` — 优先级与公平性
- [x] `05-dbo.md` — Dual Batch Overlap
- [x] `exercises.md` — 动手练习（调度参数调优实验）
