# Ch04: PagedAttention 与内存管理

> 前置知识：Ch01 KV Cache 深度剖析、[gpu-ai-systems-learning/07/06-vllm-architecture.md](../../gpu-ai-systems-learning/07-inference-optimization/06-vllm-architecture.md)

## 🎯 学习目标

- 理解 PagedAttention 从操作系统虚拟内存到 KV Cache 管理的类比与差异
- 走读 vLLM v1 的 BlockPool / KVCacheManager 核心源码
- 理解 Block 的生命周期：分配 → 使用 → 共享 → 驱逐
- 掌握 Preemption 策略（swap vs recomputation）的实现与权衡
- 理解内存碎片在长时间运行后的表现及缓解方案

## 📑 内容大纲

### 1. PagedAttention 原理深入（01-paged-attention.md）

**论文精读：**
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- 类比操作系统：Virtual Address → Physical Block, Page Table → Block Table
- 为什么 LLM 推理的显存管理问题类似 OS？—— 动态增长、不可预知长度
- 物理 Block 大小的选择：16 tokens/block 的来源与影响
- Copy-on-Write：beam search 场景下的 KV 共享

**与传统方案的对比：**
- 传统连续分配：预分配 `max_seq_len` 的显存，浪费率 60-80%
- PagedAttention：按需分配 block，浪费仅 < 4%（最后一个 block 的内部碎片）

### 2. vLLM v1 内存管理源码走读（02-vllm-memory.md）

**核心文件：**
- `vllm/v1/core/block_pool.py` — 物理 block 池管理
  - `BlockPool.__init__`：初始化所有物理 block
  - `allocate` / `free`：block 的分配与回收
  - 引用计数机制：多个 request 共享同一 block 时的管理
  - Free list 实现：如何快速找到空闲 block

- `vllm/v1/core/kv_cache_manager.py` — KV Cache 调度
  - `allocate_slots`：为新 token 分配 KV 存储
  - `get_computed_blocks`：prefix caching 命中判定
  - `free`：请求结束后释放 block
  - 与 Scheduler 的交互接口

- `vllm/v1/core/kv_cache_coordinator.py` — 多层 KV Cache 协调
  - 不同 layer group 可能有不同的 cache 配置
  - Hybrid KV Cache 场景的协调逻辑

- `vllm/v1/worker/block_table.py` — Block Table 的 GPU 端表示
  - `BlockTable` 类：CPU 侧的逻辑映射
  - 如何将 block table 传递给 GPU kernel

### 3. Preemption 策略（03-preemption.md）

**当显存不足时怎么办？**

- **Swap（换出到 CPU）：**
  - 将被抢占请求的 KV Cache 从 GPU 换出到 CPU 内存
  - 恢复时再换入
  - 优点：恢复快（无需重新计算）
  - 缺点：需要 CPU 内存、PCIe 带宽成为瓶颈

- **Recomputation（重新计算）：**
  - 直接丢弃被抢占请求的 KV Cache
  - 恢复时重新 prefill
  - 优点：不需要额外 CPU 内存
  - 缺点：需要额外计算

**源码走读：**
- vLLM Scheduler 中的 preemption 触发条件
- `preemption_mode` 配置项

**何时选择 swap vs recompute？**
- 短 prompt + 长生成 → swap 更优（KV 大、重算贵）
- 长 prompt + 短生成 → recompute 也可接受
- 可用 CPU 内存不足 → 只能 recompute

### 4. 内存碎片分析（04-fragmentation.md）

**内部碎片：**
- 最后一个 block 可能未填满（block_size=16 但只用了 3 个 slot）
- 碎片率 = 平均浪费 / block_size ≈ 50% * (block_size - 1) / avg_seq_len

**外部碎片：**
- PagedAttention 基本消除了外部碎片
- 但 block_size 的选择影响内部碎片 vs kernel 效率的权衡

**长期运行问题：**
- GPU memory allocator 的碎片化（cuMem）
- vLLM 的 `--enforce-eager` 与 CUDA Graph 对内存布局的影响

## 📄 参考论文

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) | 2023 | 分页式 KV Cache 管理 |
| [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) | 2022 | Iteration-level scheduling |
| [FlexGen: High-Throughput Generative Inference with a Single GPU](https://arxiv.org/abs/2303.06865) | 2023 | Offloading + 压缩策略 |

## 📁 文件清单

- [ ] `01-paged-attention.md` — PagedAttention 原理深入
- [ ] `02-vllm-memory.md` — vLLM v1 内存管理源码走读
- [ ] `03-preemption.md` — Preemption 策略
- [ ] `04-fragmentation.md` — 内存碎片分析
- [ ] `exercises.md` — 动手练习
