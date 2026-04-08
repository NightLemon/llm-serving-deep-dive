# 内存碎片分析

> PagedAttention 大幅降低了 KV Cache 的显存浪费，但碎片问题并未完全消除。本节深入分析内部碎片、外部碎片、block_size 权衡，以及长期运行中的 GPU memory allocator 碎片化问题。

## 1. 碎片类型概览

在讨论 PagedAttention 的碎片问题之前，先回顾操作系统中的碎片分类：

```
┌─────────────────────────────────────────────────────────┐
│                   内部碎片 (Internal)                     │
│  分配单元内部的浪费。                                      │
│  例：block_size=16，但最后一个 block 只用了 5 个 slot      │
│  浪费了 11 个 slot 的空间                                 │
│                                                         │
│  [█████░░░░░░░░░░░]  ← 最后一个 block                   │
│   used    wasted                                        │
├─────────────────────────────────────────────────────────┤
│                   外部碎片 (External)                     │
│  空闲空间足够但不连续，无法满足连续分配需求。                  │
│  例：总共有 100MB 空闲，但分散在 50 个 2MB 的碎片中，       │
│  无法分配一个 60MB 的连续块                                │
│                                                         │
│  [██░░██░░██░░██░░]  ← 空闲空间（░）不连续                │
└─────────────────────────────────────────────────────────┘
```

## 2. 内部碎片：最后一个 Block 的浪费

### 2.1 碎片来源

在 PagedAttention 中，每个请求的 KV Cache 占用 `ceil(seq_len / block_size)` 个 block。最后一个 block 通常不会被完全填满：

```
block_size = 16

Request A (seq_len = 42):
  Block 0: [████████████████]  tokens 0-15   (满)
  Block 1: [████████████████]  tokens 16-31  (满)
  Block 2: [██████████░░░░░░]  tokens 32-41  (用 10/16, 浪费 6)

Request B (seq_len = 7):
  Block 0: [███████░░░░░░░░░]  tokens 0-6   (用 7/16, 浪费 9)

Request C (seq_len = 48):
  Block 0: [████████████████]  tokens 0-15   (满)
  Block 1: [████████████████]  tokens 16-31  (满)
  Block 2: [████████████████]  tokens 32-47  (满, 浪费 0!)
```

### 2.2 碎片率公式

对于单个请求，最后一个 block 的浪费 token 数为：

```
waste(seq_len) = block_size - (seq_len mod block_size)
               = block_size - (seq_len % block_size)    当 seq_len % block_size ≠ 0
               = 0                                       当 seq_len % block_size = 0
```

**期望碎片率**（假设 `seq_len mod block_size` 均匀分布）：

```
E[waste] = (block_size - 1) / 2

碎片率 = E[waste] / E[total_slots]
       = (block_size - 1) / 2 / E[ceil(seq_len / block_size) × block_size]
       ≈ (block_size - 1) / (2 × avg_seq_len)
```

### 2.3 不同 block_size 的碎片率

| block_size | E[waste] (tokens) | avg_seq_len=100 | avg_seq_len=500 | avg_seq_len=2000 |
|-----------|-------------------|-----------------|-----------------|------------------|
| 1 | 0 | 0% | 0% | 0% |
| 8 | 3.5 | 3.5% | 0.7% | 0.18% |
| **16** | **7.5** | **7.5%** | **1.5%** | **0.38%** |
| 32 | 15.5 | 15.5% | 3.1% | 0.78% |
| 64 | 31.5 | 31.5% | 6.3% | 1.58% |
| 128 | 63.5 | 63.5% | 12.7% | 3.18% |

**关键观察**：
- 对于平均序列长度 >500 的场景（大多数实际应用），`block_size=16` 的碎片率 **< 2%**
- 与传统连续分配的 60-80% 浪费相比，这是质的飞跃
- 但对于**大量极短序列**（如 embedding 场景），大 block_size 可能造成显著浪费

### 2.4 实际影响量化

以 LLaMA-3-8B 为例（32 layers, 8 KV heads, head_dim=128, FP16）：

```
单个 block 的 KV Cache 大小：
  = 2 (K+V) × 32 × 8 × 128 × 2 bytes × 16 tokens
  = 2 MB

假设同时服务 100 个请求，平均每个请求浪费 7.5 tokens：
  浪费的 block 数 ≈ 100 × 7.5 / 16 ≈ 47 个 "等效 block"
  
  注意：浪费发生在每个请求的最后一个 block 内部，
  实际浪费 = 100 个 block × (7.5/16) ≈ 46.875 个 block 的等效空间
  
总浪费 ≈ 47 × 2 MB = 94 MB（在 80GB 显存中可忽略不计）
```

## 3. 外部碎片：PagedAttention 如何消除

### 3.1 传统连续分配的外部碎片

```
传统方案中，KV Cache 需要连续的显存空间：

显存状态（每格 = 1 个 token 的 KV）:
[A A A A _ _ B B B B B _ _ C C C _ _ _ _ _ _ _ _]

Request D 需要 6 个 token 的连续空间：
  最大连续空闲 = 7 (尾部)
  → 可以分配，但如果 D 需要 10 个呢？
  总空闲 = 12，但最大连续只有 7 → 外部碎片！

随着请求不断到达和离开：
[_ _ A A _ B _ _ C C _ _ D _ _ E _ F _ _ _ G _ _]
  总空闲 = 13，但最大连续 = 3 → 严重外部碎片！
```

### 3.2 PagedAttention 消除外部碎片

PagedAttention 的分页机制使得**任何空闲 block 都可以被任何请求使用**，无需连续：

```
PagedAttention 的 Block 池（每格 = 1 个 block）:
[A] [B] [_] [A] [C] [_] [B] [_] [A] [_] [C] [_]
  0    1   2   3   4   5   6   7   8   9  10  11

空闲 blocks: {2, 5, 7, 9, 11} = 5 个

新请求 D 需要 5 个 block：
  分配 block 2, 5, 7, 9, 11 → 成功！
  D 的 block table: [2, 5, 7, 9, 11]（不需要连续）

→ 只要有足够数量的空闲 block，就一定能分配成功
→ 外部碎片 = 0
```

这就是 PagedAttention 最核心的贡献之一：**通过分页完全消除外部碎片**。

### 3.3 唯一的"碎片"：Block Table 开销

虽然外部碎片被消除，但分页引入了 block table 的存储开销：

```
Block table 大小（per request）：
  = max_num_blocks_per_seq × sizeof(int32)
  = ceil(max_model_len / block_size) × 4 bytes

例：max_model_len=4096, block_size=16:
  = 256 × 4 = 1024 bytes = 1 KB per request

对于 1000 个并发请求：
  = 1 MB

这个开销相比 KV Cache 本身（GB 级别）可以忽略不计。
```

## 4. block_size 的选择：碎片 vs Kernel 效率

### 4.1 矛盾的需求

```
              小 block_size                     大 block_size
            ←──────────────────────────────────────────────→
  
  碎片率      低 ████████░░░░░░░░░░░░ 高
  Kernel效率  低 ░░░░░░░░░░░████████ 高
  管理开销    高 ████████░░░░░░░░░░░░ 低
  Block数量   多 ████████░░░░░░░░░░░░ 少
```

### 4.2 Kernel 效率分析

Attention kernel 的效率与 block_size 密切相关：

**小 block_size 的问题**：
- 每个 block 的计算量少，kernel launch overhead 占比增大
- 内存访问粒度小，无法充分利用 GPU 的内存带宽
- Thread block 内部并行度不足
- 更多的 block table 查找开销

**大 block_size 的优势**：
- 连续内存访问更长，利于 GPU memory coalescing
- 每个 block 的计算量足够大，摊薄 kernel overhead
- 更好的 L2 cache 利用率

**量化对比**（FlashInfer backend，A100，LLaMA-3-8B，seq_len=2048）：

| block_size | Decode latency (ms) | 相对性能 |
|-----------|---------------------|---------|
| 1 | 2.8 | 0.57x |
| 8 | 1.8 | 0.89x |
| 16 | 1.6 | 1.00x (基准) |
| 32 | 1.5 | 1.07x |
| 64 | 1.5 | 1.07x |
| 128 | 1.5 | 1.07x |

观察：从 16 到更大的 block_size，性能收益递减。但从 1 到 16，性能提升显著。

### 4.3 不同 Backend 的偏好

| Backend | 推荐 block_size | 原因 |
|---------|----------------|------|
| PagedAttention V1/V2 | 16 | 原始设计的最佳点 |
| FlashAttention | 64-256 | 大 block 更适合 tile 化计算 |
| FlashInfer | 16-64 | 原生支持 paged layout，16 已经足够高效 |

vLLM 在初始化时会根据选择的 backend 调整 block_size：

```python
# vllm/config.py (简化)
def _get_default_block_size(attention_backend: str) -> int:
    if attention_backend == "FLASH_ATTN":
        return 16  # FlashAttention 在 vLLM 中适配了 paged layout
    elif attention_backend == "FLASHINFER":
        return 16
    else:
        return 16  # 默认值
```

### 4.4 实践建议

| 场景 | 推荐 block_size | 理由 |
|------|----------------|------|
| 通用在线推理 | **16**（默认） | 碎片低、效率好、兼容性佳 |
| 长序列场景 (>8K) | **32-64** | 碎片率在长序列中已经很低，大 block 提升 kernel 效率 |
| 大量短序列 (<100) | **8** | 减少短序列的碎片浪费 |
| Benchmark/低延迟 | **32** | 略微提升 decode kernel 性能 |

## 5. 长期运行：GPU Memory Allocator 碎片化

### 5.1 超越 KV Cache 的碎片问题

上面讨论的是 **KV Cache block** 层面的碎片。但 GPU 还有更底层的内存管理问题——**CUDA memory allocator** 的碎片化。

```
GPU 显存的分层管理：

Layer 3: KV Cache Block Pool (vLLM 管理)
  └── 预分配的大张量，内部通过 block table 索引
  └── 这一层没有外部碎片（PagedAttention 保证）

Layer 2: PyTorch Caching Allocator (torch.cuda.memory)
  └── 管理 CUDA malloc/free
  └── 使用 block pool + 合并策略
  └── 长期运行后可能产生碎片

Layer 1: CUDA Driver (cuMemAlloc / cuMemAddressReserve)
  └── 管理物理 GPU 显存
  └── 虚拟地址映射
  └── 也可能碎片化

Layer 0: GPU Hardware
  └── 物理显存 (HBM2/HBM3)
```

### 5.2 PyTorch Caching Allocator 碎片化

PyTorch 的 CUDA caching allocator 会**缓存已释放的 CUDA 内存**，避免频繁调用 `cudaMalloc/cudaFree`。但长期运行后：

```python
# 碎片化场景
import torch

# 分配一系列不同大小的张量
tensors = []
for i in range(1000):
    size = random.randint(1, 100) * 1024 * 1024  # 1MB - 100MB
    tensors.append(torch.zeros(size, dtype=torch.uint8, device='cuda'))

# 随机释放一半
for i in range(0, 1000, 2):
    del tensors[i]

# 此时 GPU 显存中有大量碎片
# torch.cuda.memory_allocated() 可能只有 25GB
# torch.cuda.memory_reserved() 可能有 50GB
# 差值 25GB 就是 PyTorch allocator 缓存的碎片空间
```

**对 vLLM 的影响**：
- vLLM 在启动时通过 profiling 确定可用于 KV Cache 的显存
- 如果 profiling 阶段的显存布局与运行时不同，可能导致实际可用 block 数量不准确
- 长期运行后，PyTorch allocator 的碎片可能导致 OOM（即使逻辑上有足够空间）

### 5.3 CUDA Virtual Memory (cuMem) API

CUDA 11.0+ 引入了虚拟内存管理 API（`cuMemCreate`、`cuMemMap`），允许应用程序直接管理 GPU 虚拟地址空间：

```c
// 传统 CUDA 内存分配
CUdeviceptr ptr;
cuMemAlloc(&ptr, size);  // 分配连续的物理+虚拟内存

// CUDA Virtual Memory API
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &prop, 0);     // 分配物理内存
CUdeviceptr ptr;
cuMemAddressReserve(&ptr, size, 0, 0, 0); // 预留虚拟地址
cuMemMap(ptr, size, 0, handle, 0);        // 映射物理→虚拟
```

vLLM 和 PyTorch 正在探索使用 cuMem API 来：
1. 预分配连续虚拟地址空间
2. 动态映射/解映射物理页
3. 避免 allocator 级别的碎片化

### 5.4 `torch.cuda.memory.CUDAPluggableAllocator`

PyTorch 支持自定义 CUDA allocator。vLLM 可以通过插入自定义 allocator 来控制显存分配策略：

```python
# PyTorch 自定义 allocator（概念示例）
import torch.cuda.memory

class VllmAllocator:
    def __init__(self):
        self.pool = CUDAMemoryPool()
    
    def malloc(self, size, device, stream):
        return self.pool.allocate(size)
    
    def free(self, ptr, size, device, stream):
        self.pool.deallocate(ptr, size)

# 注册自定义 allocator
torch.cuda.memory.change_current_allocator(VllmAllocator())
```

## 6. `--enforce-eager` 与 CUDA Graph 对内存布局的影响

### 6.1 CUDA Graph 的显存影响

CUDA Graph 通过**录制**一系列 CUDA 操作并作为整体重放来减少 CPU overhead。但它对显存有重要影响：

```
CUDA Graph 录制时：
  1. 录制 decode 步骤的所有 CUDA 操作
  2. 固定所有输入/输出张量的地址
  3. 分配 graph 执行所需的额外显存

显存影响：
  - 每个 graph 需要固定的 workspace 显存
  - 不同 batch size 需要不同的 graph → 多个 graph 消耗更多显存
  - Graph 内部的中间张量不能被释放/复用
```

vLLM 默认对 decode 阶段使用 CUDA Graph：

```python
# vllm/worker/model_runner.py (简化)
class GPUModelRunner:
    def __init__(self, ...):
        if not enforce_eager:
            # 为不同 batch size 预录 CUDA Graph
            self.cuda_graphs = {}
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                graph = self._capture_graph(batch_size)
                self.cuda_graphs[batch_size] = graph
                # 每个 graph 会额外占用一些显存
```

### 6.2 `--enforce-eager` 模式

```bash
# 禁用 CUDA Graph，使用 eager execution
python -m vllm.entrypoints.openai.api_server \
    --enforce-eager
```

**使用场景**：

| 场景 | 是否使用 `--enforce-eager` | 原因 |
|------|--------------------------|------|
| 显存极度紧张 | **是** | CUDA Graph 额外占用几 GB 显存 |
| 调试/开发 | **是** | eager 模式更容易调试 |
| 不支持 CUDA Graph 的操作 | **是** | 某些自定义 op 不兼容 |
| 生产环境 | **否** | CUDA Graph 显著降低延迟 |
| 低并发/大模型 | **可选** | 权衡显存 vs 延迟 |

### 6.3 CUDA Graph 对碎片的影响

```
正常模式（使用 CUDA Graph）：
  ┌──────────────────────────────────────────────┐
  │ Model Weights    │ CUDA Graphs  │ KV Cache   │
  │ ~16 GB           │ ~2-4 GB      │ ~56-60 GB  │
  └──────────────────────────────────────────────┘
  
  CUDA Graph 的显存在初始化时分配，运行期间不会释放。
  这部分显存是"固定"的，不会碎片化。
  但它**减少了可用于 KV Cache 的显存**。

Eager 模式：
  ┌──────────────────────────────────────────────┐
  │ Model Weights    │ KV Cache                  │
  │ ~16 GB           │ ~62-64 GB                 │
  └──────────────────────────────────────────────┘
  
  中间激活在每步 forward 时动态分配/释放。
  PyTorch allocator 需要管理这些动态内存。
  长期运行后，allocator 级别的碎片可能累积。
```

## 7. 实际生产环境中的碎片监控

### 7.1 监控指标

```python
# Python 脚本：监控 GPU 显存碎片状况
import torch

def report_memory_fragmentation():
    # 已分配显存（应用层面正在使用的）
    allocated = torch.cuda.memory_allocated() / 1e9
    
    # 已预留显存（PyTorch allocator 占用的，包含碎片）
    reserved = torch.cuda.memory_reserved() / 1e9
    
    # 总显存
    total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    
    # 碎片指标
    fragmentation = (reserved - allocated) / reserved * 100 if reserved > 0 else 0
    
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")
    print(f"Total:     {total_mem:.2f} GB")
    print(f"Allocator fragmentation: {fragmentation:.1f}%")
    print(f"Free (usable): {total_mem - reserved:.2f} GB")
    print(f"Free (including fragmented): {total_mem - allocated:.2f} GB")
```

### 7.2 vLLM 内置的 KV Cache 指标

```python
# vLLM Prometheus metrics
vllm:gpu_cache_usage_perc        # 已使用的 GPU KV block 占比
vllm:cpu_cache_usage_perc        # 已使用的 CPU swap block 占比
vllm:num_gpu_blocks_total        # GPU block 总数
vllm:num_free_gpu_blocks         # 空闲 GPU block 数

# 监控仪表盘建议
# 1. gpu_cache_usage_perc 的时序图 → 观察峰值和趋势
# 2. num_free_gpu_blocks 的最小值 → 距离 preemption 还有多远
# 3. 碎片率 = 1 - (active_blocks / total_blocks - free_blocks)
#    如果 active_blocks << (total_blocks - free_blocks)，说明有碎片
```

### 7.3 缓解碎片的生产实践

**1. 定期重启**

```bash
# 最简单的碎片缓解方法：定期重启 vLLM 进程
# 配合 load balancer 实现无缝重启
# 建议周期：每 24-72 小时
```

**2. 使用 `PYTORCH_CUDA_ALLOC_CONF`**

```bash
# 调整 PyTorch CUDA allocator 行为
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# expandable_segments (PyTorch 2.1+)
# 允许 allocator 使用可扩展的内存段，减少碎片
# 通过 cuMemMap/cuMemUnmap 动态调整段大小
```

**3. 预分配策略**

vLLM 在启动时会预分配 KV Cache 的大张量：

```python
# vllm/worker/cache_engine.py (简化)
class CacheEngine:
    def __init__(self, num_gpu_blocks, ...):
        # 一次性分配所有 KV Cache 显存
        # 这避免了运行时的 malloc/free 碎片
        self.gpu_cache = [
            torch.empty(
                (num_gpu_blocks, block_size, num_kv_heads, head_dim),
                dtype=dtype,
                device='cuda',
            )
            for _ in range(num_layers * 2)  # K 和 V 各一份
        ]
```

这种预分配策略意味着 KV Cache 本身不会产生 allocator 级别的碎片——所有 block 都在同一个大张量内通过索引访问。碎片只发生在模型权重、中间激活等其他显存使用者之间。

**4. 合理设置 `gpu_memory_utilization`**

```bash
# 留出足够的余量给非 KV Cache 的显存需求
python -m vllm.entrypoints.openai.api_server \
    --gpu-memory-utilization 0.88  # 保留 12% 余量

# 过高的 utilization (如 0.98) 可能导致：
# - 中间激活分配失败
# - CUDA Graph 录制失败
# - allocator 碎片导致 OOM
```

## 8. 碎片问题的全景总结

```
┌───────────────────────────────────────────────────────────┐
│                   碎片问题全景                              │
│                                                           │
│  Level 1: KV Cache Block 层                               │
│  ├── 内部碎片：最后一个 block 未填满                        │
│  │   └── 影响：< 4%，通常可忽略                            │
│  ├── 外部碎片：PagedAttention 完全消除                      │
│  │   └── 影响：0%                                         │
│  └── 管理开销：block table 存储                             │
│      └── 影响：极小（KB 级别）                              │
│                                                           │
│  Level 2: PyTorch Allocator 层                            │
│  ├── Allocator 碎片：reserved > allocated                  │
│  │   └── 影响：长期运行后累积，可能导致 OOM                 │
│  ├── CUDA Graph 占用：固定显存                              │
│  │   └── 影响：减少可用 KV Cache 空间                       │
│  └── 缓解：expandable_segments, 预分配, 定期重启           │
│                                                           │
│  Level 3: CUDA Driver 层                                  │
│  ├── 物理页碎片：极少见                                     │
│  └── 虚拟地址碎片：cuMem API 可缓解                         │
│                                                           │
│  综合评估：                                                 │
│  PagedAttention 将显存浪费从 60-80% 降低到 < 4%            │
│  剩余碎片问题主要在 allocator 层，可通过工程手段缓解          │
└───────────────────────────────────────────────────────────┘
```

---

**关键结论：**
1. **内部碎片**是 PagedAttention 唯一的本质碎片来源，但在 `block_size=16` 下通常 < 4%
2. **外部碎片**被完全消除——这是 PagedAttention 最重要的贡献
3. **Allocator 碎片**是长期运行中的实际痛点，需要通过工程手段（预分配、`expandable_segments`、定期重启）缓解
4. `block_size` 的选择是碎片率 vs kernel 效率的权衡，16 是一个经过验证的默认值
