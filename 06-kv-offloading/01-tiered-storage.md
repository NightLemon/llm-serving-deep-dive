# 分层存储原理

> KV Cache offloading 的核心思想：将 GPU HBM 中暂时不用的 KV Cache 卸载到更便宜、更大的存储层级，在需要时再加载回来。

## 1. 内存层级全景

现代 LLM 推理服务器通常具备三个主要存储层级。理解它们的特征差异是设计 offloading 策略的基础。

### 1.1 存储层级特征表

| 层级 | 典型容量 | 带宽 | 延迟 | 每 GB 成本 | 代表硬件 |
|------|----------|------|------|-----------|---------|
| GPU HBM | 40-80 GB (单卡) | ~3.35 TB/s (HBM3) | ~ns | $$$$$ | H100 (80GB), A100 (80GB), H200 (141GB, HBM3e 4.8TB/s) |
| CPU DRAM | 256-2048 GB | ~100 GB/s (DDR5-4800 8ch) | ~80-100 ns | $$ | DDR5 RDIMM |
| NVMe SSD | 1-8 TB | ~7 GB/s (PCIe 5.0 x4) | ~10 μs | $ | Samsung PM9A3, Intel P5800X |

**关键比值：**

```
容量比  ：GPU HBM : CPU DRAM : NVMe ≈ 1 : 10-25 : 50-100
带宽比  ：GPU HBM : CPU DRAM : NVMe ≈ 1 : 1/30 : 1/500
延迟比  ：GPU HBM : CPU DRAM : NVMe ≈ 1 : 100 : 10000
```

### 1.2 GPU HBM：最稀缺的资源

以一台 8×H100 (80GB) 服务器为例，总 HBM 容量为 640 GB。这 640 GB 需要容纳：

```
┌─────────────────────────────────────────────┐
│              GPU HBM 使用分布                │
├──────────────────┬──────────────────────────┤
│  模型权重 (Weights)  │  Llama-3.1-70B FP16 ≈ 140 GB  │
│  KV Cache          │  动态分配，可达 200-400 GB    │
│  激活值 (Activations) │  取决于 batch size，~10-50 GB │
│  CUDA Context      │  每卡 ~0.5-1 GB              │
│  临时缓冲区          │  ~几 GB                     │
└──────────────────┴──────────────────────────┘
```

**KV Cache 是 HBM 最大的消费者。** 一个具体的计算：

```python
# Llama-3.1-70B 的 KV Cache 大小
num_layers = 80
num_kv_heads = 8          # GQA, 8 个 KV head
head_dim = 128
bytes_per_element = 2     # FP16

# 每个 token 的 KV Cache 大小
kv_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
# = 2 * 80 * 8 * 128 * 2 = 327,680 bytes ≈ 320 KB

# 如果 batch 中有 256 个请求，每个平均 2048 tokens
total_kv = 256 * 2048 * 327680 / 1024 / 1024 / 1024  # ≈ 160 GB (近似)
```

当 KV Cache 占满 HBM 后，新请求无法被调度，系统吞吐量骤降。这就是 offloading 的动机——将暂时不需要的 KV Cache "借存" 到 CPU DRAM 或 NVMe，为活跃请求腾出 GPU 空间。

### 1.3 H200 与 B200 的变化

NVIDIA H200 将 HBM 容量提升到 141 GB (HBM3e)，B200 进一步提升到 192 GB。这并不意味着 offloading 变得不重要——模型规模和上下文长度也在同步增长：

- Llama-3.1-405B 即使用 FP8 也需要 ~405 GB
- Claude/GPT-4 级别模型的 context window 已达 128K-1M tokens
- 单个 128K context 的 KV Cache 就可能占数十 GB

**存储层级的容量差距是结构性的**，不会因为单一层级的扩容而消失。

## 2. KV Cache 的冷热特征

### 2.1 为什么 KV Cache 可以被 offload？

LLM 推理中的 KV Cache 具有明显的**访问局部性**：

1. **Prefill 阶段**：一次性计算出所有 prompt token 的 KV Cache，之后这些 KV 值只会被读取
2. **Decode 阶段**：每个 step 只生成一个新 token，需要读取所有历史 KV Cache
3. **请求间的时间差异**：不同请求处于不同生命阶段，有些在等待 GPU 执行，有些暂时被 preempted

这意味着我们可以识别 "冷" 和 "热" 的 KV Cache：

```
热 (Hot)：当前正在 decode 的请求的 KV Cache —— 必须在 GPU HBM 中
温 (Warm)：即将被调度执行的请求 —— 最好在 GPU 中，也可以快速加载
冷 (Cold)：被 preempt 或暂停的请求 —— 可以安全地 offload 到 CPU/SSD
```

### 2.2 冷热转换的时机

```
┌──────────────┐    preempt / evict     ┌──────────────┐
│   Hot (GPU)  │ ──────────────────────► │  Cold (CPU)  │
│              │ ◄────────────────────── │              │
└──────────────┘    reload / prefetch    └──────────────┘
                                              │
                        深度 offload (可选)     │
                                              ▼
                                        ┌──────────────┐
                                        │  Frozen (SSD) │
                                        └──────────────┘
```

**典型场景：**
- **Preemption offload**：高优先级请求到来，低优先级请求的 KV Cache 被卸载到 CPU
- **Capacity-driven offload**：GPU block pool 接近满载，主动驱逐最不活跃的 KV blocks
- **Speculative offload**：预测某些请求短期内不会被调度，提前卸载

## 3. 异步传输：隐藏延迟的关键

### 3.1 PCIe 带宽与传输时间

GPU 和 CPU 之间通过 PCIe 总线通信。以 PCIe 5.0 x16 为例：

```
理论带宽：~64 GB/s（双向）
实际带宽：~50-55 GB/s（单向，考虑协议开销）
```

传输 1 GB 的 KV Cache 数据：
```
传输时间 ≈ 1 GB / 50 GB/s ≈ 20 ms
```

20 ms 对于 decode 来说是不可接受的延迟（一个 decode step 通常只需 5-15 ms）。因此 **必须使用异步传输来隐藏这个延迟**。

### 3.2 CUDA Stream Overlap

CUDA 提供了多 stream 机制，允许计算和数据传输并行执行：

```python
import torch

# 创建专用的传输 stream
transfer_stream = torch.cuda.Stream()

# 在主 stream 中执行当前 batch 的 attention 计算
with torch.cuda.stream(torch.cuda.default_stream()):
    output = attention(query, key_hot, value_hot)

# 同时在 transfer stream 中将冷 KV Cache 传到 CPU
with torch.cuda.stream(transfer_stream):
    # GPU → CPU: 异步传输（non-blocking）
    kv_cpu = kv_cold_gpu.to('cpu', non_blocking=True)

# 或者反向：将即将需要的 KV Cache 从 CPU 预取到 GPU
with torch.cuda.stream(transfer_stream):
    # CPU → GPU: 异步传输
    kv_prefetched = kv_from_cpu.to('cuda', non_blocking=True)

# 在需要使用预取数据之前，同步 transfer stream
transfer_stream.synchronize()
```

**核心原则：**

```
Timeline:
Main Stream:   [  Attention Batch N  ][  Attention Batch N+1  ][ ... ]
Transfer:      [  Offload Cold KV    ][  Prefetch Warm KV     ][ ... ]
               ▲                      ▲
               完全重叠，零额外延迟      完全重叠
```

### 3.3 Pinned Memory 优化

为了实现高效的异步传输，CPU 端必须使用 **pinned (page-locked) memory**：

```python
# 普通内存：需要先拷贝到 pinned buffer，再传到 GPU（两步）
cpu_tensor = torch.empty(size)  # pageable memory

# Pinned 内存：直接 DMA 传输（一步）
cpu_tensor_pinned = torch.empty(size, pin_memory=True)  # pinned memory
```

差异在于：
- **Pageable memory**：OS 可能将其 swap 到磁盘，GPU DMA 引擎无法直接访问，需要额外拷贝
- **Pinned memory**：锁定在物理内存中，GPU DMA 引擎可以直接访问，带宽接近 PCIe 理论值

**注意事项：** Pinned memory 不能被 OS 换出，过度分配会导致系统可用内存减少。一般建议不超过总 DRAM 的 50%。

### 3.4 传输粒度选择

offloading 的粒度选择直接影响效率：

| 粒度 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| Per-request | 整个请求的所有 KV Cache | 管理简单 | 传输量大，延迟高 |
| Per-layer | 按 Transformer 层传输 | 中等粒度 | 管理复杂度适中 |
| Per-block | 按 PagedAttention 的 block 传输 | 与 vLLM 内存管理一致 | 小块传输效率低 |
| Per-block-batch | 批量传输多个 block | 平衡效率和灵活性 | 需要聚合逻辑 |

vLLM 的实现主要采用 **per-block** 粒度，与 PagedAttention 的 block 管理自然契合。

## 4. 与 OS 虚拟内存的类比

KV Cache offloading 与操作系统的虚拟内存管理有深刻的类比关系：

### 4.1 概念映射

| OS 概念 | KV Cache Offloading 对应 |
|---------|------------------------|
| 物理内存 (RAM) | GPU HBM |
| Swap 分区 | CPU DRAM (作为 GPU 的 "swap") |
| 磁盘 | NVMe SSD |
| 页 (Page) | KV Block (PagedAttention 中的 block) |
| 页表 (Page Table) | Block Table (记录 block 位置的映射表) |
| 页面错误 (Page Fault) | Cache Miss (需要的 KV block 不在 GPU 中) |
| 页面替换算法 | 驱逐策略 (LRU, ARC 等) |
| 工作集 (Working Set) | 当前 decode batch 需要的 KV blocks |
| 预取 (Prefetch) | Speculative KV block loading |

### 4.2 关键差异

但这个类比也有重要的不同之处：

1. **延迟容忍度不同**：OS page fault 可能导致毫秒级延迟，用户可能不敏感。但 LLM decode 中的 "cache miss" 直接增加 TTFT 或 TPOT，对用户体验影响巨大。

2. **可预测性**：OS 难以预测下一个 page fault。但 LLM serving 的调度器**知道**下一个要执行哪个请求，可以**提前预取**所需的 KV blocks。

3. **粒度**：OS page 通常是 4KB。KV block 通常是几百 KB 到几 MB，传输粒度更大。

4. **传输路径**：OS swap 是 CPU↔磁盘。KV offloading 的主路径是 GPU↔CPU，通过 PCIe 总线，带宽特征完全不同。

### 4.3 借鉴 OS 的设计思路

尽管存在差异，OS 虚拟内存的很多经典设计思路都可以借鉴：

- **按需加载 (Demand Paging)**：只在请求真正需要执行时才将其 KV Cache 加载到 GPU
- **预取 (Prefetch)**：根据调度器的信息，提前加载即将执行的请求的 KV Cache
- **Copy-on-Write**：对于 prefix caching，多个请求共享同一份 prefix KV Cache，只在需要修改时才复制
- **Working Set 追踪**：追踪每个请求最近使用的 KV blocks，用于驱逐决策

## 5. 缓存替换算法

### 5.1 LRU (Least Recently Used)

最经典、最常用的替换算法。核心思想：**最久未被访问的 block 最先被驱逐**。

```python
from collections import OrderedDict

class LRUCache:
    """GPU KV Block Pool 的 LRU 驱逐管理"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity       # GPU 上最多能容纳多少 KV blocks
        self.cache = OrderedDict()     # block_id -> block_metadata
    
    def access(self, block_id: int):
        """某个 block 被使用（decode step 中被读取）"""
        if block_id in self.cache:
            self.cache.move_to_end(block_id)  # 移到最新
    
    def evict(self) -> int:
        """驱逐最久未使用的 block，返回被驱逐的 block_id"""
        if not self.cache:
            raise RuntimeError("No blocks to evict")
        block_id, _ = self.cache.popitem(last=False)  # 移除最旧
        return block_id
    
    def add(self, block_id: int, metadata: dict):
        """添加新 block 到 GPU"""
        if len(self.cache) >= self.capacity:
            self.evict()
        self.cache[block_id] = metadata
        self.cache.move_to_end(block_id)
```

**LRU 的优缺点：**
- 优点：实现简单，O(1) 操作，适合大多数场景
- 缺点：无法区分 "偶尔访问一次" 和 "频繁访问" 的 block。一次性扫描（如长 prompt 的 prefill）可能污染 cache

### 5.2 ARC (Adaptive Replacement Cache)

ARC 由 IBM 研究院提出，核心思想是**自适应地在 recency (最近性) 和 frequency (频率) 之间平衡**。

```
ARC 维护四个列表：
┌─────────────────────────────────────────────┐
│  T1: 最近访问过一次的 blocks (recency)        │
│  T2: 最近访问过多次的 blocks (frequency)       │
│  B1: T1 中被驱逐的 blocks 的 "ghost" 记录     │
│  B2: T2 中被驱逐的 blocks 的 "ghost" 记录     │
└─────────────────────────────────────────────┘

自适应参数 p：
- 如果 B1 频繁被命中 → 增大 p → 给 T1 更多空间（偏向 recency）
- 如果 B2 频繁被命中 → 减小 p → 给 T2 更多空间（偏向 frequency）
```

**ARC 的优势：**
- 自动适应不同的访问模式，无需手动调参
- 对 scan-resistant（扫描抗性）更好——一次性 prefill 不会轻易污染频繁访问的 cache
- 在 KV Cache offloading 场景中，某些 prefix 的 KV Cache 被反复使用（高 frequency），ARC 能更好地保护它们

### 5.3 算法选择建议

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| 请求间无共享前缀 | LRU | 简单有效，overhead 小 |
| 大量 prefix caching | ARC | 共享前缀被频繁访问，ARC 能保护 |
| 混合负载（长短 context） | ARC | 自适应平衡，不被长 context 的 prefill 污染 |
| 对延迟极度敏感 | LRU | 实现更简单，驱逐决策更快 |

## 6. Offloading 的延迟-吞吐量权衡

### 6.1 权衡模型

Offloading 本质上是**用延迟换吞吐量**：

```
无 Offloading：
  - 吞吐量受限于 GPU HBM 容量 → 能同时服务的请求数有上限
  - 延迟低 → 所有 KV Cache 都在 GPU 中
  - 新请求可能被排队等待

有 Offloading：
  - 吞吐量提升 → CPU DRAM 提供 10-25x 的额外 KV Cache 空间
  - 延迟可能增加 → reload KV Cache 需要 PCIe 传输时间
  - 更多请求可以被并发处理
```

### 6.2 何时使用 Offloading？

```
适合 offloading 的场景：
✓ 长 context 请求（>32K tokens），KV Cache 巨大
✓ 高并发，GPU HBM 经常成为瓶颈
✓ 批处理场景（对延迟不敏感，追求吞吐）
✓ 有大量 prefix caching 的场景（冷热区分明显）

不适合 offloading 的场景：
✗ 对 TTFT/TPOT 有严格 SLA 要求（如 <50ms）
✗ 短 context 请求为主（KV Cache 小，offloading overhead 不划算）
✗ GPU HBM 本身就够用（如 H200 141GB 跑小模型）
```

### 6.3 性能影响量化

以 Llama-3.1-70B 为例，单个请求 4096 tokens 的 KV Cache：

```
KV Cache 大小 = 4096 * 320 KB ≈ 1.28 GB

PCIe 5.0 传输时间（CPU → GPU）：
  1.28 GB / 50 GB/s ≈ 25.6 ms

如果使用异步预取，在前一个 batch 计算时传输：
  有效额外延迟 ≈ max(0, 传输时间 - 计算时间)
  
  假设每个 decode step 计算时间 = 10 ms，预取窗口 = 3 steps：
  可预取量 = 3 * 10 ms * 50 GB/s = 1.5 GB > 1.28 GB
  → 延迟完全隐藏！
```

这说明，**合理的预取策略可以将 offloading 的延迟开销降到接近零**。

## 7. 小结

| 要点 | 说明 |
|------|------|
| HBM 是瓶颈 | KV Cache 是 HBM 最大消费者，限制了并发请求数 |
| 三级存储 | GPU HBM → CPU DRAM → NVMe SSD，容量递增、带宽递减 |
| 冷热分离 | 活跃请求的 KV Cache 留在 GPU，被暂停的请求 offload 到 CPU |
| 异步传输 | CUDA stream overlap + pinned memory 可以隐藏大部分传输延迟 |
| 替换算法 | LRU 简单有效，ARC 适合有 prefix caching 的混合负载 |
| 权衡 | Offloading 用少量延迟换取显著的吞吐量提升 |

---

**下一节：** [vLLM KV Offloading 源码分析](02-vllm-offloading.md) —— 看看这些原理在 vLLM 中是如何实现的。
