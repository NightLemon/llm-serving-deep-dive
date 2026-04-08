# vLLM KV Offloading 源码分析

> 本节走读 vLLM (v0.8.x) 中 KV Cache offloading 的核心实现，理解 GPU↔CPU 数据搬运的完整链路。

## 1. 架构总览

vLLM 的 KV Cache offloading 实现分布在多个模块中，形成一个分层架构：

```
┌─────────────────────────────────────────────────────┐
│                    Scheduler                         │
│  (决定哪些请求需要 preempt / 哪些需要 reload)          │
└───────────┬─────────────────────────┬───────────────┘
            │                         │
            ▼                         ▼
┌───────────────────┐    ┌────────────────────────────┐
│  KV Offload       │    │  KV Transfer / Connector   │
│  Framework        │    │  Framework                 │
│  (v1/kv_offload/) │    │  (distributed/kv_transfer/)│
└───────────┬───────┘    └────────────┬───────────────┘
            │                         │
            ▼                         ▼
┌───────────────────────────────────────────────────────┐
│                   Worker Layer                         │
│  cpu_gpu.py — 实际执行 GPU↔CPU 数据传输                 │
│  使用 CUDA streams, pinned memory, async copy          │
└───────────────────────────────────────────────────────┘
```

### 1.1 核心模块一览

```
vllm/
├── v1/
│   ├── kv_offload/                          # 主要 offloading 框架
│   │   ├── abstract.py                      # 抽象基类
│   │   ├── cpu/
│   │   │   ├── manager.py                   # CPU offload 管理器
│   │   │   └── policies/
│   │   │       ├── base.py                  # 驱逐策略基类
│   │   │       ├── lru.py                   # LRU 策略
│   │   │       └── arc.py                   # ARC 策略
│   │   └── worker/
│   │       └── cpu_gpu.py                   # CPU↔GPU 传输 worker
│   │
│   └── simple_kv_offload/                   # 简化版 offloading
│       ├── manager.py                       # 统一管理接口
│       └── cuda_mem_ops.py                  # CUDA 内存操作封装
│
└── distributed/
    └── kv_transfer/
        └── kv_connector/
            └── v1/
                └── offloading_connector.py  # Offloading 连接器
```

## 2. 抽象接口：abstract.py

`abstract.py` 定义了 offloading 框架的核心抽象，所有具体实现都需要遵循这个接口：

```python
# vllm/v1/kv_offload/abstract.py (概念性重构)

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch

class KVOffloadManager(ABC):
    """KV Cache Offloading 管理器的抽象基类。
    
    职责：
    1. 管理 offload 存储空间（CPU DRAM 或其他后端）
    2. 执行 GPU → offload 存储的数据搬运（eviction）
    3. 执行 offload 存储 → GPU 的数据搬运（reload）
    4. 维护 block 位置映射（哪些 block 在 GPU，哪些在 CPU）
    """
    
    @abstractmethod
    def can_offload(self, num_blocks: int) -> bool:
        """检查 offload 存储是否有足够空间容纳 num_blocks 个 block"""
        ...
    
    @abstractmethod
    def offload_blocks(
        self, 
        block_ids: List[int],
        gpu_kv_cache: torch.Tensor,
    ) -> None:
        """将指定的 KV blocks 从 GPU 异步传输到 offload 存储。
        
        Args:
            block_ids: 需要 offload 的 GPU block IDs
            gpu_kv_cache: GPU 上的 KV Cache tensor
        """
        ...
    
    @abstractmethod
    def reload_blocks(
        self,
        block_ids: List[int],
        gpu_kv_cache: torch.Tensor,
    ) -> None:
        """将指定的 KV blocks 从 offload 存储异步传输回 GPU。
        
        Args:
            block_ids: 需要 reload 的 block IDs
            gpu_kv_cache: GPU 上的目标 KV Cache tensor
        """
        ...
    
    @abstractmethod
    def get_eviction_candidates(self, num_needed: int) -> List[int]:
        """根据驱逐策略选择需要被驱逐的 block IDs。
        
        Args:
            num_needed: 需要释放的 block 数量
        Returns:
            建议驱逐的 block ID 列表
        """
        ...
    
    @abstractmethod
    def sync(self) -> None:
        """等待所有异步传输操作完成"""
        ...
```

这个抽象层的设计使得 offloading 后端可以灵活替换——CPU DRAM、NVMe、甚至远程存储都可以作为后端实现。

## 3. CPU Offload Manager

`cpu/manager.py` 是最核心的实现，管理 CPU 端的 KV Cache 存储：

### 3.1 初始化与内存预分配

```python
# vllm/v1/kv_offload/cpu/manager.py (概念性重构)

class CPUKVOffloadManager(KVOffloadManager):
    """基于 CPU DRAM 的 KV Cache Offloading 管理器"""
    
    def __init__(
        self,
        num_cpu_blocks: int,        # CPU 端预分配的 block 数量
        block_size: int,            # 每个 block 的 token 数
        num_layers: int,            # Transformer 层数
        num_kv_heads: int,          # KV head 数量
        head_dim: int,              # head 维度
        dtype: torch.dtype,         # 数据类型 (FP16/BF16)
        eviction_policy: str = "lru",  # 驱逐策略
    ):
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        
        # 1. 预分配 CPU 端的 KV Cache 存储（pinned memory）
        #    形状与 GPU 端一致，方便直接 copy
        kv_cache_shape = (
            num_layers, 2,  # K 和 V
            num_cpu_blocks, block_size,
            num_kv_heads, head_dim
        )
        self.cpu_kv_cache = torch.empty(
            kv_cache_shape, dtype=dtype, 
            pin_memory=True  # 关键：使用 pinned memory
        )
        
        # 2. 空闲 block 池
        self.free_cpu_blocks: List[int] = list(range(num_cpu_blocks))
        
        # 3. 映射表：GPU block ID → CPU block ID
        self.gpu_to_cpu_map: Dict[int, int] = {}
        
        # 4. 驱逐策略
        self.eviction_policy = self._create_policy(eviction_policy)
        
        # 5. 传输用的 CUDA stream
        self.transfer_stream = torch.cuda.Stream()
        
        # 6. 统计信息
        self.stats = OffloadStats()
```

**关键设计决策解读：**

1. **预分配 pinned memory**：避免运行时的内存分配开销，pinned memory 确保 DMA 传输效率
2. **固定形状**：CPU 端 KV Cache 的 shape 与 GPU 端完全一致，使得 `copy_()` 操作可以直接执行
3. **独立的 CUDA stream**：传输操作不会阻塞 GPU 上的计算操作

### 3.2 Offload 流程

```python
    def offload_blocks(
        self,
        gpu_block_ids: List[int],
        gpu_kv_cache: torch.Tensor,
    ) -> None:
        """GPU → CPU 异步传输"""
        
        with torch.cuda.stream(self.transfer_stream):
            for gpu_bid in gpu_block_ids:
                # 分配一个空闲的 CPU block
                if not self.free_cpu_blocks:
                    raise RuntimeError("CPU offload storage is full")
                cpu_bid = self.free_cpu_blocks.pop()
                
                # 异步拷贝：GPU → CPU (all layers, K and V)
                # gpu_kv_cache[:, :, gpu_bid] 形状: [num_layers, 2, block_size, num_kv_heads, head_dim]
                self.cpu_kv_cache[:, :, cpu_bid].copy_(
                    gpu_kv_cache[:, :, gpu_bid],
                    non_blocking=True  # 异步传输
                )
                
                # 记录映射关系
                self.gpu_to_cpu_map[gpu_bid] = cpu_bid
                
                # 更新驱逐策略的状态
                self.eviction_policy.on_offload(gpu_bid)
                
                self.stats.offload_count += 1
                self.stats.offload_bytes += self._block_size_bytes()
```

### 3.3 Reload 流程

```python
    def reload_blocks(
        self,
        gpu_block_ids: List[int],
        gpu_kv_cache: torch.Tensor,
    ) -> None:
        """CPU → GPU 异步传输"""
        
        with torch.cuda.stream(self.transfer_stream):
            for gpu_bid in gpu_block_ids:
                if gpu_bid not in self.gpu_to_cpu_map:
                    raise KeyError(f"Block {gpu_bid} not found in CPU offload storage")
                
                cpu_bid = self.gpu_to_cpu_map[gpu_bid]
                
                # 异步拷贝：CPU → GPU
                gpu_kv_cache[:, :, gpu_bid].copy_(
                    self.cpu_kv_cache[:, :, cpu_bid],
                    non_blocking=True
                )
                
                # 归还 CPU block
                self.free_cpu_blocks.append(cpu_bid)
                del self.gpu_to_cpu_map[gpu_bid]
                
                # 更新驱逐策略
                self.eviction_policy.on_reload(gpu_bid)
                
                self.stats.reload_count += 1
```

### 3.4 驱逐触发条件

驱逐不是由 offload manager 自己触发的，而是由 **Scheduler** 在以下条件下触发：

```python
# Scheduler 中的驱逐触发逻辑（概念性）

class Scheduler:
    def _maybe_trigger_offload(self):
        """检查是否需要触发 KV Cache offloading"""
        
        gpu_usage = self.block_manager.get_usage_ratio()
        
        # 条件 1：GPU block pool 使用率超过高水位线
        if gpu_usage > self.high_watermark:  # 通常设为 0.9 (90%)
            num_to_evict = self._calculate_eviction_count(gpu_usage)
            candidates = self.offload_manager.get_eviction_candidates(num_to_evict)
            self.offload_manager.offload_blocks(candidates, self.gpu_kv_cache)
            self.block_manager.free_blocks(candidates)
            return
        
        # 条件 2：新请求需要空间但 GPU block pool 已满
        if self.waiting_queue and not self.block_manager.can_allocate(
            self.waiting_queue[0].num_blocks_needed
        ):
            # preempt 最低优先级的请求
            victim = self._select_preemption_victim()
            self.offload_manager.offload_blocks(
                victim.block_ids, self.gpu_kv_cache
            )
            victim.status = RequestStatus.PREEMPTED
```

**高水位 / 低水位机制：**

```
GPU Block Pool 使用率
100% ─────────────────── 满载，请求排队
 90% ─── 高水位线 ────── 触发 offload，驱逐到低水位线
      ↕ 正常运行区间
 70% ─── 低水位线 ────── 停止驱逐
      ↕ 宽裕区间
  0% ─────────────────── 空闲
```

## 4. 驱逐策略实现

### 4.1 LRU 策略

```python
# vllm/v1/kv_offload/cpu/policies/lru.py (概念性重构)

from collections import OrderedDict
from .base import EvictionPolicy

class LRUEvictionPolicy(EvictionPolicy):
    """LRU (Least Recently Used) 驱逐策略。
    
    维护一个按最后访问时间排序的有序字典。
    当需要驱逐时，选择最久未被访问的 blocks。
    """
    
    def __init__(self):
        # OrderedDict: block_id → last_access_timestamp
        # 最旧的在最前面
        self._order = OrderedDict()
    
    def on_access(self, block_id: int):
        """block 被访问（decode step 中被读取）"""
        if block_id in self._order:
            self._order.move_to_end(block_id)  # 移到末尾（最新）
        else:
            self._order[block_id] = True
    
    def on_offload(self, block_id: int):
        """block 被 offload 到 CPU"""
        self._order.pop(block_id, None)
    
    def on_reload(self, block_id: int):
        """block 被 reload 回 GPU"""
        self._order[block_id] = True
        self._order.move_to_end(block_id)
    
    def get_eviction_candidates(self, num_needed: int) -> List[int]:
        """返回最久未使用的 num_needed 个 blocks"""
        candidates = []
        for block_id in self._order:  # 从最旧开始迭代
            if len(candidates) >= num_needed:
                break
            candidates.append(block_id)
        return candidates
```

### 4.2 ARC 策略

ARC 的实现更复杂，维护四个列表和一个自适应参数：

```python
# vllm/v1/kv_offload/cpu/policies/arc.py (概念性重构)

class ARCEvictionPolicy(EvictionPolicy):
    """ARC (Adaptive Replacement Cache) 驱逐策略。
    
    维护四个列表：
    - T1: 最近访问过一次的 blocks（recency 列表）
    - T2: 最近访问过多次的 blocks（frequency 列表）
    - B1: T1 中被驱逐的 blocks 的 ghost 条目
    - B2: T2 中被驱逐的 blocks 的 ghost 条目
    
    自适应参数 p 控制 T1 和 T2 之间的空间分配。
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity  # cache 总容量（GPU blocks 数量）
        self.p = 0                # 自适应目标：T1 的理想大小
        
        # 实际 cache 中的 blocks
        self.t1 = OrderedDict()   # 访问过一次（recency）
        self.t2 = OrderedDict()   # 访问过多次（frequency）
        
        # Ghost 列表（只记录 block ID，不存储数据）
        self.b1 = OrderedDict()   # T1 的 ghost
        self.b2 = OrderedDict()   # T2 的 ghost
    
    def on_access(self, block_id: int):
        """处理 block 访问事件"""
        
        # Case 1: block 在 T1 中 → 移到 T2（从 recency 提升到 frequency）
        if block_id in self.t1:
            self.t1.pop(block_id)
            self.t2[block_id] = True
            self.t2.move_to_end(block_id)
            return
        
        # Case 2: block 在 T2 中 → 更新在 T2 中的位置
        if block_id in self.t2:
            self.t2.move_to_end(block_id)
            return
        
        # Case 3: block 在 B1 中（ghost hit）→ 增大 p，偏向 recency
        if block_id in self.b1:
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.p + delta, self.capacity)
            self.b1.pop(block_id)
            self._replace(block_id, in_b2=False)
            self.t2[block_id] = True
            return
        
        # Case 4: block 在 B2 中（ghost hit）→ 减小 p，偏向 frequency
        if block_id in self.b2:
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(self.p - delta, 0)
            self.b2.pop(block_id)
            self._replace(block_id, in_b2=True)
            self.t2[block_id] = True
            return
        
        # Case 5: 完全新的 block
        total_t1_b1 = len(self.t1) + len(self.b1)
        if total_t1_b1 == self.capacity:
            if len(self.t1) < self.capacity:
                self.b1.popitem(last=False)
                self._replace(block_id, in_b2=False)
            else:
                self.t1.popitem(last=False)
        elif total_t1_b1 < self.capacity:
            total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
            if total >= self.capacity:
                if total >= 2 * self.capacity:
                    self.b2.popitem(last=False)
                self._replace(block_id, in_b2=False)
        
        self.t1[block_id] = True
    
    def _replace(self, block_id: int, in_b2: bool):
        """内部替换操作"""
        if self.t1 and (
            len(self.t1) > self.p or 
            (in_b2 and len(self.t1) == self.p)
        ):
            # 从 T1 驱逐（最旧的）
            old_id, _ = self.t1.popitem(last=False)
            self.b1[old_id] = True  # 加入 ghost 列表
        else:
            # 从 T2 驱逐（最旧的）
            if self.t2:
                old_id, _ = self.t2.popitem(last=False)
                self.b2[old_id] = True
    
    def get_eviction_candidates(self, num_needed: int) -> List[int]:
        """获取驱逐候选"""
        candidates = []
        
        # 优先从 T1 的头部（最旧的 recency-only blocks）
        for bid in self.t1:
            if len(candidates) >= num_needed:
                break
            candidates.append(bid)
        
        # 如果不够，再从 T2 的头部
        if len(candidates) < num_needed:
            for bid in self.t2:
                if len(candidates) >= num_needed:
                    break
                candidates.append(bid)
        
        return candidates
```

**ARC 在 KV Cache 场景中的实际效果：**

考虑一个典型场景——多个请求共享同一个 system prompt：

```
请求 A: [system_prompt (共享)] [user_msg_A]
请求 B: [system_prompt (共享)] [user_msg_B]  
请求 C: [system_prompt (共享)] [user_msg_C]
请求 D: [long_document (独占)]               ← 一次性长 prefill
```

- **LRU**：请求 D 的长 document prefill 会将 system_prompt 的 KV blocks 挤出 cache
- **ARC**：system_prompt 的 KV blocks 因为被多次访问（A、B、C），进入 T2 (frequency 列表)，不会被 D 的一次性访问轻易驱逐

## 5. CPU↔GPU 传输 Worker

`worker/cpu_gpu.py` 负责实际的数据传输操作，这是整个 offloading 链路中最底层的模块：

```python
# vllm/v1/kv_offload/worker/cpu_gpu.py (概念性重构)

class CPUGPUTransferWorker:
    """CPU↔GPU 数据传输执行器。
    
    特点：
    1. 使用专用的 CUDA stream 执行异步传输
    2. 支持批量传输（多个 blocks 一次性发起）
    3. 维护传输队列和完成状态
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # 传输用的 CUDA stream（与计算 stream 分离）
        self.offload_stream = torch.cuda.Stream(device=device)
        self.reload_stream = torch.cuda.Stream(device=device)
        
        # 传输事件，用于同步
        self.offload_events: Dict[int, torch.cuda.Event] = {}
        self.reload_events: Dict[int, torch.cuda.Event] = {}
    
    def async_offload(
        self,
        src_gpu: torch.Tensor,       # GPU 上的源 KV Cache
        dst_cpu: torch.Tensor,        # CPU 上的目标存储（pinned）
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> torch.cuda.Event:
        """异步 GPU → CPU 传输，返回完成事件"""
        
        with torch.cuda.stream(self.offload_stream):
            for src_bid, dst_bid in zip(src_block_ids, dst_block_ids):
                # 逐 block 拷贝（所有 layer 的 K 和 V）
                dst_cpu[:, :, dst_bid].copy_(
                    src_gpu[:, :, src_bid],
                    non_blocking=True
                )
            
            # 记录一个事件，用于后续查询传输是否完成
            event = torch.cuda.Event()
            event.record(self.offload_stream)
        
        return event
    
    def async_reload(
        self,
        src_cpu: torch.Tensor,
        dst_gpu: torch.Tensor,
        src_block_ids: List[int],
        dst_block_ids: List[int],
    ) -> torch.cuda.Event:
        """异步 CPU → GPU 传输，返回完成事件"""
        
        with torch.cuda.stream(self.reload_stream):
            for src_bid, dst_bid in zip(src_block_ids, dst_block_ids):
                dst_gpu[:, :, dst_bid].copy_(
                    src_cpu[:, :, src_bid],
                    non_blocking=True
                )
            
            event = torch.cuda.Event()
            event.record(self.reload_stream)
        
        return event
    
    def is_transfer_complete(self, event: torch.cuda.Event) -> bool:
        """检查某次传输是否已完成（非阻塞查询）"""
        return event.query()
    
    def wait_for_transfer(self, event: torch.cuda.Event):
        """阻塞等待传输完成"""
        event.synchronize()
```

### 5.1 双 Stream 设计

```
GPU Compute Stream:  [Attention Batch 0][Attention Batch 1][Attention Batch 2]
Offload Stream:      [        Offload Cold Blocks        ]
Reload Stream:       [           Reload Warm Blocks              ]
                     ←─────── 三条 stream 并行执行 ─────────────►
```

使用两条独立的传输 stream（offload 和 reload），原因是：
1. PCIe 支持双向同时传输（全双工）
2. offload 和 reload 操作可以真正并行
3. 避免 offload 阻塞 reload（reload 通常优先级更高）

## 6. Simple KV Offload

`vllm/v1/simple_kv_offload/` 提供了一个简化版的 offloading 实现，适合理解核心概念：

```python
# vllm/v1/simple_kv_offload/manager.py (概念性重构)

class SimpleKVOffloadManager:
    """简化版 KV Offloading Manager。
    
    与完整版的区别：
    1. 只支持 LRU 策略（无 ARC）
    2. 同步传输（无异步 stream overlap）
    3. 无批量传输优化
    4. 更简单的内存管理
    
    适合：
    - 快速原型验证
    - 理解 offloading 核心概念
    - 对性能要求不高的场景
    """
    
    def __init__(self, num_cpu_blocks: int, kv_cache_spec):
        # 预分配 CPU 存储
        self.cpu_cache = self._allocate_cpu_cache(num_cpu_blocks, kv_cache_spec)
        self.block_map = {}  # gpu_block_id -> cpu_block_id
        self.free_blocks = list(range(num_cpu_blocks))
    
    def offload(self, gpu_block_id: int, gpu_cache: torch.Tensor):
        """同步 offload 一个 block"""
        cpu_bid = self.free_blocks.pop()
        # 同步传输（简单但会阻塞 GPU）
        self.cpu_cache[:, :, cpu_bid] = gpu_cache[:, :, gpu_block_id].cpu()
        self.block_map[gpu_block_id] = cpu_bid
    
    def reload(self, gpu_block_id: int, gpu_cache: torch.Tensor):
        """同步 reload 一个 block"""
        cpu_bid = self.block_map.pop(gpu_block_id)
        gpu_cache[:, :, gpu_block_id].copy_(
            self.cpu_cache[:, :, cpu_bid].cuda()
        )
        self.free_blocks.append(cpu_bid)
```

### 6.1 CUDA 内存操作封装

```python
# vllm/v1/simple_kv_offload/cuda_mem_ops.py (概念性重构)

class CUDAMemOps:
    """封装 CUDA 内存操作，提供统一的 copy/allocate 接口。
    
    主要功能：
    1. pinned memory 分配
    2. 异步 copy 封装
    3. 内存池管理
    """
    
    @staticmethod
    def alloc_pinned(shape, dtype):
        """分配 pinned (page-locked) CPU memory"""
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    
    @staticmethod
    def async_copy_d2h(src_gpu, dst_cpu, stream=None):
        """Device to Host 异步拷贝"""
        if stream is None:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            dst_cpu.copy_(src_gpu, non_blocking=True)
    
    @staticmethod
    def async_copy_h2d(src_cpu, dst_gpu, stream=None):
        """Host to Device 异步拷贝"""
        if stream is None:
            stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            dst_gpu.copy_(src_cpu, non_blocking=True)
```

## 7. Offloading Connector

`offloading_connector.py` 是 vLLM 的 KV transfer 框架与 offloading 模块之间的桥梁：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py (概念性)

class OffloadingConnector:
    """将 KV Offloading 集成到 vLLM 的 KV Transfer 框架中。
    
    KV Transfer 框架原本用于 disaggregated serving（prefill-decode 分离），
    Offloading Connector 复用了这个框架来实现 GPU↔CPU offloading。
    
    这种设计的好处：
    1. 统一的 KV 数据传输接口
    2. 可以同时支持跨节点传输和本地 offloading
    3. 复用 KV transfer 的序列化/反序列化逻辑
    """
    
    def __init__(self, config):
        self.offload_manager = CPUKVOffloadManager(
            num_cpu_blocks=config.num_cpu_offload_blocks,
            block_size=config.block_size,
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            eviction_policy=config.offload_eviction_policy,
        )
    
    def send_kv_caches(self, block_ids, kv_cache):
        """'发送' KV Cache —— 在 offloading 场景下就是 offload 到 CPU"""
        self.offload_manager.offload_blocks(block_ids, kv_cache)
    
    def recv_kv_caches(self, block_ids, kv_cache):
        """'接收' KV Cache —— 在 offloading 场景下就是从 CPU reload"""
        self.offload_manager.reload_blocks(block_ids, kv_cache)
```

**设计亮点：** 通过将 offloading 适配到 `kv_connector` 接口，vLLM 实现了 offloading 和 disaggregated serving 的统一抽象。Scheduler 不需要关心数据是传输到远程节点还是 offload 到本地 CPU，都通过同一套 connector 接口操作。

## 8. 配置与使用

### 8.1 启用 KV Offloading

```bash
# 启动 vLLM 时启用 CPU offloading
vllm serve meta-llama/Llama-3.1-70B \
    --kv-cache-dtype fp16 \
    --cpu-offload-gb 50 \          # 分配 50 GB CPU DRAM 给 offload
    --swap-space 20 \              # 传统 swap 空间（与 offload 互补）
    --gpu-memory-utilization 0.95  # GPU 内存利用率
```

### 8.2 关键配置参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--cpu-offload-gb` | CPU offload 使用的 DRAM 大小 (GB) | 总 DRAM 的 30-50% |
| `--swap-space` | 传统 swap 空间大小 (GB) | 10-30 GB |
| `--gpu-memory-utilization` | GPU HBM 使用率上限 | 0.90-0.95 |
| eviction policy | 驱逐策略 (LRU / ARC) | ARC (有 prefix caching 时) |

### 8.3 Offloading vs Swap 的区别

vLLM 中 "swap" 和 "offloading" 看似相似，但有微妙的区别：

| 维度 | Swap | KV Offloading |
|------|------|---------------|
| 触发时机 | 被 preempt 时 | 主动 + 被动 |
| 粒度 | Per-request (所有 blocks) | Per-block (可选择性) |
| 策略 | 无选择（整个请求 swap out） | 有驱逐策略（LRU/ARC） |
| 目标 | 腾出 GPU 空间给高优先级请求 | 扩展有效 KV Cache 容量 |
| 预取 | 无 | 有（基于调度预测） |

## 9. 性能调优要点

### 9.1 传输 batch size

单个 block 的传输效率较低（PCIe 有固定的启动开销）。建议批量传输：

```python
# 差：逐个 block 传输
for block_id in blocks_to_offload:
    offload_one_block(block_id)  # 每次 PCIe 启动开销 ~1-2 μs

# 好：批量传输（合并多个 block 的数据，减少 PCIe 事务数）
offload_batch(blocks_to_offload)  # 一次 PCIe 事务
```

### 9.2 监控指标

```python
# 关键监控指标
metrics = {
    "offload_rate": "offload 次数/秒",
    "reload_rate": "reload 次数/秒",
    "offload_hit_rate": "reload 命中率（block 确实在 CPU 中）",
    "transfer_overlap_ratio": "传输与计算重叠比例",
    "gpu_block_utilization": "GPU block pool 使用率",
    "cpu_block_utilization": "CPU offload 存储使用率",
    "avg_offload_latency_ms": "平均 offload 延迟",
    "avg_reload_latency_ms": "平均 reload 延迟",
}
```

## 10. 小结

| 模块 | 职责 | 关键实现 |
|------|------|---------|
| `abstract.py` | 定义 offloading 接口 | `offload_blocks()`, `reload_blocks()`, `get_eviction_candidates()` |
| `cpu/manager.py` | CPU 端存储管理 | pinned memory 预分配, block 映射, 异步传输 |
| `cpu/policies/` | 驱逐策略 | LRU (OrderedDict), ARC (四列表 + 自适应参数) |
| `worker/cpu_gpu.py` | 数据传输执行 | 双 CUDA stream, 异步 copy, 事件同步 |
| `simple_kv_offload/` | 简化版实现 | 同步传输, 仅 LRU, 适合原型 |
| `offloading_connector.py` | 框架集成 | 适配 kv_connector 接口, 统一 offload 和 disagg serving |

---

**下一节：** [OpenAI Extended Prompt Caching 分析](03-extended-caching.md) —— 看看 OpenAI 是如何在生产环境中实现大规模 KV Cache 持久化的。
