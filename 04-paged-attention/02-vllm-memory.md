# vLLM v1 内存管理源码走读

> 本节走读 vLLM v1 架构中 KV Cache 内存管理的核心源码，涵盖 BlockPool、KVCacheManager、KVCacheCoordinator 和 BlockTable 四个关键组件。
> 
> **源码版本基准**：vLLM ≥ 0.8.x（v1 架构）。v1 是 vLLM 从 0.7 开始引入的新调度 / worker 架构，显著简化了代码路径。

## 1. 架构总览

在 vLLM v1 中，KV Cache 内存管理采用**分层设计**：

```
┌──────────────────────────────────────────────────────────────┐
│                      Scheduler                               │
│  负责请求级别的调度决策：哪些请求 prefill，哪些 decode，       │
│  哪些需要 preempt                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────┐                         │
│  │      KVCacheManager             │                         │
│  │  逻辑层：管理每个 request 的     │                         │
│  │  block 分配、prefix cache 命中   │                         │
│  │         │                       │                         │
│  │         ▼                       │                         │
│  │  ┌─────────────────────────┐    │                         │
│  │  │  KVCacheCoordinator     │    │                         │
│  │  │  协调多层 KV Cache 配置  │    │                         │
│  │  │         │               │    │                         │
│  │  │         ▼               │    │                         │
│  │  │  ┌─────────────────┐    │    │                         │
│  │  │  │   BlockPool     │    │    │                         │
│  │  │  │  物理 block 池   │    │    │                         │
│  │  │  │  分配 / 释放     │    │    │                         │
│  │  │  └─────────────────┘    │    │                         │
│  │  └─────────────────────────┘    │                         │
│  └─────────────────────────────────┘                         │
│                                                              │
│  ════════════════ CPU / GPU 边界 ════════════════            │
│                                                              │
│  ┌─────────────────────────────────┐                         │
│  │   BlockTable (GPU Worker 端)    │                         │
│  │   维护 block mapping 的 GPU 张量│                         │
│  │   传递给 attention kernel       │                         │
│  └─────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

## 2. BlockPool：物理 Block 池管理

> 源码位置：`vllm/v1/core/block_pool.py`

### 2.1 初始化

```python
class BlockPool:
    def __init__(self, num_gpu_blocks: int, enable_caching: bool, ...):
        # 核心数据结构
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        
        # 每个物理 block 的引用计数
        # ref_count > 0 表示 block 正在使用
        # ref_count == 0 表示 block 空闲（但可能还在 cache 中）
        self.ref_cnts = np.zeros(num_gpu_blocks, dtype=np.int32)
        
        # Free list：空闲 block 的集合
        # 使用 deque 实现 O(1) 的 pop/append
        self.free_blocks: deque[int] = deque(range(num_gpu_blocks))
        
        # Cached blocks 的管理（用于 prefix caching）
        # block_hash → block_id 的映射
        self.cached_block_hash_to_block: Dict[int, List[int]] = {}
        
        # Eviction 相关
        # 当 free list 为空时，需要从 cache 中驱逐 block
        self.eviction_order: Optional[...] = ...  # LRU 或其他策略
```

`num_gpu_blocks` 的值在 vLLM 启动时确定：

```python
# 简化的 block 数量计算逻辑
total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
model_memory = profile_model_memory()  # 通过 profiling 测量
available_for_kv = (total_gpu_memory * gpu_memory_utilization) - model_memory
block_memory = block_size * num_layers * 2 * num_kv_heads * head_dim * dtype_size
num_gpu_blocks = available_for_kv // block_memory
```

### 2.2 Block 分配

```python
def allocate(self, num_blocks: int) -> List[int]:
    """从 free list 分配指定数量的物理 block"""
    if len(self.free_blocks) < num_blocks:
        if self.enable_caching:
            # 尝试从 cache 中驱逐 block
            self._evict_blocks(num_blocks - len(self.free_blocks))
        
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("Out of GPU memory for KV cache blocks!")
    
    allocated = []
    for _ in range(num_blocks):
        block_id = self.free_blocks.popleft()
        self.ref_cnts[block_id] = 1
        allocated.append(block_id)
    
    return allocated
```

### 2.3 Block 释放与引用计数

```python
def free(self, block_id: int) -> None:
    """释放一个物理 block（引用计数减 1）"""
    self.ref_cnts[block_id] -= 1
    assert self.ref_cnts[block_id] >= 0
    
    if self.ref_cnts[block_id] == 0:
        if self.enable_caching:
            # 不立即回收，保留在 cache 中
            # 后续可能被其他请求复用（prefix caching）
            self._add_to_cache(block_id)
        else:
            # 直接加入 free list
            self.free_blocks.append(block_id)

def increase_ref_count(self, block_id: int) -> None:
    """增加引用计数（用于 prefix sharing、beam search）"""
    self.ref_cnts[block_id] += 1
```

引用计数的使用场景：

| 场景 | ref_count 行为 |
|------|---------------|
| 新分配 | ref_count = 1 |
| Prefix caching 命中 | ref_count += 1（共享已有 block） |
| Beam search fork | ref_count += beam_width - 1 |
| 请求完成释放 | ref_count -= 1 |
| ref_count 归零 | 加入 free list 或 cache |

### 2.4 Free List 实现细节

vLLM 的 free list 使用 Python `deque` 实现。相比链表：
- `popleft()` 和 `append()` 都是 O(1)
- 内存局部性更好
- 但不支持 O(1) 的随机删除（驱逐 cached block 时需要）

对于 cached block 的驱逐，vLLM 使用独立的数据结构（如 LRU list）来管理驱逐顺序，与 free list 分离。

## 3. KVCacheManager：KV Cache 调度

> 源码位置：`vllm/v1/core/kv_cache_manager.py`

### 3.1 核心职责

KVCacheManager 是 Scheduler 和 BlockPool 之间的桥梁：

```python
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, ...):
        # 持有 BlockPool 实例（通过 KVCacheCoordinator）
        self.coordinator = KVCacheCoordinator(kv_cache_config, ...)
        
        # 每个 request 的 block table（逻辑 → 物理映射）
        self.req_to_blocks: Dict[str, List[int]] = {}
        
        # Prefix caching 相关
        self.enable_caching = kv_cache_config.enable_prefix_caching
```

### 3.2 allocate_slots：为新 token 分配 KV 存储

这是最核心的方法，在每次调度循环中被调用：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_computed_tokens: int,
) -> Optional[KVCacheAllocationResult]:
    """
    为 request 的新 token 分配 KV cache slots。
    
    Args:
        request: 请求对象
        num_new_tokens: 本次需要处理的新 token 数
        num_computed_tokens: 已经计算（缓存命中）的 token 数
    
    Returns:
        分配结果，包含新分配的 block IDs；
        如果显存不足返回 None（触发 preemption）
    """
    # 计算需要多少个 block
    total_tokens = num_computed_tokens + num_new_tokens
    num_required_blocks = ceildiv(total_tokens, self.block_size)
    
    # 已有的 block 数量
    existing_blocks = self.req_to_blocks.get(request.request_id, [])
    num_existing_blocks = len(existing_blocks)
    
    # 需要新分配的 block 数量
    num_new_blocks = num_required_blocks - num_existing_blocks
    
    if num_new_blocks <= 0:
        # 现有 block 足够（新 token 填入最后一个 block 的空 slot）
        return KVCacheAllocationResult(new_blocks=[])
    
    # 尝试分配
    new_blocks = self.coordinator.allocate(num_new_blocks)
    if new_blocks is None:
        return None  # 分配失败，需要 preemption
    
    # 更新 block table
    existing_blocks.extend(new_blocks)
    self.req_to_blocks[request.request_id] = existing_blocks
    
    return KVCacheAllocationResult(new_blocks=new_blocks)
```

关键设计点：

1. **增量分配**：不是一次性分配所有 block，而是随着 token 生成逐步分配
2. **复用现有 block**：新 token 优先填入最后一个 block 的空 slot
3. **失败即 preempt**：返回 None 告知 Scheduler 需要抢占其他请求

### 3.3 get_computed_blocks：Prefix Caching 命中判定

```python
def get_computed_blocks(
    self,
    request: Request,
) -> Tuple[List[int], int]:
    """
    检查 request 的 prompt tokens 是否有 prefix cache 命中。
    
    Returns:
        (cached_block_ids, num_computed_tokens):
        命中的物理 block ID 列表和已计算的 token 数
    """
    if not self.enable_caching:
        return [], 0
    
    # 计算 prompt tokens 的 block hash
    token_ids = request.prompt_token_ids
    block_hashes = self._compute_block_hashes(token_ids)
    
    cached_blocks = []
    num_computed = 0
    
    for i, block_hash in enumerate(block_hashes):
        block_id = self.coordinator.get_cached_block(block_hash)
        if block_id is not None:
            cached_blocks.append(block_id)
            # 增加引用计数
            self.coordinator.increase_ref_count(block_id)
            num_computed += self.block_size
        else:
            break  # Prefix caching 要求连续命中
    
    # 更新 request 的 block table
    self.req_to_blocks[request.request_id] = cached_blocks
    
    return cached_blocks, min(num_computed, len(token_ids))
```

### 3.4 free：请求结束后释放 Block

```python
def free(self, request_id: str) -> None:
    """释放 request 占用的所有 block"""
    blocks = self.req_to_blocks.pop(request_id, [])
    for block_id in blocks:
        self.coordinator.free(block_id)
```

### 3.5 与 Scheduler 的交互

```python
# vllm/v1/core/scheduler.py 中的简化调用流程
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        # 1. 尝试为 waiting 队列中的请求分配
        for request in self.waiting_queue:
            computed_blocks, num_computed = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            
            result = self.kv_cache_manager.allocate_slots(
                request, 
                num_new_tokens=len(request.prompt_token_ids) - num_computed,
                num_computed_tokens=num_computed,
            )
            
            if result is None:
                # 显存不足，停止调度新请求
                break
            
            self.running_queue.append(request)
        
        # 2. 为 running 队列中的请求分配 decode token 的 slot
        for request in self.running_queue:
            result = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens=1,  # decode 每步只生成 1 个 token
                num_computed_tokens=request.num_computed_tokens,
            )
            
            if result is None:
                # 显存不足，需要 preempt 某些请求
                self._preempt(request)
        
        # 3. 请求完成后释放
        for request in completed_requests:
            self.kv_cache_manager.free(request.request_id)
```

## 4. KVCacheCoordinator：多层 KV Cache 协调

> 源码位置：`vllm/v1/core/kv_cache_coordinator.py`

### 4.1 设计动机

在 vLLM v1 中引入 KVCacheCoordinator 的原因：

1. **Hybrid KV Cache**：某些模型的不同 layer group 可能使用不同的 KV cache 配置（例如 sliding window attention + full attention）
2. **多级缓存**：未来可能支持 GPU + CPU + SSD 多级 KV cache
3. **解耦管理逻辑**：将 block 分配的协调逻辑从 KVCacheManager 中抽离

```python
class KVCacheCoordinator:
    def __init__(self, kv_cache_config, ...):
        # 可能持有多个 BlockPool，每个对应不同的 layer group
        self.block_pools: List[BlockPool] = []
        
        # Layer group → BlockPool 的映射
        self.layer_to_pool: Dict[int, int] = {}
        
        # 初始化 block pools
        for group_config in kv_cache_config.groups:
            pool = BlockPool(
                num_gpu_blocks=group_config.num_blocks,
                enable_caching=kv_cache_config.enable_prefix_caching,
            )
            self.block_pools.append(pool)
```

### 4.2 协调分配

```python
def allocate(self, num_blocks: int) -> Optional[List[int]]:
    """
    从所有相关 pool 分配 block。
    对于标准模型（所有 layer 共享同一配置），
    这只是简单地委托给唯一的 BlockPool。
    """
    # 简单情况：所有 layer 共享一个 pool
    if len(self.block_pools) == 1:
        try:
            return self.block_pools[0].allocate(num_blocks)
        except RuntimeError:
            return None
    
    # 复杂情况：不同 layer group 需要独立分配
    # 需要确保所有 group 都能成功分配（原子性）
    all_allocated = []
    for pool in self.block_pools:
        try:
            blocks = pool.allocate(num_blocks)
            all_allocated.append(blocks)
        except RuntimeError:
            # 回滚已分配的 blocks
            for prev_pool, prev_blocks in zip(self.block_pools, all_allocated):
                for block_id in prev_blocks:
                    prev_pool.free(block_id)
            return None
    
    return all_allocated
```

### 4.3 Hybrid Attention 场景

对于使用 sliding window attention + full attention 混合架构的模型（如 Gemma 2、Jamba），不同 layer 的 KV cache 行为不同：

```
Layer 0-5:  Sliding Window Attention (window=4096)
  → 只需要保留最近 4096 tokens 的 KV
  → block 可以被复用（旧的 block 超出窗口后释放）

Layer 6-11: Full Attention
  → 需要保留所有 tokens 的 KV
  → block 不能提前释放

KVCacheCoordinator 确保两组 layer 的 block 独立管理，
避免 sliding window layer 的 block 回收影响 full attention layer。
```

## 5. BlockTable：GPU 端表示

> 源码位置：`vllm/v1/worker/block_table.py`

### 5.1 CPU 侧的 BlockTable 类

```python
class BlockTable:
    """管理所有请求的 block table，并负责与 GPU 同步。"""
    
    def __init__(self, max_num_reqs: int, max_num_blocks_per_req: int, 
                 pin_memory: bool, device: torch.device):
        # CPU 侧的 block table（pinned memory for fast H2D transfer）
        # 形状: [max_num_reqs, max_num_blocks_per_req]
        # 值: 物理 block ID
        self.block_table_np = np.full(
            (max_num_reqs, max_num_blocks_per_req), 
            -1,  # -1 表示未分配
            dtype=np.int32,
        )
        
        # GPU 侧的 block table 张量
        self.block_table_gpu = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=torch.int32,
            device=device,
        )
```

### 5.2 更新与同步

```python
def update(self, req_index: int, block_ids: List[int]) -> None:
    """更新某个请求的 block table"""
    for i, block_id in enumerate(block_ids):
        self.block_table_np[req_index, i] = block_id

def commit(self) -> None:
    """
    将 CPU 侧的 block table 同步到 GPU。
    使用 pinned memory → device 的异步拷贝。
    """
    # 注意：实际实现中会做增量更新，只同步变化的部分
    self.block_table_gpu.copy_(
        torch.from_numpy(self.block_table_np), 
        non_blocking=True
    )
```

### 5.3 传递给 Attention Kernel

```python
# 在 model_runner 中，block_table_gpu 被传递给 attention kernel
# 简化的调用链：

class GPUModelRunner:
    def execute_model(self, scheduler_output):
        # ... 准备输入 ...
        
        # attention kernel 需要 block table 来定位 KV cache
        attn_metadata = AttentionMetadata(
            block_table=self.block_table.block_table_gpu,
            context_lens=context_lens,
            # ... 其他元数据 ...
        )
        
        # 执行模型前向传播
        output = self.model(input_ids, positions, attn_metadata)
```

Attention kernel 使用 block table 的方式（以 FlashInfer 为例）：

```
Query token at position p:
  1. 计算逻辑 block index = p // block_size
  2. 查 block_table 得到物理 block index
  3. 在 key_cache / value_cache 中定位到正确的物理位置
  4. 执行 attention 计算
```

## 6. 完整的 Block 分配流程图

```
用户发送请求
    │
    ▼
Scheduler.schedule()
    │
    ├─── 新请求 (Waiting → Running)
    │       │
    │       ▼
    │    KVCacheManager.get_computed_blocks()
    │       │
    │       ├── 有 prefix cache 命中
    │       │       │
    │       │       ▼
    │       │    BlockPool.increase_ref_count()  ← 共享已有 block
    │       │       │
    │       │       ▼
    │       │    返回 (cached_blocks, num_computed)
    │       │
    │       ├── 无命中
    │       │       │
    │       │       ▼
    │       │    返回 ([], 0)
    │       │
    │       ▼
    │    KVCacheManager.allocate_slots(num_new_tokens)
    │       │
    │       ▼
    │    KVCacheCoordinator.allocate(num_new_blocks)
    │       │
    │       ▼
    │    BlockPool.allocate()
    │       │
    │       ├── Free list 有足够 block → 分配成功
    │       │
    │       ├── Free list 不足，cache 中有可驱逐 block
    │       │       │
    │       │       ▼
    │       │    BlockPool._evict_blocks()  → 驱逐 LRU block
    │       │       │
    │       │       ▼
    │       │    分配成功
    │       │
    │       └── 完全不足 → 返回 None
    │               │
    │               ▼
    │            Scheduler 停止调度新请求
    │
    ├─── 运行中请求 (Decode)
    │       │
    │       ▼
    │    KVCacheManager.allocate_slots(num_new_tokens=1)
    │       │
    │       ├── 最后一个 block 还有空 slot → 无需新 block
    │       │
    │       └── 最后一个 block 满了 → 分配 1 个新 block
    │               │
    │               ├── 成功
    │               └── 失败 → Preemption（见 03-preemption.md）
    │
    └─── 完成的请求
            │
            ▼
         KVCacheManager.free(request_id)
            │
            ▼
         BlockPool.free(block_id)  × N
            │
            ├── enable_caching=True → 保留在 cache 中
            └── enable_caching=False → 加入 free list
```

## 7. 关键设计决策总结

| 设计决策 | 选择 | 原因 |
|---------|------|------|
| Free list 数据结构 | `deque` | O(1) 分配/回收，简单高效 |
| 引用计数 | `numpy array` | 向量化操作，避免 Python dict 开销 |
| Block table 同步 | Pinned memory + async copy | 最小化 H2D 传输延迟 |
| Cache 驱逐策略 | LRU | 简单有效，局部性假设合理 |
| 分配粒度 | 单个 block | 最小化内部碎片 |
| Coordinator 设计 | 可扩展多 pool | 支持 hybrid attention 和未来多级缓存 |

## 8. v0（旧架构）vs v1 的差异

| 方面 | v0 | v1 |
|------|-----|-----|
| Block 管理 | `BlockSpaceManager` + `BlockAllocator` | `BlockPool` + `KVCacheManager` |
| 代码复杂度 | 多层抽象，~2000 行 | 扁平化，~800 行 |
| Prefix caching | 独立的 `PrefixCachingBlockAllocator` | 集成在 `BlockPool` 中 |
| Swap 支持 | 完整的 CPU block allocator | v1 初期简化，后续逐步补全 |
| 多 worker | 复杂的 block 映射同步 | 简化的直接传递 |

v1 架构的核心理念是**简化**——移除不必要的抽象层，让代码路径更直接。这也使得源码阅读更加友好。

---

**建议的源码阅读顺序：**
1. `block_pool.py` — 理解物理 block 的生命周期
2. `kv_cache_manager.py` — 理解请求级别的 block 分配逻辑
3. `kv_cache_coordinator.py` — 理解多层协调（可跳过，除非研究 hybrid attention）
4. `block_table.py` — 理解 CPU-GPU 同步
5. `scheduler.py` — 理解以上组件如何被调度器驱动
