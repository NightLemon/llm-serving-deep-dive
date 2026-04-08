# vLLM APC 源码分析

> 本节深入走读 vLLM 的 Automatic Prefix Caching（APC）实现。vLLM 从 v0.3 起引入 APC，在 v1 架构中进行了重大重构。本节基于 vLLM v1 架构（2025 年）的源码进行分析。

## 1. APC 概览

vLLM 的 Automatic Prefix Caching 允许不同请求之间自动共享相同前缀的 KV Cache block。其核心思路是：

1. 将 token 序列按 `block_size` 分成 block
2. 对每个 block 计算 content hash（链式 hash）
3. 维护一个全局的 `hash → physical_block_id` 映射表
4. 新请求到来时，逐 block 查找 hash 匹配
5. 匹配的 block 直接复用物理内存，跳过 prefill 计算

启用方式：

```bash
# vLLM v1 默认启用 prefix caching
vllm serve meta-llama/Llama-3.1-8B-Instruct

# 显式启用/禁用
vllm serve meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching
vllm serve meta-llama/Llama-3.1-8B-Instruct --no-enable-prefix-caching
```

在 vLLM v1 中，prefix caching 默认启用（`--enable-prefix-caching` 为 True）。

## 2. 核心文件一览

```
vllm/v1/core/
├── kv_cache_utils.py      # Hash 计算、Block Hash 数据结构
├── kv_cache_manager.py    # KV Cache 管理器：分配、匹配、驱逐
├── block_pool.py          # 物理 block 池：引用计数、空闲列表
└── sched/
    └── scheduler.py       # 调度器：决定哪些请求执行 prefill
```

## 3. Hash 计算：`kv_cache_utils.py`

### 3.1 `PrefixHash` 的定义

vLLM 使用 Python 内置的 `hash()` 函数计算 block hash。每个 block 的 hash 是一个链式结构：

```python
# 简化后的核心逻辑
# 来自 vllm/v1/core/kv_cache_utils.py

# 前缀 hash 的特殊初始值
NONE_HASH = -1

def hash_block_tokens(
    parent_block_hash: int,
    curr_block_token_ids: tuple[int, ...],
    extra_keys: Optional[tuple] = None,
) -> int:
    """计算一个 block 的 hash。
    
    Args:
        parent_block_hash: 前一个 block 的 hash，保证链式依赖
        curr_block_token_ids: 当前 block 的 token ID 元组
        extra_keys: 额外的哈希键（如 LoRA ID、多模态内容 hash 等）
    
    Returns:
        当前 block 的 hash 值
    """
    if extra_keys is not None:
        return hash((parent_block_hash, curr_block_token_ids, extra_keys))
    return hash((parent_block_hash, curr_block_token_ids))
```

### 3.2 链式 Hash 的工作过程

对于一个 token 序列 `[t_0, t_1, ..., t_63]`（假设 block_size = 16）：

```
Block 0: tokens = (t_0, t_1, ..., t_15)
  hash_0 = hash((NONE_HASH, (t_0, ..., t_15)))

Block 1: tokens = (t_16, t_17, ..., t_31)
  hash_1 = hash((hash_0, (t_16, ..., t_31)))

Block 2: tokens = (t_32, t_33, ..., t_47)
  hash_2 = hash((hash_1, (t_32, ..., t_47)))

Block 3: tokens = (t_48, t_49, ..., t_63)
  hash_3 = hash((hash_2, (t_48, ..., t_63)))
```

`hash_3` 隐含了 Block 0-3 所有 token 的信息。两个不同请求只有在 Block 0-3 的 token 完全相同时，`hash_3` 才会相同。

### 3.3 `BlockHashData` 数据结构

vLLM 将每个请求的 block hash 信息封装在 `BlockHashData` 中：

```python
class BlockHashData:
    """存储一个请求的所有 block hash 信息。"""
    
    def __init__(self):
        # block_index -> block_hash
        self.block_hashes: list[int] = []
    
    def append(self, block_hash: int):
        self.block_hashes.append(block_hash)
```

### 3.4 Extra Keys：LoRA 和多模态支持

`extra_keys` 参数允许在 hash 中加入非 token 内容的信息：

```python
# LoRA 场景：相同 token 但不同 LoRA adapter → 不同 KV
extra_keys = (lora_id,)
block_hash = hash_block_tokens(parent_hash, token_ids, extra_keys)

# 多模态场景：图片 token 的 hash 需要包含图片内容
extra_keys = (image_hash,)
block_hash = hash_block_tokens(parent_hash, token_ids, extra_keys)
```

这保证了：

- 相同 token + 不同 LoRA = 不同 hash（KV 值不同）
- 相同 placeholder token + 不同图片 = 不同 hash（embedding 不同）

### 3.5 `generate_block_hash_extras` 函数

vLLM 提供了一个函数来为每个 block 生成对应的 `extra_keys`：

```python
def generate_block_hash_extras(
    request: "Request",
    start_token_idx: int,
    end_token_idx: int,
    start_mm_idx: int,
) -> tuple[Optional[tuple], int]:
    """为指定范围的 token 生成 extra hash keys。
    
    处理多模态输入中 placeholder token 与实际内容的映射关系。
    返回 (extra_keys, next_mm_idx)
    """
    # 检查该 block 范围内是否包含多模态 placeholder token
    # 如果包含，将多模态内容的 hash 加入 extra_keys
    ...
```

## 4. KV Cache Manager：`kv_cache_manager.py`

### 4.1 `KVCacheManager` 类

这是 APC 的核心管理器，负责 block 的分配、匹配和回收。

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        ...
    ):
        # 物理 block 池
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_gpu_blocks,
            enable_caching=enable_caching,
        )
        
        # 请求 -> 已分配的 block ID 列表
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = {}
        
        # 是否启用 prefix caching
        self.enable_caching = enable_caching
        
        # Block size
        self.block_size = kv_cache_config.block_size
```

### 4.2 `get_computed_blocks`：Cache 命中判定

当新请求到来时，`get_computed_blocks` 判断有多少 block 可以从 cache 中复用：

```python
def get_computed_blocks(
    self, 
    request: "Request",
) -> tuple[list[KVCacheBlock], int]:
    """查找请求的前缀中有多少 block 已经在 cache 中。
    
    Returns:
        computed_blocks: 可以复用的 block 列表
        num_computed_tokens: 可以跳过 prefill 的 token 数
    """
    if not self.enable_caching:
        return [], 0
    
    # 1. 计算请求所有 block 的 hash
    block_hashes = self._compute_block_hashes(request)
    
    # 2. 逐 block 在 block pool 中查找匹配
    computed_blocks = []
    for i, block_hash in enumerate(block_hashes):
        # 在 hash table 中查找
        cached_block = self.block_pool.get_cached_block(block_hash)
        
        if cached_block is None:
            break  # 前缀匹配在此中断
        
        # 验证这是一个完整的 block（非 partial）
        if not cached_block.is_full:
            break
            
        computed_blocks.append(cached_block)
    
    num_computed_tokens = len(computed_blocks) * self.block_size
    return computed_blocks, num_computed_tokens
```

**关键点：** 匹配在第一个 miss 处停止（`break`），保证 exact prefix match 语义。

### 4.3 `allocate_slots`：Block 分配

为请求分配新的 KV cache block，同时复用缓存中已匹配的 block：

```python
def allocate_slots(
    self,
    request: "Request",
    num_new_tokens: int,
    computed_blocks: list[KVCacheBlock],
) -> Optional[list[KVCacheBlock]]:
    """为请求分配 KV cache slot。
    
    Args:
        request: 当前请求
        num_new_tokens: 需要新计算的 token 数
        computed_blocks: 从 cache 复用的 block
    
    Returns:
        新分配的 block 列表，如果显存不足返回 None
    """
    num_required_blocks = cdiv(
        len(request.token_ids), self.block_size
    )
    num_new_blocks = num_required_blocks - len(computed_blocks)
    
    # 检查是否有足够的空闲 block
    if num_new_blocks > self.block_pool.get_num_free_blocks():
        # 尝试驱逐一些缓存 block
        num_evicted = self.block_pool.evict(num_new_blocks)
        if num_evicted < num_new_blocks:
            return None  # 显存不足，无法调度此请求
    
    # 增加复用 block 的引用计数
    for block in computed_blocks:
        self.block_pool.touch(block)  # 更新 LRU 位置
        block.ref_count += 1
    
    # 分配新 block
    new_blocks = self.block_pool.allocate(num_new_blocks)
    
    all_blocks = computed_blocks + new_blocks
    self.req_to_blocks[request.request_id] = all_blocks
    
    return new_blocks
```

### 4.4 `free`：Block 释放

当请求完成时，释放其占用的 block：

```python
def free(self, request: "Request"):
    """释放请求占用的所有 block。"""
    blocks = self.req_to_blocks.pop(request.request_id, [])
    
    for block in blocks:
        block.ref_count -= 1
        
        if block.ref_count == 0:
            if self.enable_caching:
                # 不立即释放，而是标记为可驱逐（进入 LRU 候选）
                self.block_pool.mark_evictable(block)
            else:
                # 不启用 caching 时，立即释放
                self.block_pool.free(block)
```

**关键设计决策：** 当 `enable_caching=True` 时，ref_count 降为 0 的 block 不会立即释放，而是保留在缓存中，等待可能的后续请求复用。只有当显存不足时，才通过 LRU 策略驱逐。

## 5. Block Pool：`block_pool.py`

### 5.1 `BlockPool` 数据结构

```python
class BlockPool:
    """管理物理 KV cache block 的池。"""
    
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
    ):
        # 所有物理 block
        self.num_gpu_blocks = num_gpu_blocks
        self.blocks = [
            KVCacheBlock(block_id=i) for i in range(num_gpu_blocks)
        ]
        
        # 空闲 block 栈（后进先出，提高 cache locality）
        self.free_blocks: list[KVCacheBlock] = list(self.blocks)
        
        # Hash -> Block 映射（核心查找表）
        self.cached_blocks: dict[int, KVCacheBlock] = {}
        
        # LRU 驱逐队列：ref_count == 0 但仍缓存的 block
        self.evictable_blocks: LinkedList[KVCacheBlock] = LinkedList()
        
        self.enable_caching = enable_caching
```

### 5.2 `KVCacheBlock` 数据结构

```python
class KVCacheBlock:
    """一个物理 KV cache block。"""
    
    def __init__(self, block_id: int):
        self.block_id = block_id        # 物理 block ID
        self.ref_count: int = 0          # 引用计数
        self.block_hash: Optional[int] = None  # 内容 hash
        self.is_full: bool = False       # 是否已填满
        self.num_tokens: int = 0         # 已填入的 token 数
        
        # LRU 链表节点指针
        self.evictable_node: Optional[LinkedListNode] = None
```

### 5.3 `get_cached_block`：Hash 查找

```python
def get_cached_block(self, block_hash: int) -> Optional[KVCacheBlock]:
    """通过 hash 查找缓存的 block。"""
    return self.cached_blocks.get(block_hash)
```

时间复杂度 O(1)——Python dict 的哈希表查找。

### 5.4 `allocate`：分配新 Block

```python
def allocate(self, num_blocks: int) -> list[KVCacheBlock]:
    """从空闲池中分配指定数量的 block。"""
    if len(self.free_blocks) < num_blocks:
        raise RuntimeError("Not enough free blocks")
    
    allocated = []
    for _ in range(num_blocks):
        block = self.free_blocks.pop()  # LIFO
        block.ref_count = 1
        block.block_hash = None
        block.is_full = False
        block.num_tokens = 0
        allocated.append(block)
    
    return allocated
```

### 5.5 `register_block_hash`：注册缓存

当一个 block 被完全填充（prefill 完成）后，将其 hash 注册到缓存表中：

```python
def register_block_hash(self, block: KVCacheBlock, block_hash: int):
    """将 block 的 hash 注册到缓存表。"""
    block.block_hash = block_hash
    block.is_full = True
    
    # 如果 hash 已存在（另一个请求刚好同时写入了相同的前缀），
    # 需要处理冲突
    existing = self.cached_blocks.get(block_hash)
    if existing is not None and existing.block_id != block.block_id:
        # 已有相同 hash 的 block，当前 block 可以释放
        # 指向已有的 block（copy-on-write 思想）
        return existing
    
    self.cached_blocks[block_hash] = block
    return block
```

### 5.6 `evict`：LRU 驱逐

```python
def evict(self, num_blocks: int) -> int:
    """驱逐指定数量的缓存 block。
    
    Returns:
        实际驱逐的 block 数量（可能少于请求数）
    """
    num_evicted = 0
    
    while num_evicted < num_blocks and len(self.evictable_blocks) > 0:
        # 从 LRU 队列头部取出最久未使用的 block
        block = self.evictable_blocks.pop_front()
        
        # 从缓存表中移除
        if block.block_hash is not None:
            self.cached_blocks.pop(block.block_hash, None)
        
        # 重置 block 状态
        block.ref_count = 0
        block.block_hash = None
        block.is_full = False
        block.num_tokens = 0
        block.evictable_node = None
        
        # 放回空闲列表
        self.free_blocks.append(block)
        num_evicted += 1
    
    return num_evicted
```

### 5.7 `touch`：更新 LRU 位置

```python
def touch(self, block: KVCacheBlock):
    """将 block 移到 LRU 队列尾部（标记为最近使用）。"""
    if block.evictable_node is not None:
        self.evictable_blocks.remove(block.evictable_node)
        block.evictable_node = None
```

当一个 block 被新请求引用（ref_count > 0）时，它会从 evictable 列表中移除——正在使用的 block 不可被驱逐。

## 6. 端到端流程

### 6.1 请求到达时的完整流程

```
新请求: "You are a helpful assistant.\n\nUser: Hello"
token_ids: [128000, 2675, 527, 264, 11190, 18328, 13, ...]

Step 1: 分 block
  Block 0: (128000, 2675, 527, 264, 11190, 18328, 13, ..., t_15)
  Block 1: (t_16, t_17, ..., t_31)
  Block 2: (t_32, ..., t_39)  ← partial block, 不满

Step 2: 计算 hash
  hash_0 = hash((-1, (128000, 2675, 527, ...)))
  hash_1 = hash((hash_0, (t_16, t_17, ...)))
  hash_2 = 不计算（partial block 不参与 hash 匹配）

Step 3: 查找 cache
  hash_0 → 命中！Block #42 (ref_count: 0 → 1)
  hash_1 → 未命中

Step 4: 分配
  复用:   Block #42 (hash_0 对应的物理 block)
  新分配: Block #99 (用于存储 hash_1 对应的新计算结果)
         Block #100 (用于 partial block)

Step 5: Prefill
  只需要计算 Block 1 和 Block 2 的 token（跳过 Block 0）

Step 6: 注册
  Block #99 计算完成后, register_block_hash(#99, hash_1)
```

### 6.2 请求完成时的流程

```
请求完成:
  Block #42: ref_count 1 → 0, 进入 evictable 列表
  Block #99: ref_count 1 → 0, 进入 evictable 列表
  Block #100: ref_count 1 → 0, 进入 evictable 列表

下一个请求如果有相同前缀:
  hash_0 → 命中 Block #42
  hash_1 → 命中 Block #99
  → 两个 block 都从 evictable 移出，ref_count 恢复为 1
```

## 7. Lookback 窗口与性能优化

### 7.1 为什么需要限制回溯？

在一些场景中（如长对话），token 序列可能非常长（数万 token）。对所有 block 计算 hash 并查找缓存的开销可能很大。vLLM 引入了 lookback 窗口来限制回溯范围。

### 7.2 Lookback 的权衡

```python
# 配置项
max_num_batched_tokens: int = 8192  # 一次 prefill 的最大 token 数

# 如果前缀长度远超 max_num_batched_tokens，
# 即使全部命中 cache，也需要分多个 step 来处理
# vLLM 的调度器会将超长请求拆分为多个 chunk
```

实际上在 vLLM v1 中，lookback 限制与 chunked prefill 机制紧密相关：

- 如果一个请求有 10000 token 的前缀且全部命中 cache，调度器可以跳过 prefill 直接开始 decode
- 如果部分命中（如前 8000 token 命中，后 2000 需要计算），调度器安排一次 2000 token 的 prefill

### 7.3 `--enable-prefix-caching` 参数的代码路径

```python
# vllm/config.py
class CacheConfig:
    def __init__(
        self,
        ...
        enable_prefix_caching: bool = True,  # v1 默认 True
        ...
    ):
        self.enable_prefix_caching = enable_prefix_caching

# vllm/v1/core/kv_cache_manager.py
class KVCacheManager:
    def __init__(self, ..., enable_caching: bool = True):
        self.enable_caching = enable_caching
        # enable_caching 为 False 时:
        # - get_computed_blocks 直接返回空
        # - free 时立即释放 block，不保留缓存
        # - block_pool 不维护 cached_blocks 表
```

## 8. 高级特性

### 8.1 Multi-LoRA 场景下的 Prefix Caching

当使用多个 LoRA adapter 时，相同的 token 在不同 LoRA 下会产生不同的 KV 值。vLLM 通过 `extra_keys` 机制处理：

```python
# 同一个 system prompt，不同 LoRA
# Request A (LoRA #1): hash = hash((parent, tokens, (lora_1,)))
# Request B (LoRA #2): hash = hash((parent, tokens, (lora_2,)))
# → 不同 hash，不会错误地共享 KV cache
```

### 8.2 Speculative Decoding 与 Prefix Caching 的交互

Speculative Decoding 会投机地生成多个 candidate token。如果 candidate 被接受，对应的 KV cache 可以保留；如果被拒绝，需要回退。这与 Prefix Caching 的交互需要特别注意：

- 投机生成的 KV 不应该被注册到缓存中（因为可能被回退）
- 只有确认接受的 token 对应的 block 才能注册 hash

### 8.3 Chunked Prefill 与 Cache 的配合

vLLM v1 支持 Chunked Prefill——将长 prefill 拆分为多个小 chunk。与 Prefix Caching 配合时：

```
请求: 10000 tokens, 前 8000 命中 cache
chunk_size = 2048

Step 1: 发现前 8000 tokens 命中 → num_computed_tokens = 8000
Step 2: 只需 prefill 剩余 2000 tokens（一个 chunk 内完成）
Step 3: 新计算的 block 注册到 cache 中
```

这意味着 Prefix Caching 可以与 Chunked Prefill 协同，大幅减少首次响应延迟（TTFT）。

## 9. 性能影响分析

### 9.1 Cache 命中时的收益

```
不命中: prefill 10000 tokens → TTFT ≈ 2000ms (假设)
命中 8000: prefill 2000 tokens → TTFT ≈ 400ms
→ TTFT 降低 80%
```

### 9.2 Cache 维护的开销

即使在 cache miss 的情况下，APC 也引入了一些开销：

1. **Hash 计算**：对每个 block 计算 hash（Python 层面，微秒级）
2. **Hash 查找**：dict lookup（微秒级）
3. **内存开销**：`cached_blocks` 字典和 LRU 链表
4. **驱逐处理**：当显存紧张时，驱逐逻辑的 CPU 开销

在实践中，这些开销相比 prefill 的 GPU 计算时间可以忽略不计。vLLM 默认启用 APC 正是因为其 overhead 极小。

### 9.3 显存利用率影响

启用 APC 后，一些 ref_count=0 的 block 不会立即释放，而是保留在 GPU 显存中等待复用。这意味着：

- **正面：** 后续相同前缀的请求无需重新计算
- **负面：** 可用于新请求的显存减少，可能导致并发请求数下降

vLLM 通过 LRU 驱逐策略来平衡这一 trade-off：当显存不足时，优先驱逐最久未使用的缓存 block。

---

**下一节：** [SGLang RadixAttention](03-radix-attention.md) — 了解 SGLang 如何用 Radix Tree 实现更灵活的前缀缓存。
