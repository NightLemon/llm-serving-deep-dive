# Hybrid KV Cache Manager

> 当模型不再只有 Transformer 层——混合架构如何颠覆 KV Cache 管理的基本假设

## 1. 问题背景：为什么需要 Hybrid KV Cache？

传统 LLM serving 系统（如 vLLM v0）有一个隐含假设：**模型的每一层都是标准的 Multi-Head Attention（MHA），每一层都需要相同大小和格式的 KV Cache**。在这个假设下，KV Cache Manager 只需要一种分配策略——为每个 sequence 的每一层分配固定大小的 KV block。

但 2024-2025 年涌现的混合架构模型打破了这一假设：

| 模型 | 架构 | KV Cache 需求 |
|------|------|---------------|
| Jamba (AI21, 2024) | Transformer + Mamba 交替层 | Transformer 层需要 KV Cache；Mamba 层需要 SSM state（固定大小） |
| Zamba (Zyphra, 2024) | Mamba + 共享 Attention 层 | 少数 Attention 层需要 KV Cache；大部分 Mamba 层需要 state |
| Jamba-1.5 (AI21, 2024) | 改进的 Transformer-Mamba 混合 | 不同 layer group 有不同的 attention 配置 |
| DeepSeek-V2/V3 | MLA (Multi-head Latent Attention) | KV Cache 维度远小于标准 MHA |
| Mixtral-style MoE | 标准 Attention + MoE FFN | Attention 标准，但 FFN 路由影响计算 |

### 核心挑战

```
标准 Transformer (e.g., LLaMA-70B):
Layer 0:  [MHA] → KV Cache: (2, num_heads, head_dim) per token
Layer 1:  [MHA] → KV Cache: (2, num_heads, head_dim) per token
...
Layer 79: [MHA] → KV Cache: (2, num_heads, head_dim) per token
→ 每层相同，可以用统一的 block manager

Jamba-style 混合模型:
Layer 0:  [Mamba]     → SSM State: (d_state, d_inner), 固定大小，与 seq_len 无关
Layer 1:  [Attention] → KV Cache: (2, num_heads, head_dim) per token
Layer 2:  [Mamba]     → SSM State
Layer 3:  [Mamba]     → SSM State
Layer 4:  [Attention] → KV Cache
...
→ 不同层需要不同的缓存策略！
```

这意味着 KV Cache Manager 需要做到：

1. **识别不同层的类型**：哪些层需要 paged KV cache，哪些需要固定大小的 state
2. **独立管理不同类型的缓存**：为 Attention 层分配 paged blocks，为 Mamba 层分配 state buffer
3. **协调分配决策**：一个 sequence 能否被调度，取决于所有类型的缓存是否都有空间

## 2. vLLM v1 Hybrid KV Cache 架构

vLLM v1 引入了 **KV Cache Coordinator** 来解决这一问题。其核心设计思路是**将不同类型的层分组，每组使用独立的 KV Cache Manager，再由 Coordinator 统一协调**。

### 2.1 架构总览

```
┌─────────────────────────────────────────────────┐
│                 KVCacheCoordinator              │
│  ┌───────────────────────────────────────────┐  │
│  │  统一接口：allocate / free / get_num_free  │  │
│  └────────────────┬──────────────────────────┘  │
│                   │                              │
│    ┌──────────────┼──────────────┐               │
│    ▼              ▼              ▼               │
│ ┌────────┐  ┌────────────┐  ┌────────────┐      │
│ │Group 0 │  │  Group 1   │  │  Group 2   │      │
│ │FullAttn│  │SlidingWin  │  │  Mamba     │      │
│ │Manager │  │  Manager   │  │  Manager   │      │
│ └────────┘  └────────────┘  └────────────┘      │
│  Layer 0,4    Layer 1,3,5    Layer 2,6,7        │
│  Layer 8,12   Layer 9,11     Layer 10,13        │
└─────────────────────────────────────────────────┘
```

### 2.2 Layer Group 的概念

**Layer Group** 是具有相同 KV Cache 特征的层的集合。同一个 group 内的层共享相同的：

- **Cache 类型**：Full attention / Sliding window attention / SSM state / 无需缓存
- **KV head 数量和维度**：影响每个 block 的大小
- **Block size**：每个 block 存储多少 token 的 KV

```python
# 概念模型：Layer Group 定义
@dataclass
class LayerGroup:
    group_id: int
    layer_indices: List[int]        # 属于此 group 的层编号
    cache_type: CacheType           # FULL_ATTENTION / SLIDING_WINDOW / SSM_STATE / NONE
    num_kv_heads: int               # KV head 数量
    head_dim: int                   # 每个 head 的维度
    block_size: int                 # 每个 block 的 token 数
    sliding_window_size: Optional[int]  # 滑动窗口大小（如果适用）
```

### 2.3 核心设计决策

**决策 1：每个 group 独立的 free block pool**

不同 group 使用各自的 GPU 显存区域和 free block 列表。这避免了不同大小的 block 互相干扰。

**决策 2：分配时取最小可用**

Coordinator 在分配时，检查所有 group 的可用 block 数，**取最小值作为系统的可用容量**。这保证了一个 sequence 在所有 group 中都有足够的空间。

**决策 3：统一的分配/释放接口**

上层 scheduler 不需要知道底层有多少个 group，只通过 Coordinator 的统一接口操作。

## 3. 源码分析：`kv_cache_coordinator.py`

> 基于 vLLM v1 架构，源码路径：`vllm/v1/core/kv_cache_coordinator.py`

### 3.1 KVCacheCoordinator 类

```python
class KVCacheCoordinator:
    """
    协调多个 KV Cache Manager，为混合架构模型提供统一的缓存管理接口。
    
    核心职责:
    1. 维护多个 group 各自的 KVCacheManager
    2. 在分配时协调各 group，确保一致性
    3. 提供统一的 can_allocate / allocate / free 接口
    """
    
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        ...
    ):
        # 为每个 layer group 创建独立的 KVCacheManager
        self.managers: List[KVCacheManager] = []
        for group in kv_cache_config.groups:
            manager = KVCacheManager(
                num_blocks=group.num_blocks,
                block_size=group.block_size,
                max_model_len=max_model_len,
                ...
            )
            self.managers.append(manager)
```

### 3.2 可用容量计算

```python
def get_num_free_blocks(self) -> int:
    """
    返回系统可用的 block 数量。
    
    关键：取所有 group 的最小值。
    因为一个 sequence 需要在每个 group 都能分配 block，
    所以系统的瓶颈是最紧张的那个 group。
    """
    return min(
        manager.get_num_free_blocks() 
        for manager in self.managers
    )
```

这个 `min()` 操作看似简单，但蕴含着重要的设计权衡：

- **优点**：保证分配一定成功（如果 Coordinator 报告有 N 个 free blocks，那每个 group 至少有 N 个）
- **缺点**：可能低估了某些 group 的可用空间，导致显存利用率下降
- **改进空间**：当不同 group 的 block 大小不同时（例如 SSM state block vs KV cache block），简单取 min 可能不合理——需要按 token 数而非 block 数来比较

### 3.3 分配流程

```python
def allocate(self, request_id: str, num_tokens: int) -> bool:
    """
    为一个 request 在所有 group 中分配缓存空间。
    
    流程:
    1. 先检查每个 group 是否都有足够空间
    2. 如果全部满足，逐个 group 执行分配
    3. 如果任一 group 空间不足，分配失败（不做部分分配）
    """
    # Step 1: 预检查（all-or-nothing 语义）
    for manager in self.managers:
        if not manager.can_allocate(num_tokens):
            return False
    
    # Step 2: 执行分配
    for manager in self.managers:
        manager.allocate(request_id, num_tokens)
    
    return True
```

**All-or-nothing 语义**至关重要：我们不能出现 "Group 0 分配成功但 Group 1 失败" 的中间状态，否则需要复杂的回滚逻辑。

### 3.4 释放流程

```python
def free(self, request_id: str):
    """释放一个 request 在所有 group 中的缓存。"""
    for manager in self.managers:
        manager.free(request_id)
```

### 3.5 与 Scheduler 的交互

```python
# 在 Scheduler 中的使用（简化）
class Scheduler:
    def __init__(self, ...):
        self.kv_cache_coordinator = KVCacheCoordinator(...)
    
    def schedule(self) -> SchedulerOutput:
        # Scheduler 完全不需要知道底层有几个 group
        num_free = self.kv_cache_coordinator.get_num_free_blocks()
        
        for request in self.waiting_queue:
            tokens_needed = request.num_tokens
            if self.kv_cache_coordinator.allocate(request.id, tokens_needed):
                scheduled.append(request)
            else:
                break  # 空间不足，停止调度
```

## 4. 混合架构模型的具体案例

### 4.1 Jamba：Transformer-Mamba 混合

Jamba (AI21 Labs, 2024) 是最早的商业级 Transformer-Mamba 混合模型之一。其架构特点：

```
Jamba 架构 (52B 参数):
┌─────────────────────────────────┐
│ Block 1: Mamba × 6 + Attn × 1  │  ← 每 7 层中 1 层是 Attention
│ Block 2: Mamba × 6 + Attn × 1  │
│ Block 3: Mamba × 6 + Attn × 1  │
│ ...                             │
│ Block 8: Mamba × 6 + Attn × 1  │
└─────────────────────────────────┘

KV Cache 需求分析:
- Attention 层 (8 层): 需要标准 KV Cache, 随 seq_len 线性增长
- Mamba 层 (48 层): 需要 SSM State, 固定大小 (不随 seq_len 增长!)
```

**Mamba SSM State 的关键特性**：

与 KV Cache 不同，Mamba 的 SSM state 是**固定大小**的，不随序列长度增长：

```python
# Mamba SSM State 结构 (每层, 每个 sequence)
ssm_state = torch.zeros(
    batch_size,
    d_inner,     # 通常是 d_model 的 2 倍
    d_state,     # SSM 状态维度, 通常 16
    dtype=torch.float32
)
# 大小: batch_size × d_inner × d_state
# 例: 1 × 8192 × 16 = 128K 个 float32 = 512KB per layer per sequence
# 重要: 无论 seq_len 是 100 还是 100,000, 大小不变!
```

这对 KV Cache Manager 的影响：

1. **不需要 paged allocation**：SSM state 大小固定，直接预分配即可
2. **不需要 block table**：没有 "哪个 token 在哪个 block" 的概念
3. **显存效率极高**：超长序列下 Mamba 层的缓存开销几乎可以忽略

### 4.2 Zamba：以 Mamba 为主体的混合架构

Zamba (Zyphra, 2024) 走得更远——它只有 **1-2 个共享的 Attention 层**穿插在大量 Mamba 层之间：

```
Zamba 架构:
Layer 0-5:   Mamba × 6
Layer 6:     Shared Attention (唯一的 attention 层, 被多处复用)
Layer 7-12:  Mamba × 6
Layer 13:    Shared Attention (同一组权重!)
Layer 14-19: Mamba × 6
...
```

**Shared Attention 的 KV Cache 管理挑战**：

```python
# Shared Attention 意味着同一组 KV 会被多个位置使用
# 两种处理策略:

# 策略 1: 每次调用独立计算 KV (不缓存)
# 优点: 简单, 不需要跨层 cache 管理
# 缺点: 重复计算, 浪费算力

# 策略 2: 缓存 KV, 多层共享
# 优点: 计算效率高
# 缺点: 需要 reference counting, 确保所有使用者都完成后再释放
```

### 4.3 对 vLLM KV Cache Coordinator 的要求

混合架构对 Coordinator 提出了以下额外要求：

```python
# 需求 1: 支持 SSM State 类型的 "cache"
class CacheType(Enum):
    FULL_ATTENTION = "full_attention"          # 标准 MHA/GQA
    SLIDING_WINDOW = "sliding_window"          # 滑动窗口 attention
    SSM_STATE = "ssm_state"                    # Mamba SSM state
    CROSS_ATTENTION = "cross_attention"        # 编码器-解码器交叉注意力
    NONE = "none"                              # 不需要缓存的层

# 需求 2: 不同 cache type 的 manager 实现不同
class SSMStateManager:
    """为 SSM State 设计的 manager, 不使用 paging"""
    def allocate(self, request_id: str) -> int:
        # 分配固定大小的 state buffer (不依赖 num_tokens)
        slot_id = self.free_slots.pop()
        self.request_to_slot[request_id] = slot_id
        return slot_id
    
    def get_num_free_blocks(self) -> int:
        # 对于 SSM State, "block" 的概念是 "slot"
        # 每个 slot = 一个固定大小的 state buffer
        return len(self.free_slots)
```

## 5. 不同 Attention 类型的 Cache 策略

除了 Transformer-SSM 混合架构，即使在纯 Transformer 模型中，不同类型的 attention 也需要不同的 cache 策略：

### 5.1 Full Attention vs Sliding Window Attention

Mistral 系列模型在不同层使用不同的 attention 模式：

```
Mixtral 8x22B:
Layer 0-7:   Full Attention (需要缓存所有历史 token)
Layer 8-15:  Sliding Window Attention (只需缓存最近 4096 个 token)
Layer 16-23: Full Attention
Layer 24-31: Sliding Window Attention
...
```

对 KV Cache Manager 的影响：

```python
# Full Attention 层: 需要保留所有 token 的 KV
# 显存使用: O(seq_len)

# Sliding Window 层: 只保留最近 W 个 token 的 KV
# 显存使用: O(W), 上限固定
# 可以循环使用 block: 当新 token 进来, 覆盖最旧的 block
```

**Sliding Window 的 Block 回收优化**：

```python
class SlidingWindowManager(KVCacheManager):
    def __init__(self, window_size: int, block_size: int, ...):
        self.window_size = window_size
        self.max_blocks_per_seq = math.ceil(window_size / block_size)
    
    def append_token(self, request_id: str):
        blocks = self.request_blocks[request_id]
        if len(blocks) >= self.max_blocks_per_seq:
            # 回收最旧的 block，循环使用
            oldest_block = blocks.pop(0)
            self.free_block(oldest_block)
        # 如果当前 block 满了，分配新 block
        if self.current_block_full(request_id):
            new_block = self.allocate_block()
            blocks.append(new_block)
```

### 5.2 GQA / MQA 的 Cache 大小差异

即使都是 Full Attention，不同的 attention 变体 KV Cache 大小也不同：

```
LLaMA-3 70B (GQA, 8 KV heads):
  KV per layer per token = 2 × 8 × 128 = 2048 bytes (FP16)

DeepSeek-V3 (MLA, 压缩后):
  KV per layer per token = 512 bytes (压缩维度)

Falcon-180B (MQA, 1 KV head):  
  KV per layer per token = 2 × 1 × 128 = 256 bytes (FP16)
```

当一个模型的不同层使用不同的 attention 方案时，block 大小会不同，必须用不同的 manager。

## 6. Prefix Caching 在混合架构下的挑战

Prefix caching（前缀缓存）在混合架构下面临独特的挑战：

### 6.1 Attention 层：可以复用

Attention 层的 KV Cache 是 prompt token 的函数，**相同前缀产生相同 KV**，因此可以在 request 间共享。这与标准 Transformer 模型完全一样。

### 6.2 Mamba 层：不能简单复用

Mamba 的 SSM State 在处理完一个序列前缀后，虽然也是确定性的（相同前缀 → 相同 state），但有两个问题：

1. **SSM State 不可分割**：KV Cache 可以 per-token 共享（共享前 N 个 token 的 cache），但 SSM State 是一个整体——你不能 "共享前半部分 state"
2. **增量更新不同**：KV Cache 追加新 token 只需要追加新的 KV 对；SSM State 追加新 token 需要在已有 state 上做矩阵运算，修改整个 state

```python
# KV Cache 的 prefix sharing:
# Request A: [sys_prompt, user_msg_A] → KV Cache = [KV_sys, KV_user_A]
# Request B: [sys_prompt, user_msg_B] → 共享 KV_sys, 只计算 KV_user_B
# → 节省了 sys_prompt 的 prefill 计算

# SSM State 的 prefix sharing:
# Request A: [sys_prompt, user_msg_A] → State_A = f(State_sys, user_msg_A)
# Request B: [sys_prompt, user_msg_B] → 需要 copy State_sys, 然后更新
# → 需要完整复制 state, 不能像 KV Cache 一样只引用
```

### 6.3 Coordinator 的 Prefix Caching 策略

```python
class HybridPrefixCacheCoordinator:
    def try_reuse_prefix(self, request_id: str, token_ids: List[int]):
        # 对 Attention group: 使用标准 prefix caching
        prefix_len = self.attention_manager.find_cached_prefix(token_ids)
        
        # 对 SSM group: 查找完全匹配的 state checkpoint
        state_checkpoint = self.ssm_manager.find_checkpoint(
            token_ids[:prefix_len]
        )
        
        if state_checkpoint is not None:
            # 两者都能复用 → 从 prefix_len 处继续计算
            self.ssm_manager.restore_state(request_id, state_checkpoint)
            return prefix_len
        else:
            # SSM state 没有缓存 → 需要从头计算 SSM 部分
            # 但 Attention 部分仍然可以复用!
            # 这是一个部分复用的场景
            return 0  # 保守策略: 不复用, 全部重算
```

## 7. 显存分配策略

### 7.1 静态预分配 vs 动态分配

```python
# SSM State: 适合静态预分配
# 理由: 大小固定, 与 seq_len 无关, 可以预先分配 max_batch_size 份
ssm_states = torch.zeros(
    max_batch_size, num_mamba_layers, d_inner, d_state,
    device='cuda', dtype=torch.float32
)

# KV Cache: 需要动态分配 (PagedAttention)
# 理由: 大小随 seq_len 增长, 不同 request 长度不同
kv_blocks = torch.zeros(
    num_blocks, block_size, 2, num_kv_heads, head_dim,
    device='cuda', dtype=torch.float16
)
```

### 7.2 显存预算分割

Coordinator 需要在初始化时决定如何分割 GPU 显存：

```python
def compute_memory_split(
    total_gpu_memory: int,
    model_memory: int,
    layer_groups: List[LayerGroup],
) -> Dict[int, int]:  # group_id → num_blocks
    """
    将可用显存按比例分配给各个 group。
    
    策略: 按每个 group 的 "每 token 显存需求" 比例分配。
    """
    available = total_gpu_memory - model_memory - overhead
    
    # 计算每个 group 单 token 的显存需求
    per_token_memory = {}
    for group in layer_groups:
        if group.cache_type == CacheType.SSM_STATE:
            # SSM: 固定大小, 按 max_batch_size 预留
            per_token_memory[group.group_id] = 0  # 不按 token 计
        else:
            # Attention: 按 token 计算
            per_token_memory[group.group_id] = (
                2 * group.num_kv_heads * group.head_dim * 
                len(group.layer_indices) * dtype_size
            )
    
    # 先预留 SSM state 空间
    ssm_reserved = sum(
        group.state_size * max_batch_size 
        for group in layer_groups 
        if group.cache_type == CacheType.SSM_STATE
    )
    
    # 剩余空间按 attention group 的需求比例分配
    remaining = available - ssm_reserved
    # ... 比例分配逻辑
```

## 8. 实践建议

### 8.1 选择混合架构模型时的考量

| 考量因素 | Transformer-only | Mamba-only | 混合架构 |
|----------|-----------------|------------|----------|
| 长序列显存 | O(n) per layer | O(1) per layer | 取决于比例 |
| Serving 复杂度 | 低（成熟工具链） | 中（工具链不完善） | 高（需要 hybrid manager） |
| Prefix caching | 完全支持 | 不适用 | 部分支持 |
| 生态支持 | vLLM, SGLang, TRT-LLM | 有限 | vLLM v1 开始支持 |
| 推荐场景 | 通用 | 超长序列，低延迟 | 平衡长序列和质量 |

### 8.2 调优建议

1. **显存分配比例**：对于 Jamba 类模型（~15% Attention 层），大部分可用显存可分配给 Attention 层的 KV Cache
2. **Block size 选择**：Attention 层仍然使用标准的 block size（如 16）；SSM state 不需要分块
3. **Batch size 限制**：由于 SSM state 需要预分配，`max_batch_size` 需要考虑 SSM state 的显存开销
4. **Prefix caching 策略**：如果 prompt 复用率高，建议缓存 SSM state checkpoints（以空间换时间）

### 8.3 未来方向

- **统一的 Cache 抽象**：将 KV Cache 和 SSM State 统一为 "Layer Cache" 抽象，简化上层逻辑
- **自适应分配**：根据运行时负载动态调整各 group 的显存比例
- **跨层 Cache 共享**：利用混合架构中 Attention 层 KV 的相似性做跨层共享
- **硬件协同设计**：SSM State 的读写模式与 KV Cache 不同，可能需要不同的内存层级策略

## 9. 小结

Hybrid KV Cache Manager 是 serving 系统适应模型架构多样化的关键能力。核心要点：

1. **分组管理**：将不同类型的层分组（Layer Group），每组用最适合的 cache 策略
2. **协调分配**：Coordinator 确保 all-or-nothing 语义，避免部分分配导致的不一致
3. **最小可用原则**：系统可用容量取决于最紧张的 group
4. **SSM State 特殊性**：固定大小、不可分割、prefix sharing 受限
5. **显存预算分割**：需要在 SSM state 预分配和 KV Cache 动态分配间平衡

随着 Jamba、Zamba 等混合架构模型逐渐成熟，Hybrid KV Cache Manager 将从 "前沿特性" 变为 serving 系统的标准能力。vLLM v1 的 `KVCacheCoordinator` 提供了一个清晰的参考实现，值得深入研究。
