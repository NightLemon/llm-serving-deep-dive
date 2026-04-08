# SGLang RadixAttention

> 本节解读 SGLang 的 RadixAttention 机制——一种基于 Radix Tree（基数树）的 KV Cache 管理方案。与 vLLM 的哈希表方案相比，RadixAttention 天然支持树形前缀共享，在多分支场景（如 tree-of-thought、beam search）中更具优势。

## 1. SGLang 论文概要

### 1.1 论文信息

- **标题：** SGLang: Efficient Execution of Structured Language Model Programs
- **作者：** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. (UC Berkeley)
- **发表：** 2023 年 12 月（arXiv），后续被 NeurIPS 2024 接收
- **核心贡献：**
  1. 提出 SGLang 前端 DSL（Domain-Specific Language），用于编写结构化的 LLM 程序
  2. 提出 RadixAttention——基于 Radix Tree 的 KV Cache 自动复用机制
  3. 提出 compressed finite state machine 用于加速 constrained decoding

### 1.2 核心观察

SGLang 的核心观察是：**现实中的 LLM 应用往往不是单次 API 调用，而是结构化的多步程序**。例如：

```python
# 典型的 LLM 程序：多步推理
@sgl.function
def multi_step_reasoning(s, question):
    s += sgl.system("You are a helpful assistant.")    # 共享前缀 1
    s += sgl.user(question)                              
    s += sgl.assistant(sgl.gen("analysis", max_tokens=200))
    s += sgl.user("Based on your analysis, give a final answer.")
    s += sgl.assistant(sgl.gen("answer", max_tokens=100))
```

```python
# Tree of thought：多分支推理
@sgl.function
def tree_of_thought(s, question):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question)
    
    # 生成 3 个不同的思路（fork）
    forks = s.fork(3)
    for f in forks:
        f += sgl.assistant(sgl.gen("thought", max_tokens=200))
    
    # 每个 fork 共享相同的前缀，但后续生成不同
```

在这些场景中，大量请求共享相同的前缀，且前缀之间存在**树形关系**——多个分支从同一个前缀派生。

### 1.3 为什么哈希表不够好？

vLLM 的 APC 使用 flat hash table：

```
hash(block_0) → physical_block_42
hash(block_1) → physical_block_99
hash(block_2) → physical_block_17
```

这种方案的局限性：

1. **无法表达树形关系**：不知道 block_42、block_99、block_17 之间的从属关系
2. **驱逐时不高效**：驱逐 block_99 后，block_17 的 hash 仍在表中（但实际上已经无效，因为它依赖 block_99 的 parent hash）
3. **查找是线性的**：需要逐 block 查找，每次一次 hash lookup
4. **无法利用共享子树**：如果多个请求共享前 5 个 block，然后在第 6 个 block 分叉，哈希表无法自然表达这种结构

## 2. Radix Tree 数据结构

### 2.1 什么是 Radix Tree？

Radix Tree（基数树，也叫 compressed trie/Patricia trie）是 trie 的压缩版本：

- **Trie**：每条边存储一个字符（或 token）
- **Radix Tree**：连续的单分支路径被压缩为一条边，边上存储多个字符（或 token 序列）

```
Trie 表示 {"abc", "abd", "xyz"}:

     root
    /    \
   a      x
   |      |
   b      y
  / \     |
 c   d    z

Radix Tree 表示：

     root
    /    \
  ab     xyz
  / \
 c   d
```

### 2.2 在 KV Cache 中的应用

在 SGLang 中，Radix Tree 的每个节点存储一段 KV Cache：

```
                    root
                   /    \
        [system prompt]  [另一个 system prompt]
              |                    |
    [user: "Hello"]        [user: "Hi there"]
        /      \
[asst: "Hi!"]  [asst: "Hello!"]
```

每个节点包含：
- **key：** token ID 序列（这段路径的 token）
- **value：** 对应的 KV cache block（GPU 显存中的物理地址）
- **children：** 子节点映射
- **ref_count：** 引用计数
- **last_access_time：** 最后访问时间（用于 LRU 驱逐）

### 2.3 Radix Tree vs Hash Table 对比

| 特性 | Radix Tree (SGLang) | Hash Table (vLLM APC) |
|------|---------------------|----------------------|
| 数据结构 | 树 | 平面哈希表 |
| 前缀查找 | 沿树路径遍历，O(L) | 逐 block hash lookup，O(L/B) |
| 共享表达 | 天然树形共享 | 隐式（通过链式 hash） |
| 插入 | 可能需要节点分裂 | O(1) |
| 驱逐 | 从叶节点回收，保持树完整性 | 从 LRU 队列弹出 |
| 子树共享 | 天然支持 | 不直接支持 |
| 内存开销 | 树节点指针 | 哈希表 bucket |
| 实现复杂度 | 较高（需要处理分裂/合并） | 较低 |

其中 L 是前缀长度（token 数），B 是 block size。

## 3. RadixCache 核心实现

### 3.1 节点定义

```python
class TreeNode:
    """Radix Tree 的节点。"""
    
    def __init__(self):
        self.children: dict[int, TreeNode] = {}  # first_token -> child_node
        self.parent: Optional[TreeNode] = None
        
        # 这个节点存储的 token 序列
        self.token_ids: list[int] = []
        
        # 对应的 KV cache block 索引
        self.kv_indices: list[int] = []  # GPU 显存中的位置
        
        # 引用计数：有多少活跃请求引用此节点
        self.ref_count: int = 0
        
        # 最后访问时间
        self.last_access_time: float = 0.0
```

### 3.2 `match_prefix`：前缀匹配

```python
class RadixCache:
    def __init__(self):
        self.root = TreeNode()
        self.total_size = 0  # 总缓存 token 数
    
    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """在 Radix Tree 中查找最长匹配前缀。
        
        Args:
            token_ids: 要查找的 token 序列
        
        Returns:
            kv_indices: 匹配到的 KV cache 索引
            match_length: 匹配的 token 数
        """
        node = self.root
        matched_kv_indices = []
        pos = 0  # 当前在 token_ids 中的位置
        
        while pos < len(token_ids):
            # 查找下一个匹配的子节点
            first_token = token_ids[pos]
            
            if first_token not in node.children:
                break  # 没有匹配的子节点
            
            child = node.children[first_token]
            
            # 比较 child 节点的 token 序列
            child_len = len(child.token_ids)
            query_segment = token_ids[pos : pos + child_len]
            
            if len(query_segment) < child_len:
                # 查询序列比节点短，只能部分匹配
                # 检查部分匹配的有效性
                if query_segment == child.token_ids[:len(query_segment)]:
                    # 部分匹配——需要在 block 边界处截断
                    num_full_blocks = len(query_segment) // self.block_size
                    num_matched = num_full_blocks * self.block_size
                    matched_kv_indices.extend(
                        child.kv_indices[:num_matched]
                    )
                    pos += num_matched
                break
            
            if list(query_segment) != child.token_ids:
                # 不匹配（前几个 token 相同但中间有分歧）
                # 找到分歧点
                diverge = 0
                while diverge < child_len and \
                      token_ids[pos + diverge] == child.token_ids[diverge]:
                    diverge += 1
                
                # 只匹配到分歧点之前的完整 block
                num_full_blocks = diverge // self.block_size
                num_matched = num_full_blocks * self.block_size
                matched_kv_indices.extend(child.kv_indices[:num_matched])
                pos += num_matched
                break
            
            # 完全匹配当前子节点
            matched_kv_indices.extend(child.kv_indices)
            pos += child_len
            node = child
        
        return matched_kv_indices, pos
```

### 3.3 `insert`：插入新前缀

```python
def insert(self, token_ids: list[int], kv_indices: list[int]):
    """将新的 token 序列及其 KV cache 索引插入 Radix Tree。
    
    如果前缀已存在，只需延伸或添加分支。
    """
    node = self.root
    pos = 0
    
    while pos < len(token_ids):
        first_token = token_ids[pos]
        
        if first_token not in node.children:
            # 创建新的子节点
            new_node = TreeNode()
            new_node.token_ids = token_ids[pos:]
            new_node.kv_indices = kv_indices[pos:]
            new_node.parent = node
            node.children[first_token] = new_node
            self.total_size += len(new_node.token_ids)
            return
        
        child = node.children[first_token]
        child_len = len(child.token_ids)
        remaining = token_ids[pos:]
        
        # 找到公共前缀长度
        common_len = 0
        while common_len < child_len and \
              common_len < len(remaining) and \
              child.token_ids[common_len] == remaining[common_len]:
            common_len += 1
        
        if common_len < child_len:
            # 需要分裂当前节点
            # child: [A B C D E]
            # 新序列: [A B X Y Z]
            # 分裂为:
            #   [A B] → child_common
            #     ├── [C D E] → child_suffix (原来的 child)
            #     └── [X Y Z] → new_suffix (新分支)
            
            # 创建公共前缀节点
            common_node = TreeNode()
            common_node.token_ids = child.token_ids[:common_len]
            common_node.kv_indices = child.kv_indices[:common_len]
            common_node.parent = node
            
            # 修改原 child 为后缀
            child.token_ids = child.token_ids[common_len:]
            child.kv_indices = child.kv_indices[common_len:]
            child.parent = common_node
            common_node.children[child.token_ids[0]] = child
            
            # 替换父节点的引用
            node.children[first_token] = common_node
            
            # 如果新序列还有剩余，创建新分支
            if common_len < len(remaining):
                new_suffix = TreeNode()
                new_suffix.token_ids = list(remaining[common_len:])
                new_suffix.kv_indices = list(kv_indices[pos + common_len:])
                new_suffix.parent = common_node
                common_node.children[remaining[common_len]] = new_suffix
                self.total_size += len(new_suffix.token_ids)
            
            return
        
        # 完全匹配当前节点，继续向下
        pos += child_len
        node = child
    
    # token_ids 是某个已有节点的前缀
    # 可能需要分裂节点
```

### 3.4 `evict`：驱逐策略

SGLang 的驱逐策略基于 **叶节点优先** + **LRU**：

```python
def evict(self, num_tokens_to_evict: int) -> int:
    """驱逐指定数量的 token 对应的 KV cache。
    
    策略：从叶节点开始，按 LRU 顺序驱逐。
    只有 ref_count == 0 的节点才能被驱逐。
    
    Returns:
        实际驱逐的 token 数
    """
    evicted = 0
    
    # 收集所有可驱逐的叶节点
    leaves = self._collect_evictable_leaves()
    
    # 按 last_access_time 排序（最久未使用的优先）
    leaves.sort(key=lambda n: n.last_access_time)
    
    for leaf in leaves:
        if evicted >= num_tokens_to_evict:
            break
        
        if leaf.ref_count > 0:
            continue  # 正在使用，不可驱逐
        
        # 释放 KV cache 显存
        num_freed = len(leaf.kv_indices)
        self._free_kv_indices(leaf.kv_indices)
        
        # 从树中移除叶节点
        parent = leaf.parent
        del parent.children[leaf.token_ids[0]]
        
        # 如果父节点只剩一个子节点，合并（保持 radix tree 的压缩特性）
        if len(parent.children) == 1 and parent != self.root \
           and parent.ref_count == 0:
            self._merge_with_child(parent)
        
        self.total_size -= num_freed
        evicted += num_freed
    
    return evicted

def _collect_evictable_leaves(self) -> list[TreeNode]:
    """收集所有 ref_count == 0 的叶节点。"""
    leaves = []
    stack = [self.root]
    
    while stack:
        node = stack.pop()
        if not node.children:
            # 叶节点
            if node.ref_count == 0 and node != self.root:
                leaves.append(node)
        else:
            for child in node.children.values():
                stack.append(child)
    
    return leaves
```

**关键设计：叶节点优先驱逐**。这确保了共享的前缀（内部节点）尽可能保留，只有当所有依赖某个前缀的分支都被驱逐后，该前缀本身才可能被驱逐。

### 3.5 可视化示例

考虑以下场景——三个请求共享同一个 system prompt：

```
请求 A: "You are a helpful assistant. User: What is AI?"
请求 B: "You are a helpful assistant. User: What is ML?"
请求 C: "You are a coding assistant. User: Write hello world"

初始状态（空树）:
  root

插入请求 A 后:
  root
   └── "You are a helpful assistant. User: What is AI?" [KV: 0-49]

插入请求 B 后（与 A 有共同前缀）:
  root
   └── "You are a helpful assistant. User: What is " [KV: 0-39]
        ├── "AI?" [KV: 40-42]   ← 请求 A 的分支
        └── "ML?" [KV: 43-45]   ← 请求 B 的分支

插入请求 C 后（前缀 "You are a " 与前两个共享）:
  root
   └── "You are a " [KV: 0-9]
        ├── "helpful assistant. User: What is " [KV: 10-39]
        │    ├── "AI?" [KV: 40-42]
        │    └── "ML?" [KV: 43-45]
        └── "coding assistant. User: Write hello world" [KV: 46-80]
```

树的结构自然反映了请求之间的前缀共享关系。

## 4. 与 vLLM APC 的深度对比

### 4.1 匹配效率

**vLLM APC：**
- 每个 block 需要一次 hash 计算 + 一次 dict lookup
- $n$ 个 block 的前缀需要 $n$ 次 lookup
- 所有 lookup 都是 O(1)，总复杂度 O(n)

**SGLang RadixAttention：**
- 沿树路径遍历，每个内部节点需要一次 dict lookup + 一次序列比较
- 树的深度通常远小于 block 数
- 但每个节点的序列比较是 O(m)，m 为节点 token 数

在实践中，两者的查找性能差异不大，因为查找本身不是瓶颈（相比 GPU 计算）。

### 4.2 共享模式

**vLLM APC 擅长的场景：**
- 大量请求共享同一个 system prompt（一条直线前缀）
- 简单的前缀复用模式

**SGLang RadixAttention 擅长的场景：**
- Tree-of-thought：从同一前缀分叉出多个推理路径
- Multi-turn 对话：不同用户的对话在 system prompt 处合并
- Fork-join 模式：先分叉探索，再汇总
- Few-shot 场景：不同的 few-shot example 组合

```
vLLM APC 视角（flat hash table）:
  hash_A → block_1
  hash_B → block_2
  hash_C → block_3
  （看不出 block 之间的关系）

SGLang RadixAttention 视角（tree）:
  root → [system] → [user_1] → [asst_1]
                  → [user_2] → [asst_2]
                  → [user_3] → [asst_3]
  （树形结构清晰表达共享关系）
```

### 4.3 驱逐策略差异

**vLLM：** 全局 LRU，从 evictable 列表头部驱逐。可能驱逐一个中间 block，导致后续 block 的 hash 依赖断裂（但不会出错，因为 hash 是链式的，后续查找自然会 miss）。

**SGLang：** 叶节点优先 + LRU。保证共享的前缀（内部节点）尽可能保留，驱逐从"分支末端"开始，最大化共享收益。

这是 SGLang 在 multi-branch 场景下的一个关键优势。

### 4.4 实际性能对比

根据 SGLang 论文和社区 benchmark（2024-2025 数据）：

| 场景 | vLLM APC | SGLang RadixAttention |
|------|----------|----------------------|
| 单前缀共享（chatbot） | 基本持平 | 基本持平 |
| Tree-of-thought | 较差（无法表达分支） | 显著优势 |
| Multi-turn 对话（共享 system prompt） | 良好 | 良好 |
| 高并发混合负载 | 良好 | 良好 |
| Few-shot 动态组合 | 一般 | 较好 |

在最常见的 chatbot 场景中，两者性能接近。SGLang 的优势主要体现在**结构化 LLM 程序**中。

## 5. SGLang 的高级特性

### 5.1 Cache-Aware Scheduling

SGLang 在调度时会考虑 cache 状态。调度器优先选择 **cache hit rate 最高的请求**执行：

```python
def schedule_next_batch(self, waiting_queue):
    """Cache-aware 调度：优先选择 cache 命中最多的请求。"""
    candidates = []
    
    for request in waiting_queue:
        # 预查找 cache 命中情况
        _, match_length = self.radix_cache.match_prefix(
            request.token_ids
        )
        cache_hit_ratio = match_length / len(request.token_ids)
        candidates.append((request, cache_hit_ratio))
    
    # 按 cache hit ratio 降序排列
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 选择 cache hit 最高的请求优先执行
    batch = candidates[:self.max_batch_size]
    return [req for req, _ in batch]
```

这种调度策略可以显著提高系统吞吐——优先执行 cache 命中高的请求意味着这些请求的 prefill 更短，更快完成并释放资源。

### 5.2 与 Continuous Batching 的配合

SGLang 的 Radix Cache 与 continuous batching 紧密配合：

1. **Prefill 阶段：** 只计算未命中 cache 的 token，减少计算量
2. **Decode 阶段：** 新生成的 token 扩展树的分支
3. **请求完成：** ref_count 减少，叶节点变为可驱逐
4. **新请求到达：** 先查树找最长匹配，再调度 prefill

### 5.3 多模态支持

SGLang 同样支持多模态输入的前缀缓存。对于包含图像的请求，图像 embedding 的 hash 被用作树节点的一部分。这保证了相同图片 + 相同文本前缀才能复用 KV cache。

## 6. 实现权衡与局限性

### 6.1 树的维护成本

Radix Tree 的分裂和合并操作比哈希表的 insert/delete 更复杂：

- **分裂（split）：** 当新前缀与现有节点部分匹配时，需要创建新的内部节点
- **合并（merge）：** 驱逐叶节点后，如果内部节点只剩一个子节点，需要合并
- **并发控制：** 多线程环境下需要对树的修改加锁

vLLM 的哈希表方案在并发场景下更容易实现（dict 操作天然线程安全，或用简单的锁即可）。

### 6.2 内存碎片

Radix Tree 的每个节点可能持有不同长度的 KV cache 片段，导致 GPU 显存碎片化。SGLang 通过将 KV cache 仍然以 block 为单位分配来缓解这个问题——树节点只存储 block 索引，实际的 KV 数据仍然在固定大小的 block 中。

### 6.3 冷启动性能

两种方案在冷启动（cache 为空）时性能相同——都需要完整 prefill。差异只在 cache 建立之后的后续请求中体现。

## 7. 选择建议

| 选择 | 推荐场景 |
|------|---------|
| vLLM APC | 标准 chatbot 服务、简单的前缀共享、对实现简单性有要求 |
| SGLang RadixAttention | 结构化 LLM 程序、tree-of-thought、fork-join 模式、研究探索 |
| 两者皆可 | 大多数生产场景（性能差异不大） |

在实际生产部署中，选择哪个框架通常取决于**整体特性集**（如 SGLang 的前端 DSL、vLLM 的广泛模型支持等），而非单纯的 prefix caching 实现差异。

---

**下一节：** [API 提供商 Prompt Caching 实践](04-api-caching.md) — 了解 OpenAI、Anthropic、Google 等 API 提供商如何实现和暴露 Prompt Caching 能力。
