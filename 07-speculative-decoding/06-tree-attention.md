# Tree Attention

> 投机解码生成的候选 token 通常不是一条线性序列，而是一棵**树**——多条候选路径共享公共前缀。Tree Attention 是高效验证这棵候选树的关键机制：它允许 target model 在一次 forward pass 中同时评估所有路径，通过精心构造的 attention mask 确保每个 token 只 attend 到其合法的祖先节点。

## 1. 为什么需要 Tree Attention

### 1.1 从线性到树形

最简单的投机解码生成一条线性的 draft 序列（$\gamma$ 个 token），target model 验证时按顺序处理。但这种方式有一个问题：**如果第一个 draft token 就被拒绝，后续所有 draft token 都浪费了**。

树形结构的解决思路：在每个位置考虑多个候选 token（top-k），形成一棵候选树，target model 一次性验证所有路径。

```
线性 draft (γ=4):
prefix → a → b → c → d
如果 a 被拒绝 → b, c, d 全部浪费

树形 draft:
                ┌── b₁ → c₁
        ┌── a₁─┤
        │       └── b₂ → c₂
prefix ─┤
        │       ┌── b₃
        └── a₂─┤
                └── b₄

如果 a₁ 被拒绝 → 还可以尝试 a₂ 路径
如果 a₁b₁ 被接受但 c₁ 被拒绝 → 还可以尝试 c₂
```

### 1.2 候选树的优势

设树中有 $N$ 个节点，最长路径长度为 $D$：

- **线性 draft**：$N = D = \gamma$，只有一条路径
- **树形 draft**：$N > D$，多条路径，增加了至少一条被完全接受的概率

**定量分析**：假设每个位置的 token-level 接受率为 $\alpha$，线性 draft 接受 $k$ 个 token 的概率为 $\alpha^k (1-\alpha)$。而树形 draft 中，如果深度为 $d$ 的节点有 $b$ 个分支（branching factor），则深度 $d$ 处至少一个被接受的概率为 $1 - (1-\alpha)^b$，显著高于单条路径的 $\alpha$。

### 1.3 验证的计算约束

树中有 $N$ 个候选 token。如果逐条路径验证，需要多次 target model forward pass——失去了投机解码的意义。

Tree Attention 的核心目标：**在一次 forward pass 中同时验证所有 $N$ 个候选 token**，开销约等于处理一个长度为 $N$ 的序列。

## 2. Tree Attention Mask 的构建

### 2.1 基本原理

在标准 causal attention 中，位置 $i$ 可以 attend 到所有 $j \leq i$（即所有前面的 token）。这种三角形 mask 假设 token 是线性排列的。

但在树结构中，不同分支的 token 不应该互相看到——它们代表的是互斥的候选路径。**每个 token 只能 attend 到它在树中的祖先节点**（包括所有共享的 prefix token）。

```
例：候选树结构
      root
     / \
    A   B
   /|    \
  C  D    E

扁平化顺序: [root, A, B, C, D, E]

Tree Attention Mask:
         root  A  B  C  D  E
root  [   1   0  0  0  0  0 ]
A     [   1   1  0  0  0  0 ]
B     [   1   0  1  0  0  0 ]
C     [   1   1  0  1  0  0 ]
D     [   1   1  0  0  1  0 ]
E     [   1   0  1  0  0  1 ]

注意:
- A 和 B 互相不可见（不同分支）
- C 可以看到 root 和 A（祖先），但看不到 B, D, E
- D 可以看到 root 和 A，但看不到 B, C, E
```

### 2.2 构建算法

```python
def build_tree_attention_mask(parent_indices, num_prefix_tokens):
    """
    构建 tree attention mask

    Args:
        parent_indices: 每个候选 token 的父节点索引
            例: [None, 0, 0, 1, 1, 2] 表示
                node 0 (root): 无父节点
                node 1 (A): 父节点是 0 (root)
                node 2 (B): 父节点是 0 (root)
                node 3 (C): 父节点是 1 (A)
                node 4 (D): 父节点是 1 (A)
                node 5 (E): 父节点是 2 (B)
        num_prefix_tokens: prefix 的长度 (所有候选 token 都能看到 prefix)

    Returns:
        mask: [num_candidates, num_prefix + num_candidates] 的 bool tensor
    """
    num_candidates = len(parent_indices)
    total_len = num_prefix_tokens + num_candidates

    # 初始化: 所有候选 token 都能看到所有 prefix tokens
    mask = torch.zeros(num_candidates, total_len, dtype=torch.bool)
    mask[:, :num_prefix_tokens] = True  # 所有 candidate 能看到 prefix

    # 每个 candidate 能看到自己
    for i in range(num_candidates):
        mask[i, num_prefix_tokens + i] = True

    # 每个 candidate 能看到其所有祖先
    for i in range(num_candidates):
        ancestor = parent_indices[i]
        while ancestor is not None:
            mask[i, num_prefix_tokens + ancestor] = True
            ancestor = parent_indices[ancestor]

    return mask
```

### 2.3 Position IDs 的设置

树结构中不同分支的同深度节点共享相同的 position ID（因为它们在序列中处于同一位置）：

```python
def compute_tree_position_ids(parent_indices, prefix_length):
    """
    计算 tree 中每个 token 的 position ID

    关键: 同一深度的 token 有相同的 position ID
    """
    num_candidates = len(parent_indices)
    depths = [0] * num_candidates

    # 计算每个节点的深度
    for i in range(num_candidates):
        if parent_indices[i] is not None:
            depths[i] = depths[parent_indices[i]] + 1

    # position ID = prefix_length + depth
    position_ids = [prefix_length + d for d in depths]
    return position_ids

# 例: parent_indices = [None, 0, 0, 1, 1, 2]
# depths = [0, 1, 1, 2, 2, 2]
# position_ids (prefix_length=10) = [10, 11, 11, 12, 12, 12]
```

**为什么要用深度作为 position ID**？因为 RoPE 等位置编码是基于绝对位置的。同一深度的候选 token（如 A 和 B）在原始序列中占据的是同一个位置，应该使用相同的位置编码。

## 3. 如何利用 FlashAttention 处理 Tree 结构

### 3.1 挑战

FlashAttention 原生支持的是 causal mask（三角形），不直接支持任意形状的 tree mask。如何在保留 FlashAttention 的性能优势的同时支持 tree attention？

### 3.2 方案一：稠密 Attention Mask

最直接但效率较低的方案——传入完整的 attention mask 矩阵：

```python
# 使用 PyTorch SDPA 的 custom mask
def tree_attention_with_dense_mask(query, key, value, tree_mask):
    """
    使用稠密 attention mask 实现 tree attention

    缺点: 
    - 无法使用 FlashAttention-2 的 fused kernel
    - 显存占用 O(N^2)
    - 计算不能利用 causal 的稀疏性
    """
    # tree_mask: [seq_len, seq_len], True = 可见
    # 转换为 additive mask: 0 = 可见, -inf = 不可见
    attn_mask = torch.where(tree_mask, 0.0, float('-inf'))

    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        is_causal=False  # 不能使用 causal 优化
    )
    return output
```

### 3.3 方案二：拆分 Prefix 和 Tree 部分

更高效的方案——将 attention 拆分为两部分：

```python
def tree_attention_split(query_tree, key_prefix, value_prefix,
                         key_tree, value_tree, tree_mask):
    """
    拆分为 prefix attention + tree attention

    Prefix 部分: 所有 tree token attend to prefix (标准 full attention)
    Tree 部分: tree token 之间的 attention (需要 tree mask)
    """
    # Part 1: Tree tokens attend to prefix tokens
    # 这部分是标准的 cross-attention, 所有 tree tokens 都能看到所有 prefix tokens
    attn_prefix = F.scaled_dot_product_attention(
        query_tree, key_prefix, value_prefix,
        is_causal=False  # full attention
    )

    # Part 2: Tree tokens attend to tree tokens (with tree mask)
    tree_mask_additive = torch.where(tree_mask, 0.0, float('-inf'))
    attn_tree = F.scaled_dot_product_attention(
        query_tree, key_tree, value_tree,
        attn_mask=tree_mask_additive,
        is_causal=False
    )

    # 合并（需要正确处理 softmax 的分母）
    # 实际实现中通常用 online softmax 技巧合并
    output = merge_attention_outputs(attn_prefix, attn_tree)
    return output
```

### 3.4 方案三：FlashAttention 的 Block Sparse 支持

FlashAttention-2/3 的最新版本支持 **block sparse attention mask**，可以更高效地处理 tree 结构：

```python
# FlashAttention v2.5+ 支持的 block mask
def tree_attention_flash(query, key, value, tree_structure):
    """
    使用 FlashAttention 的 block sparse 模式

    原理:
    1. 将 tree mask 转换为 block-level sparse pattern
    2. FlashAttention kernel 只计算非零 block
    3. 在 block 内部用 element-wise mask 处理边界
    """
    from flash_attn import flash_attn_func

    # 转换 tree structure 为 FlashAttention 支持的格式
    block_mask = convert_tree_to_block_mask(tree_structure)

    # 使用 FlashAttention 的 custom mask 接口
    output = flash_attn_func(
        query, key, value,
        causal=False,
        custom_mask=block_mask  # Flash Attention 3 支持
    )
    return output
```

### 3.5 方案四：Packing 策略

vLLM 中采用的实际策略更为精巧——将 tree 的不同路径 "pack" 成多个独立的 causal sequence，利用 FlashAttention 的 varlen 接口处理：

```python
def tree_attention_packed(tree, prefix_kv, model):
    """
    将 tree 拆解为多个独立路径，pack 成一个 batch

    例: tree 有 3 条路径 [root→A→C, root→A→D, root→B→E]

    Pack 为:
    seq 1: prefix + root + A + C  (用 causal attention)
    seq 2: prefix + root + A + D  (用 causal attention)
    seq 3: prefix + root + B + E  (用 causal attention)

    使用 FlashAttention varlen 一次性处理所有路径
    """
    paths = tree.enumerate_paths()

    # 拼接所有路径 (prefix KV 可以共享)
    packed_q = []
    packed_k = []
    packed_v = []
    cu_seqlens = [0]

    for path in paths:
        path_tokens = path.token_ids
        q, k, v = model.compute_qkv(path_tokens)

        # Prefix 的 KV 可以复用
        k_full = torch.cat([prefix_kv.k, k])
        v_full = torch.cat([prefix_kv.v, v])

        packed_q.append(q)
        packed_k.append(k_full)
        packed_v.append(v_full)
        cu_seqlens.append(cu_seqlens[-1] + len(q))

    # FlashAttention varlen
    output = flash_attn_varlen_func(
        torch.cat(packed_q),
        torch.cat(packed_k),
        torch.cat(packed_v),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        causal=True
    )
    return output
```

## 4. KV Cache 与 Tree 结构的交互

### 4.1 验证后的 KV Cache 处理

候选树中有些 token 会被接受，有些会被拒绝。验证后需要正确处理 KV Cache：

```python
def update_kv_cache_after_verification(kv_cache, tree, accepted_path):
    """
    验证后更新 KV Cache

    只保留被接受路径上的 KV 对, 丢弃其他分支的 KV
    """
    # 在验证的 forward pass 中, 所有 tree token 的 KV 都被计算了
    # 但只有 accepted_path 上的 token 的 KV 需要保留

    accepted_indices = get_tree_indices(tree, accepted_path)

    # 将被接受的 KV 移到正确的位置
    for layer in range(num_layers):
        k, v = kv_cache[layer]
        # 只保留 prefix + accepted tokens 的 KV
        prefix_len = kv_cache.prefix_length
        for i, idx in enumerate(accepted_indices):
            # 将 tree 中位置 idx 的 KV 复制到 prefix_len + i
            k[prefix_len + i] = k[prefix_len + idx]
            v[prefix_len + i] = v[prefix_len + idx]

        # 更新 KV Cache 长度
        kv_cache[layer] = (
            k[:prefix_len + len(accepted_indices)],
            v[:prefix_len + len(accepted_indices)]
        )
```

### 4.2 PagedAttention 与 Tree KV Cache

在 vLLM 的 PagedAttention 框架下，tree 结构的 KV cache 管理更复杂：

```python
class TreeKVCacheManager:
    """
    管理 tree 结构下的分页 KV Cache

    关键挑战:
    1. 不同分支共享 prefix 的 KV blocks
    2. 验证后需要释放被拒绝分支的 blocks
    3. 被接受路径的 blocks 需要保留并可能 compact
    """

    def allocate_tree_blocks(self, tree):
        """
        为候选树分配 KV blocks

        优化: 共享前缀只分配一份 blocks
        """
        # Prefix blocks 已经存在
        prefix_blocks = self.get_prefix_blocks()

        # 为 tree 中的每个节点分配 block slot
        tree_blocks = {}
        for node in tree.bfs_order():
            if node.depth == 0:
                # Root 节点复用 prefix 的最后一个 block
                tree_blocks[node.id] = prefix_blocks[-1]
            else:
                # 分配新 block（或复用 parent 的 block 如果还有空间）
                parent_block = tree_blocks[node.parent.id]
                if parent_block.has_space():
                    tree_blocks[node.id] = parent_block
                else:
                    tree_blocks[node.id] = self.allocate_new_block()

        return tree_blocks

    def cleanup_after_verification(self, tree_blocks, accepted_path):
        """验证后释放未接受分支的 blocks"""
        accepted_ids = set(n.id for n in accepted_path)
        for node_id, block in tree_blocks.items():
            if node_id not in accepted_ids:
                self.maybe_free_block(block)  # 引用计数减 1
```

## 5. Tree 验证的实现细节

### 5.1 多路径 Rejection Sampling

```python
def tree_rejection_sampling(tree, draft_probs, target_probs):
    """
    在树结构上进行 rejection sampling

    与线性 rejection sampling 的区别:
    - 需要评估多条路径
    - 选择接受长度最长的路径
    - 被拒绝位置从修正分布采样的 token 可能开启新路径
    """
    best_path = None
    best_accepted_length = 0

    for path in tree.enumerate_paths():
        accepted_length = 0

        for i, node in enumerate(path):
            x = node.token_id
            q_x = draft_probs[node.id][x]
            p_x = target_probs[node.id][x]

            r = random.uniform(0, 1)
            if r < min(1, p_x / q_x):
                accepted_length += 1
            else:
                # 从修正分布采样一个 token
                residual = np.maximum(0, target_probs[node.id] - draft_probs[node.id])
                residual /= residual.sum()
                replacement_token = np.random.choice(vocab_size, p=residual)
                accepted_length += 1  # 替换的 token 也算一个
                break

        if accepted_length > best_accepted_length:
            best_accepted_length = accepted_length
            best_path = path[:accepted_length]

    return best_path
```

### 5.2 高效的 Tree 验证：Token-level 贪心

一种更高效的验证策略是**token-level 贪心**——不是逐路径验证，而是自顶向下在树上贪心选择：

```python
def greedy_tree_verification(tree, target_probs):
    """
    自顶向下的贪心 tree 验证

    从 root 开始, 在每个深度:
    1. 计算所有同深度候选 token 的 target 概率
    2. 选择概率最高的 token
    3. 如果这个 token 不在候选中 → 停止 (从 target 分布采样)
    4. 如果在候选中 → 接受, 进入下一层

    优势: O(D) 而非 O(paths) 的复杂度
    """
    accepted = []
    current_nodes = [tree.root]

    for depth in range(tree.max_depth):
        # 收集这一层所有候选 token
        candidates = {}
        for node in current_nodes:
            for child in node.children:
                candidates[child.token_id] = child

        # Target model 在这一层选择的 token
        target_token = target_probs[depth].argmax()  # greedy mode

        if target_token in candidates:
            # 接受
            accepted.append(candidates[target_token])
            current_nodes = [candidates[target_token]]
        else:
            # 不在候选中, 使用 target 的选择
            accepted.append(make_node(target_token))
            break

    return accepted
```

## 6. SpecInfer 论文的 Tree-based Verification

### 6.1 SpecInfer 概述

SpecInfer (Miao et al., 2023) 是最早系统性地提出 tree-based speculative inference 的工作。

**核心贡献**：

1. **Multiple draft models**：同时使用多个不同的 draft model（而非单一 draft），每个 draft model 生成一条候选路径
2. **Token Tree**：将多个 draft model 的候选合并为一棵 token tree
3. **Tree verification**：target model 一次性验证整棵树

```
SpecInfer 流程:

Draft Model 1: prefix → a₁ → b₁ → c₁
Draft Model 2: prefix → a₂ → b₂ → c₂
Draft Model 3: prefix → a₁ → b₃ → c₃

合并为 Token Tree:
           prefix
          /     \
        a₁      a₂
       / \       |
      b₁  b₃    b₂
      |    |     |
      c₁  c₃    c₂

Target Model 用 Tree Attention 一次性验证
```

### 6.2 SpecInfer 的 Tree 合并

```python
def merge_draft_trees(draft_outputs):
    """
    将多个 draft model 的输出合并为一棵树

    优化: 共享公共前缀以减少 tree 大小
    """
    merged_tree = Tree()

    for draft_tokens in draft_outputs:
        current_node = merged_tree.root
        for token in draft_tokens:
            # 检查是否已有相同 token 的子节点
            existing_child = current_node.find_child(token)
            if existing_child:
                # 共享前缀
                current_node = existing_child
            else:
                # 创建新分支
                new_node = current_node.add_child(token)
                current_node = new_node

    return merged_tree
```

### 6.3 SpecInfer 的多 Draft Model 协调

SpecInfer 中多个 draft model 可以是不同类型（不同大小、不同微调方向），利用"集成"效果提高覆盖率：

| Draft Model | 特点 | 擅长预测 |
|------------|------|---------|
| Small LM | 通用能力 | 常见 pattern |
| Fine-tuned LM | 领域特化 | 领域术语 |
| N-gram model | 快速匹配 | 重复文本 |

这种多 draft model 的思路在后续的 EAGLE-2（动态 tree）和 Medusa（多 head）中以不同形式得到了继承。

## 7. 实践中的 Tree 参数调优

### 7.1 Tree 大小的权衡

| 参数 | 太小 | 太大 |
|------|------|------|
| 总节点数 $N$ | 覆盖率低，可能错过正确路径 | 验证开销大，attention 计算量增加 |
| Branching factor $b$ | 同层候选少 | 同层候选多但深度受限 |
| 最大深度 $D$ | 即使全对也只加速 $D$ 倍 | 后面的层预测质量差 |

**经验法则**：总节点数 $N \in [32, 128]$ 是一个好的范围。更大的 tree 在当前 GPU 上通常不划算。

### 7.2 Tree 形状的选择

```python
# 典型的 tree 配置

# 宽而浅 (适合低接受率场景)
wide_shallow = {
    'depth': 3,
    'branching': [8, 4, 2],  # 每层的分支数
    'total_nodes': 8 + 32 + 64  # ≈ 104
}

# 窄而深 (适合高接受率场景)
narrow_deep = {
    'depth': 7,
    'branching': [2, 2, 2, 2, 2, 2, 2],
    'total_nodes': 2 + 4 + 8 + 16 + 32 + 64 + 128  # ≈ 254, 需要剪枝
}

# 动态调整 (EAGLE-2 风格)
# 基于 confidence 在运行时决定 branching
dynamic = {
    'max_nodes': 64,
    'confidence_threshold': 0.8,
    'max_branch': 3,
    'max_depth': 8,
}
```

### 7.3 与 Continuous Batching 的交互

Tree attention 在 continuous batching 环境下会增加额外的复杂度：

```
Batch 中的不同请求:
Request 1: prefix(100 tokens) + tree(64 nodes)
Request 2: prefix(200 tokens) + tree(32 nodes)
Request 3: prefix(50 tokens) + tree(64 nodes)

挑战:
- 不同请求的 tree 大小不同
- 需要 padding 或 packing
- 验证后各请求接受的 token 数不同
- KV Cache 更新量不同
```

在 vLLM 中，这些问题通过 v1 架构的统一调度器处理——spec decode 的验证结果在调度器层面被协调，确保不同请求的进度正确同步。

---

> **下一节**：[exercises.md](exercises.md) — 动手练习：对比不同投机解码方案的加速比
