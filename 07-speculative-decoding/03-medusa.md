# Medusa：多头并行预测

> Medusa 的核心思想极其简洁：在 target model 的最后一层之后，附加多个独立的 MLP "头"，每个头预测不同未来位置的 token。这些头可以并行推理，配合 tree attention 一次性验证多条候选路径，实现无需独立 draft model 的投机解码。

## 1. 动机与核心思想

### 1.1 从"一步一头"到"一步多头"

标准 LLM 只有一个 LM head，每次 forward pass 预测下一个 token。Medusa 的想法是：**既然最后一层的 hidden state 包含了丰富的上下文信息，为什么不用多个 head 同时预测多个未来 token？**

```
标准 LLM:
hidden_state → LM Head → next_token (位置 t+1)

Medusa:
                    ┌── Medusa Head 1 → token at t+1
hidden_state ──────┼── Medusa Head 2 → token at t+2
                    ├── Medusa Head 3 → token at t+3
                    └── Medusa Head 4 → token at t+4
```

每个 Medusa Head $k$ 独立预测位置 $t+k$ 的 token，这些预测是**并行**的——不像 EAGLE 那样需要自回归地依次预测。

### 1.2 关键优势

1. **无需额外 draft model**：只增加几个小 MLP head（参数量极小）
2. **并行生成所有 draft tokens**：一次 forward pass 同时得到所有位置的预测
3. **训练简单**：冻结 target model，只训练 Medusa heads
4. **工程简洁**：不需要管理两个模型的 KV Cache

## 2. 架构详解

### 2.1 Medusa Head 结构

每个 Medusa head 是一个轻量级的 MLP（通常 1-2 层 ResNet-style block）：

```python
class MedusaHead(nn.Module):
    """
    单个 Medusa Head: 预测位置 t+k 的 token

    结构: hidden_state → MLP (with residual) → logits
    """
    def __init__(self, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(hidden_size) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = hidden_states
        for block in self.blocks:
            x = block(x)
        return self.linear(x)


class ResBlock(nn.Module):
    """带残差连接的 MLP block"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """完整的 Medusa 模型：target model + 多个 Medusa heads"""
    def __init__(self, base_model, num_heads=4, num_layers=1):
        super().__init__()
        self.base_model = base_model  # 冻结的 target model
        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size

        # 创建 K 个 Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size, num_layers)
            for _ in range(num_heads)
        ])

    def forward(self, input_ids, **kwargs):
        # 获取 base model 的 hidden states
        with torch.no_grad():
            outputs = self.base_model(
                input_ids, output_hidden_states=True, **kwargs
            )
        hidden_states = outputs.hidden_states[-1]

        # 原始 LM head 的输出（位置 t+1 的预测，即 head 0）
        base_logits = outputs.logits

        # 每个 Medusa head 独立预测
        medusa_logits = [head(hidden_states) for head in self.medusa_heads]
        # medusa_logits[k] 预测位置 t+k+2 的 token (k=0,1,...,K-1)

        return base_logits, medusa_logits
```

### 2.2 参数量分析

| 组件 | 参数量 | 占比（以 7B 模型为例） |
|------|--------|------------------------|
| 单个 Medusa Head (1 层) | $\approx h^2 + h \times V \approx$ 60M | ~0.9% |
| 4 个 Medusa Heads | ~240M | ~3.4% |
| Target Model (7B) | 7000M | 100% |

> Medusa heads 的参数量极小。但要注意，因为有 $h \times V$ 的线性层（映射到 vocab），在大 vocab 模型上，单个 head 的参数也不算少。实际中一些实现会**共享 target model 的 LM head 权重**来减少参数。

## 3. 候选生成与 Tree Attention 验证

### 3.1 构建候选树

每个 Medusa head 输出一个概率分布，从中取 top-k 个候选 token。将所有 head 的候选组合成一棵**候选树**：

```python
def build_candidate_tree(base_logits, medusa_logits, top_k=5):
    """
    构建候选 token 树

    假设有 K=3 个 Medusa heads, 每个取 top-5:
    - 原始 LM head: top-5 candidates for t+1
    - Medusa head 1: top-5 candidates for t+2
    - Medusa head 2: top-5 candidates for t+3

    笛卡尔积会产生 5^3 = 125 条路径，实际中会剪枝。
    """
    # 获取每个位置的 top-k candidates
    candidates = []
    candidates.append(base_logits.topk(top_k))      # t+1
    for head_logits in medusa_logits:
        candidates.append(head_logits.topk(top_k))   # t+2, t+3, ...

    # 构建树结构（剪枝后）
    tree = build_tree_with_pruning(candidates, max_nodes=64)
    return tree
```

### 3.2 候选树的剪枝

直接使用笛卡尔积会导致候选数量指数增长。实际中使用以下剪枝策略：

1. **基于概率的剪枝**：只保留联合概率 $\prod_k p_k(x_k)$ 最高的路径
2. **固定拓扑结构**：预定义 tree 的形状（Medusa 论文的默认方法）
3. **预算限制**：限制总节点数（如 64 个）

Medusa 论文中默认使用**固定拓扑结构**（称为 Medusa topology），在训练数据上统计最优的树形状：

```python
# Medusa 默认拓扑示例 (以 4 heads, 每个 top-3 为例)
# 格式: (head_0_candidate, head_1_candidate, head_2_candidate, head_3_candidate)
# -1 表示该路径到此终止
MEDUSA_CHOICES = [
    (0,),                    # 只有 head_0 的 top-1
    (0, 0),                  # head_0 top-1 + head_1 top-1
    (0, 0, 0),               # head_0 top-1 + head_1 top-1 + head_2 top-1
    (0, 0, 0, 0),            # 完整路径 (全 top-1)
    (0, 1),                  # head_0 top-1 + head_1 top-2
    (1,),                    # head_0 top-2
    (1, 0),                  # head_0 top-2 + head_1 top-1
    (0, 0, 1),               # ...
    (0, 2),
    (2,),
    # ... 总共约 60-64 个候选路径
]
```

### 3.3 Tree Attention 验证

构建好候选树后，target model 使用 **tree attention** 一次性验证所有路径：

```
输入: prefix tokens + 候选树中的所有 tokens (扁平化)
Attention Mask: tree attention mask (每个 token 只 attend 到其祖先)
输出: 每个位置的 target model 分布
```

验证过程：

```python
def verify_candidates(target_model, prefix, candidate_tree):
    """
    使用 tree attention 一次性验证所有候选路径
    """
    # 1. 将 tree 扁平化为 token 序列
    flat_tokens, tree_mask, parent_indices = candidate_tree.flatten()

    # 2. 构建 tree attention mask
    # tree_mask[i][j] = 1 iff token j 是 token i 的祖先
    attention_mask = build_tree_attention_mask(tree_mask)

    # 3. Target model forward (with tree attention mask)
    target_logits = target_model.forward(
        input_ids=torch.cat([prefix, flat_tokens]),
        attention_mask=attention_mask,
    )

    # 4. 对每条路径做 rejection sampling
    best_path, accepted_length = select_best_path(
        candidate_tree, target_logits, flat_tokens
    )

    return best_path[:accepted_length]
```

详见 [06-tree-attention.md](06-tree-attention.md) 了解 tree attention 的具体实现。

## 4. 验证策略

### 4.1 标准 Rejection Sampling

与标准投机解码相同，对每条路径上的每个 token 做 rejection sampling。但由于有多条路径，选择**接受长度最长的路径**。

### 4.2 Typical Acceptance（Medusa 特色）

Medusa 论文提出了 **Typical Acceptance** 策略，放松无损性约束以获得更高接受率：

```python
def typical_acceptance(target_probs, candidate_token, epsilon=0.3):
    """
    Typical Acceptance: 只要 target model 认为这个 token
    的概率足够高（> epsilon），就接受

    不看 draft model 的分布，因此:
    - 不保证严格无损
    - 但实践中质量损失极小
    - 接受率显著提高
    """
    p = target_probs[candidate_token]
    return p > epsilon
```

**Typical Acceptance 的直觉**：如果 target model 认为一个 token 的概率 > 30%，那这个 token 在分布中就是"典型的"（typical），即使不是概率最高的 token，也是合理的选择。

**Medusa 论文的实验结果**：Typical Acceptance 在 MT-Bench 上的质量评分（由 GPT-4 打分）与标准 rejection sampling 几乎相同（差异在噪声范围内），但加速比提高 10-20%。

### 4.3 后验验证 (Posterior Validation)

Medusa-2 进一步提出了介于 rejection sampling 和 typical acceptance 之间的验证方法——以概率接受并通过后验调整保持分布质量。

## 5. 与 EAGLE 的详细对比

| 维度 | Medusa | EAGLE |
|------|--------|-------|
| **Draft 机制** | 多个并行 MLP head，一次 forward 得到所有位置的预测 | 自回归地用 EAGLE head 逐步预测 |
| **输入信息** | 仅用最后一层 hidden state | Hidden state + token embedding |
| **位置依赖** | 各 head 独立，不建模位置间依赖 | 自回归链/树，显式建模位置间依赖 |
| **Draft 质量** | 后续位置预测质量快速下降（因为独立预测） | 质量下降较慢（因为逐步条件化） |
| **Draft 速度** | 极快（一次并行 forward） | 较慢（$\gamma$ 次 EAGLE head forward） |
| **参数效率** | 多个 head 共享 target 的 LM head | 独立的 EAGLE head + 共享 LM head |
| **训练难度** | 简单（冻结 base，只训练小 MLP） | 中等（需要采集 hidden states 数据） |
| **接受率** | 中等 | 较高 |
| **最终加速比** | 2.2-2.8x | 2.8-3.5x |
| **适用场景** | 对工程简洁性要求高 | 对加速比要求高 |

**核心差异的深度分析**：

Medusa 的各 head 是独立预测的——head $k$ 在预测 $x_{t+k}$ 时，**不知道** $x_{t+1}, ..., x_{t+k-1}$ 是什么。这是一个很强的独立性假设，在自然语言中通常不成立（token 之间有很强的依赖关系）。

EAGLE 则通过自回归链保留了位置间的依赖：预测 $x_{t+2}$ 时，已经知道了 $\hat{x}_{t+1}$（虽然可能不准确）。这就是为什么 EAGLE 的 draft 质量通常高于 Medusa。

但 Medusa 的优势在于并行性——所有 head 的预测可以在一次 forward pass 中完成，不需要 $\gamma$ 次串行的 EAGLE head 推理。当 $\gamma$ 较大时，EAGLE 的 draft 开销更高。

## 6. vLLM 源码走读

### 6.1 Medusa 模型定义

```python
# vllm/model_executor/models/medusa.py (简化)
class MedusaModel(nn.Module):
    """Medusa speculative decoding model"""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.num_heads = config.medusa_num_heads
        self.num_layers = config.medusa_num_layers

        # Medusa heads
        self.medusa_heads = nn.ModuleList([
            MedusaHead(
                config.hidden_size,
                config.vocab_size,
                num_layers=self.num_layers,
            )
            for _ in range(self.num_heads)
        ])

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, hidden_size] - 来自 target model 的最后一层

        Returns:
            medusa_logits: list of [batch, vocab_size], 每个 head 一个
        """
        return [head(hidden_states) for head in self.medusa_heads]
```

### 6.2 Medusa Proposer

```python
# vllm/v1/spec_decode/medusa.py (简化)
class MedusaProposer:
    """
    Medusa draft token 提议器

    核心区别于 EAGLE: 一次 forward 就生成所有位置的候选
    """

    def __init__(self, medusa_model, config):
        self.medusa_model = medusa_model
        self.num_heads = config.medusa_num_heads
        self.top_k = config.medusa_top_k  # 每个 head 的 top-k
        self.choices = config.medusa_choices  # 预定义的拓扑结构

    def propose(self, hidden_states):
        """
        生成 draft tokens

        Args:
            hidden_states: target model 的最后一层输出

        Returns:
            candidate_tree: 候选 token 树
        """
        # 一次 forward pass 得到所有 head 的 logits
        medusa_logits = self.medusa_model(hidden_states)

        # 从每个 head 取 top-k
        candidates = []
        for logits in medusa_logits:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(self.top_k)
            candidates.append((top_ids, top_probs))

        # 根据预定义拓扑构建候选树
        tree = self.build_tree(candidates, self.choices)
        return tree

    def build_tree(self, candidates, choices):
        """基于 Medusa choices 构建候选树"""
        tree_tokens = []
        tree_indices = []

        for choice in choices:
            tokens = []
            for depth, idx in enumerate(choice):
                tokens.append(candidates[depth][0][idx])  # top_ids[idx]
            tree_tokens.append(tokens)
            tree_indices.append(choice)

        return CandidateTree(tree_tokens, tree_indices)
```

### 6.3 Medusa 的 Tree Attention 集成

```python
# vllm/v1/spec_decode/medusa.py (tree mask 构建部分)
def create_medusa_tree_attn_mask(choices, device):
    """
    为 Medusa 候选树创建 attention mask

    关键：每个候选 token 只能 attend 到:
    1. 所有 prefix tokens
    2. 同一路径上的祖先 tokens
    不能 attend 到其他路径的 tokens
    """
    num_candidates = len(choices)
    # 初始化为全 0 (不可见)
    mask = torch.zeros(num_candidates, num_candidates, dtype=torch.bool, device=device)

    for i, choice_i in enumerate(choices):
        for j, choice_j in enumerate(choices):
            # choice_j 是 choice_i 的前缀 → i 可以看到 j
            if is_prefix(choice_j, choice_i):
                mask[i][j] = True

    return mask

def is_prefix(a, b):
    """检查序列 a 是否是序列 b 的前缀"""
    if len(a) > len(b):
        return False
    return a == b[:len(a)]
```

## 7. Medusa Head 的训练策略

### 7.1 Stage 1：冻结 Base Model

最常用的训练方式——冻结 target model 的全部参数，只训练 Medusa heads：

```python
# 训练循环
def train_medusa_heads(base_model, medusa_heads, dataset, epochs=3):
    # 冻结 base model
    for param in base_model.parameters():
        param.requires_grad = False

    optimizer = Adam(medusa_heads.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch in dataset:
            input_ids = batch['input_ids']  # [batch, seq_len]

            # 获取 base model 的 hidden states
            with torch.no_grad():
                outputs = base_model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

            # 每个 Medusa head 预测对应位置的 token
            total_loss = 0
            for k, head in enumerate(medusa_heads):
                logits = head(hidden_states[:, :-(k+1), :])
                # 目标: 位置 t 的 hidden state 预测位置 t+k+1 的 token
                targets = input_ids[:, (k+1):]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                total_loss += loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 7.2 Stage 2：联合微调（可选）

在 Stage 1 之后，可以解冻 base model 的最后几层，与 Medusa heads 一起微调：

```python
# 解冻最后 2 层
for layer in base_model.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
```

> **注意**：联合微调可能影响 target model 的原始能力。Medusa 论文建议在大多数场景下只做 Stage 1。

### 7.3 训练数据与超参数

| 项目 | 推荐值 |
|------|--------|
| 训练数据 | ShareGPT 或领域数据，10K-50K 样本 |
| 学习率 | 1e-3（只训练 heads） |
| Epochs | 1-3 |
| Medusa heads 数量 | 3-5 |
| 每个 head 的 MLP 层数 | 1 |
| Batch size | 8-16 |
| 训练时间 | 几小时（单 A100） |

## 8. 实际性能数据

### 8.1 加速比

| 模型 | 方法 | Speedup (greedy) | Speedup (temp=0.7) |
|------|------|------------------|---------------------|
| Vicuna-7B | Medusa-1 (4 heads) | 2.3x | 2.0x |
| Vicuna-13B | Medusa-1 (4 heads) | 2.5x | 2.2x |
| Vicuna-33B | Medusa-1 (5 heads) | 2.7x | 2.4x |
| Vicuna-7B | Medusa-2 (joint finetune) | 2.8x | 2.5x |

### 8.2 各 Head 的预测准确率

| Head | 预测位置 | Top-1 准确率 | Top-5 命中率 |
|------|---------|-------------|-------------|
| Head 0 (原始 LM Head) | t+1 | ~75% | ~95% |
| Medusa Head 1 | t+2 | ~55% | ~85% |
| Medusa Head 2 | t+3 | ~40% | ~75% |
| Medusa Head 3 | t+4 | ~30% | ~65% |
| Medusa Head 4 | t+5 | ~25% | ~55% |

**观察**：后续位置的预测准确率快速下降，这是 Medusa 独立预测设计的固有局限。这也是为什么 Medusa heads 数量通常不超过 5——更多 head 的边际收益很小。

## 9. Medusa 的局限与改进方向

### 9.1 核心局限

1. **独立性假设**：各 head 独立预测，无法建模 token 间依赖
2. **固定拓扑**：预定义的候选树形状在不同输入上可能不是最优的
3. **加速比天花板**：由于独立预测的质量限制，很难超过 3x

### 9.2 改进方向

- **Medusa + EAGLE 混合**：用 EAGLE 的自回归方式生成更准确的 draft，但以 Medusa 的训练简洁性为基础
- **动态拓扑**：类似 EAGLE-2，根据 confidence 动态调整候选树
- **Self-Distillation**：用 target model 的 soft labels 训练 Medusa heads，而非 hard labels

---

> **下一节**：[04-mtp.md](04-mtp.md) — Multi-Token Prediction：从训练阶段开始优化
