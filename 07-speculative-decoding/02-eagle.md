# EAGLE 系列

> EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 系列是目前最高效的投机解码方案之一。其核心洞察：与其训练一个独立的 draft model，不如直接利用 target model 内部的 hidden states 来预测未来 token——feature level 的信息比 token level 丰富得多。

## 1. 动机：为什么需要 EAGLE？

传统投机解码使用独立的 draft model（如用 LLaMA-7B 给 LLaMA-70B 做 draft），存在以下问题：

1. **分布对齐差**：小模型和大模型的分布差异大，接受率 $\alpha$ 低
2. **额外显存开销**：需要加载两个模型
3. **工程复杂度高**：需要管理两个模型的 KV Cache、调度等
4. **同系列模型不一定有**：许多模型没有官方的小版本

EAGLE 的关键创新：**不训练独立模型，而是在 target model 的 LM head 之前添加一个轻量级的 EAGLE head**，直接利用 target model 的 hidden states 进行推测。

## 2. EAGLE 架构

### 2.1 核心思路

EAGLE 观察到，LLM 最后一层 Transformer block 输出的 hidden state $h_t$ 包含了丰富的上下文语义信息。如果我们用 $h_t$ 来预测 $h_{t+1}$（下一步的 hidden state），然后通过 LM head 将 $h_{t+1}$ 映射为 token 分布，就能实现高质量的投机。

但直接在 hidden state 空间做 next-state prediction 面临一个问题：**hidden state 的不确定性**。$h_t$ 的后续演化取决于实际选取的 token $x_t$——不同的 token 会导致不同的 $h_{t+1}$。

EAGLE 的解决方案：**将 token embedding 和 hidden state 拼接作为输入**。

```
EAGLE Head 输入 = Concat(Embed(x_t), h_t)
                          ↓
                   Lightweight Transformer (1-2 layers)
                          ↓
                  Predicted hidden state ĥ_{t+1}
                          ↓
                   Shared LM Head (frozen)
                          ↓
                 Draft distribution q(x_{t+1})
```

### 2.2 架构细节

**EAGLE Head 的结构**：

```python
class EAGLEHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        # 将 token embedding 和 hidden state 融合
        # 输入维度: hidden_size (embed) + hidden_size (hidden state)
        self.fc = nn.Linear(2 * hidden_size, hidden_size)

        # 1-2 层 Transformer decoder layer
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, n_heads, ...)
            for _ in range(config.eagle_num_layers)  # 通常 1-2 层
        ])

        self.norm = RMSNorm(hidden_size)
        # 注意：LM Head 不在这里，复用 target model 的 LM Head

    def forward(self, token_embeds, hidden_states):
        """
        Args:
            token_embeds: [batch, seq_len, hidden_size] - token 的 embedding
            hidden_states: [batch, seq_len, hidden_size] - target model 的最后一层输出
        Returns:
            predicted_hidden: [batch, seq_len, hidden_size] - 预测的下一步 hidden state
        """
        # 拼接 token embedding 和 hidden state
        x = torch.cat([token_embeds, hidden_states], dim=-1)
        x = self.fc(x)  # 映射回 hidden_size

        # 通过轻量级 Transformer layers
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
```

**为什么用 `Concat(Embed(x_t), h_t)` 而不只是 `h_t`**？

论文中分析了 hidden state 的不确定性：给定 $h_t$，下一步的 hidden state $h_{t+1}$ 取决于实际采样的 token $x_t$。在 sampling 模式下，$x_t$ 是随机的，因此 $h_{t+1}$ 也有不确定性。通过显式地将 $x_t$ 的 embedding 作为输入，EAGLE head 能消除这种不确定性，在 feature level 做更准确的预测。

### 2.3 推理流程

```
Round i:

1. Target model 正常计算，得到 h_t 和 token x_t
2. EAGLE head 用 (Embed(x_t), h_t) 预测 ĥ_{t+1}
3. 用 LM head(ĥ_{t+1}) 得到 draft 分布，采样 x̂_{t+1}
4. EAGLE head 用 (Embed(x̂_{t+1}), ĥ_{t+1}) 预测 ĥ_{t+2}
5. ...重复 γ 步，得到 γ 个 draft tokens
6. Target model 并行验证所有 draft tokens
7. Rejection sampling 决定接受多少个 token
```

> **注意**：EAGLE head 的自回归推理只涉及 1-2 层 Transformer，比 target model 快一个数量级。

## 3. EAGLE-2：动态 Draft Tree

### 3.1 从链到树

EAGLE (v1) 生成的是一条线性的 draft token 链。但实际上，draft model 在不同位置的 confidence 是不同的——有些位置非常确定（应该继续扩展），有些位置不确定（不值得扩展）。

EAGLE-2 引入了**动态 draft tree**：根据每个节点的 confidence，决定是否扩展更多分支。

```
EAGLE (v1) - 线性链:
x₁ → x₂ → x₃ → x₄ → x₅

EAGLE-2 - 动态树:
        x₁
       / \
      x₂  x₂'
     / \    \
    x₃  x₃'  x₃''
    |
    x₄
```

### 3.2 Confidence 度量

EAGLE-2 使用 draft 分布的熵（或 top-1 概率）作为 confidence 度量：

```python
def should_expand(draft_logits, threshold):
    """决定是否扩展当前节点"""
    probs = softmax(draft_logits)
    top1_prob = probs.max()
    # 高 confidence → 继续扩展（可能只扩展 top-1）
    # 低 confidence → 扩展多个分支（top-k）或停止
    if top1_prob > threshold:
        return "expand_top1"
    elif top1_prob > threshold * 0.5:
        return "expand_topk"
    else:
        return "stop"
```

### 3.3 Tree 构建算法

```python
def build_draft_tree(eagle_head, lm_head, h_t, x_t, max_nodes, threshold):
    """
    动态构建 draft tree

    Args:
        eagle_head: EAGLE head 模型
        lm_head: 共享的 LM Head
        h_t: 当前 hidden state
        x_t: 当前 token
        max_nodes: tree 的最大节点数（计算预算）
        threshold: confidence 阈值
    """
    tree = Tree()
    queue = [(x_t, h_t, tree.root)]  # BFS 队列

    while queue and tree.num_nodes < max_nodes:
        token, hidden, parent = queue.pop(0)

        # EAGLE head 预测下一步
        embed = embedding(token)
        predicted_h = eagle_head(embed, hidden)
        logits = lm_head(predicted_h)
        probs = softmax(logits)

        # 基于 confidence 决定扩展策略
        top_probs, top_indices = probs.topk(k=3)

        if top_probs[0] > threshold:
            # 高 confidence: 只扩展 top-1
            child = tree.add_child(parent, top_indices[0])
            queue.append((top_indices[0], predicted_h, child))
        else:
            # 低 confidence: 扩展 top-k
            for i in range(min(3, max_nodes - tree.num_nodes)):
                child = tree.add_child(parent, top_indices[i])
                queue.append((top_indices[i], predicted_h, child))

    return tree
```

### 3.4 EAGLE-2 的优势

| 特性 | EAGLE (v1) | EAGLE-2 |
|------|-----------|---------|
| Draft 结构 | 固定长度链 | 动态树 |
| 节点数控制 | $\gamma$ 固定 | 基于计算预算动态调整 |
| 验证效率 | 线性 attention | Tree attention |
| 接受率 | 中等 | 更高（多条路径增加命中率） |
| 实际加速 | 2-3x | 3-4x |

## 4. EAGLE-3：进一步优化

EAGLE-3 在 EAGLE-2 的基础上进行了多方面改进：

### 4.1 更好的特征利用

EAGLE-3 不仅使用最后一层的 hidden state，还引入了多层特征融合：

```python
# EAGLE-3: 多层特征融合
hidden_states = []
for i, layer in enumerate(target_model.layers):
    h = layer(h)
    if i in selected_layers:  # 选择性提取中间层特征
        hidden_states.append(h)

# 融合多层特征
fused = feature_fusion(hidden_states)  # 加权求和或 attention
eagle_input = concat(embed, fused)
```

### 4.2 训练策略改进

EAGLE-3 在训练 EAGLE head 时加入了 tree-aware 的训练目标：不仅优化单步预测的准确性，还优化在 tree 结构下的端到端接受率。

### 4.3 与 vLLM v1 架构的深度集成

EAGLE-3 的实现与 vLLM 的 v1 架构（`vllm/v1/`）紧密集成，利用了 v1 架构的 block-level KV cache 管理和 FlashAttention 后端。

## 5. vLLM 中 EAGLE 的源码走读

vLLM 对 EAGLE 的实现主要位于 `vllm/v1/spec_decode/eagle/` 目录下。

### 5.1 目录结构

```
vllm/v1/spec_decode/eagle/
├── __init__.py
├── proposer.py       # EAGLE draft proposer (核心逻辑)
└── utils.py          # 辅助函数

vllm/v1/worker/gpu/spec_decode/
├── eagle_proposer.py # GPU worker 端的 EAGLE proposer
└── ...

vllm/model_executor/models/
├── eagle.py          # EAGLE head 模型定义
└── ...
```

### 5.2 EAGLEProposer（提议器）

`EAGLEProposer` 负责使用 EAGLE head 生成 draft tokens：

```python
# vllm/v1/spec_decode/eagle/proposer.py (简化)
class EAGLEProposer:
    """
    EAGLE draft token 提议器

    核心职责：
    1. 管理 EAGLE head 的推理
    2. 构建 draft tree（EAGLE-2 模式）
    3. 与 target model 的 hidden states 交互
    """

    def __init__(self, eagle_head, target_model, config):
        self.eagle_head = eagle_head
        self.lm_head = target_model.lm_head  # 共享 LM Head
        self.num_speculative_tokens = config.num_speculative_tokens
        self.method = config.speculative_draft_tensor_parallel_size

    def propose(self, hidden_states, sampled_token_ids, ...):
        """
        生成 draft tokens

        Args:
            hidden_states: target model 最后一层的输出
            sampled_token_ids: target model 最后采样的 token
        Returns:
            draft_token_ids: 提议的 token 序列
            draft_probs: 每个位置的 draft 分布
        """
        # 获取 token embedding
        token_embeds = self.eagle_head.embed_tokens(sampled_token_ids)

        draft_tokens = []
        draft_probs = []

        current_hidden = hidden_states
        current_embed = token_embeds

        for step in range(self.num_speculative_tokens):
            # EAGLE head forward
            predicted_hidden = self.eagle_head(current_embed, current_hidden)

            # 通过共享 LM Head 得到 logits
            logits = self.lm_head(predicted_hidden)
            probs = torch.softmax(logits, dim=-1)

            # 采样下一个 draft token
            next_token = torch.multinomial(probs, num_samples=1)
            draft_tokens.append(next_token)
            draft_probs.append(probs)

            # 准备下一步输入
            current_embed = self.eagle_head.embed_tokens(next_token)
            current_hidden = predicted_hidden

        return draft_tokens, draft_probs
```

### 5.3 与 Target Model 的集成

在 vLLM v1 的架构中，EAGLE 与 target model 的交互方式：

```python
# 简化的 speculative decoding 循环
class SpecDecodeWorker:
    def execute_model(self, ...):
        # 1. Target model 正常 forward
        #    - 生成当前步的 token
        #    - 保存最后一层 hidden states
        output = self.target_model.forward(input_ids, ...)
        hidden_states = output.hidden_states[-1]
        sampled_token = output.sampled_token_ids

        # 2. EAGLE proposer 生成 draft tokens
        draft_tokens, draft_probs = self.eagle_proposer.propose(
            hidden_states=hidden_states,
            sampled_token_ids=sampled_token,
        )

        # 3. Target model 验证 draft tokens (并行 forward)
        verify_input = torch.cat([sampled_token] + draft_tokens)
        verify_output = self.target_model.forward(verify_input, ...)
        target_probs = verify_output.logits  # 所有位置的 target 分布

        # 4. Rejection sampling
        accepted = self.rejection_sampler(
            draft_tokens, draft_probs, target_probs
        )

        return accepted
```

### 5.4 Hidden States 的缓存与管理

EAGLE 需要 target model 输出 hidden states，这带来额外的显存开销。vLLM 的处理方式：

```python
# 在 model runner 中配置 hidden states 输出
class ModelRunner:
    def __init__(self, ...):
        if self.speculative_config and self.speculative_config.method == "eagle":
            # 配置 target model 保存最后一层 hidden states
            self.return_hidden_states = True
            # 预分配 hidden states buffer
            self.hidden_states_buffer = torch.empty(
                max_batch_size, hidden_size,
                dtype=dtype, device=device
            )
```

## 6. 训练 EAGLE Head

### 6.1 训练数据准备

EAGLE head 的训练数据来自 target model 的推理过程：

```python
# 训练数据采集
def collect_training_data(target_model, dataset):
    """
    对 dataset 中的每个样本做 forward pass，收集:
    - hidden states: 每个位置的 target model 最后一层输出
    - token embeddings: 每个位置的 token embedding
    - next token: 下一个位置的 ground truth token
    """
    training_data = []
    for text in dataset:
        input_ids = tokenize(text)
        with torch.no_grad():
            outputs = target_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [seq_len, hidden_size]
            embeddings = target_model.embed_tokens(input_ids)

        # 构建训练对
        for t in range(len(input_ids) - 1):
            training_data.append({
                'input_embed': embeddings[t],
                'input_hidden': hidden_states[t],
                'target_hidden': hidden_states[t + 1],
                'target_token': input_ids[t + 1],
            })

    return training_data
```

### 6.2 训练目标

EAGLE head 的训练有两种目标：

1. **Hidden state regression**：预测下一步的 hidden state

$$
\mathcal{L}_{\text{hidden}} = \| \hat{h}_{t+1} - h_{t+1} \|^2
$$

2. **Next token prediction**（通过共享 LM head）：

$$
\mathcal{L}_{\text{token}} = -\log p_{\text{LM\_head}}(x_{t+1} | \hat{h}_{t+1})
$$

实践中通常使用 $\mathcal{L}_{\text{token}}$ 或两者的加权组合。

### 6.3 训练成本

| 项目 | 数值 |
|------|------|
| 训练数据量 | 10K-100K 样本 |
| 训练时间（单 GPU） | 几小时到 1 天 |
| EAGLE head 参数量 | ~100M-500M（取决于 target model 大小） |
| 相对 target model 参数占比 | 1-5% |

## 7. 实际加速比数据

以下数据来自 EAGLE 论文和社区实测（不同硬件和任务会有差异）：

### 7.1 EAGLE (v1) 加速比

| Target Model | Task | Speedup (greedy) | Speedup (sampling) |
|-------------|------|-------------------|---------------------|
| Vicuna-7B | MT-Bench | 2.78x | 2.42x |
| Vicuna-13B | MT-Bench | 2.89x | 2.55x |
| LLaMA2-Chat-70B | MT-Bench | 3.06x | 2.72x |
| Mixtral-8x7B | MT-Bench | 2.51x | 2.16x |

### 7.2 EAGLE-2 加速比

| Target Model | Task | Speedup (greedy) | Speedup (sampling) |
|-------------|------|-------------------|---------------------|
| Vicuna-7B | MT-Bench | 3.24x | 2.78x |
| LLaMA2-Chat-70B | MT-Bench | 3.55x | 3.11x |
| Mixtral-8x7B | MT-Bench | 3.01x | 2.65x |

### 7.3 影响加速比的因素

1. **任务类型**：结构化输出（JSON、代码）通常加速比更高（因为 draft 更容易命中）
2. **Temperature**：temperature=0 通常比高 temperature 加速比更高
3. **Batch size**：单请求加速比最高，大 batch 时递减
4. **模型大小**：target model 越大，decode 阶段越 memory-bound，加速比越高
5. **GPU 型号**：HBM 带宽越低的 GPU（相对计算能力），加速比越高

## 8. vLLM 中使用 EAGLE 的配置

```bash
# 启动 vLLM 时指定 EAGLE
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model path/to/eagle-head \
    --speculative-config '{"method": "eagle"}' \
    --num-speculative-tokens 5

# 或使用 HuggingFace 上预训练的 EAGLE head
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model yuhuili/EAGLE-LLaMA3.1-Instruct-8B \
    --num-speculative-tokens 5
```

**关键参数**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--num-speculative-tokens` | Draft 长度 $\gamma$ | 3-7 |
| `--speculative-model` | EAGLE head 路径 | HuggingFace 模型 ID |
| `--spec-decoding-acceptance-method` | 验证方法 | `rejection_sampler` |

## 9. 与其他方案的对比

| 维度 | EAGLE | 独立 Draft Model | Medusa | MTP |
|------|-------|------------------|--------|-----|
| 额外模型 | EAGLE head (~1-5%) | 完整小模型 (~10-100%) | Medusa heads (~1%) | MTP heads (训练时内建) |
| 接受率 | 高（feature-level） | 中等 | 中等 | 高 |
| 显存开销 | 小 | 大 | 很小 | 很小 |
| 训练成本 | 中（需采集 hidden states） | 零（用现有模型） | 低 | 高（需重新训练主模型） |
| 工程复杂度 | 中 | 高 | 低 | 低（如果模型原生支持） |

---

> **下一节**：[03-medusa.md](03-medusa.md) — Medusa：多头并行预测
