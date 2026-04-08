# Multi-Token Prediction (MTP)

> MTP 的核心洞察：与其在推理阶段"亡羊补牢"地添加 draft heads，不如在**训练阶段**就让模型学会预测多个未来 token。这样训练出来的模型天然具备投机解码能力——MTP heads 既是训练正则化手段，也是推理时的 draft 提议器。

## 1. 论文解读：Better & Faster LLMs via Multi-token Prediction

### 1.1 研究动机

Meta 在 2024 年提出的 MTP 论文指出了传统 next-token prediction (NTP) 训练目标的两个局限：

1. **短视性 (Myopia)**：NTP 只优化下一个 token 的预测，模型不被激励去理解更长距离的依赖关系
2. **采样效率低**：每个训练 token 只提供一个梯度信号，信息利用不充分

MTP 通过让模型同时预测未来 $K$ 个 token 来解决这些问题：

$$
\mathcal{L}_{\text{MTP}} = \sum_{k=1}^{K} \lambda_k \cdot \mathcal{L}_k = \sum_{k=1}^{K} \lambda_k \cdot \left(-\frac{1}{T}\sum_{t=1}^{T} \log p_k(x_{t+k} | x_1, ..., x_t)\right)
$$

其中 $p_k(\cdot)$ 是第 $k$ 个预测 head 的输出分布，$\lambda_k$ 是对应的权重系数。

### 1.2 架构设计

```
                    输入: x₁, x₂, ..., x_T
                           ↓
                   Shared Transformer Backbone
                           ↓
                      Hidden States h₁, ..., h_T
                    ╱      |        |         ╲
               Head 1   Head 2   Head 3   Head 4
               (k=1)    (k=2)    (k=3)    (k=4)
                 ↓        ↓        ↓        ↓
              x_{t+1}  x_{t+2}  x_{t+3}  x_{t+4}
```

**关键设计选择**：

- **共享主干 (Shared Backbone)**：所有 heads 共享同一个 Transformer backbone，只有最后的预测 head 不同
- **独立 heads**：每个 head 是一个独立的 unembedding layer（或带几层 MLP 的 head）
- **权重共享 vs 独立**：论文发现独立的 heads 表现更好（与 Medusa 类似）

### 1.3 训练阶段的收益

MTP 在训练阶段的好处不仅仅是为推理加速准备——它还能提升模型质量：

| 效果 | 原因 |
|------|------|
| 更好的表征学习 | 预测多个 token 迫使模型学习更抽象、更长距离的特征 |
| 正则化 | 多任务学习天然有正则化效果 |
| 更高的数据效率 | 每个 token 提供 $K$ 个梯度信号 |
| 改善代码生成 | 代码有较强的 pattern，多步预测帮助理解结构 |

论文在 code benchmarks 上观察到 MTP 训练的模型（即使只用 head 1 推理）也比 NTP 基线更好。

## 2. 与 Medusa 的区别

MTP 和 Medusa 都是在主模型上附加多个预测 head，但设计哲学完全不同：

| 维度 | MTP | Medusa |
|------|-----|--------|
| **训练方式** | 预训练/继续训练时加入多 head loss | 预训练后冻结 base model，只训练 heads |
| **Base model 是否变化** | 是（backbone 被 MTP loss 共同优化） | 否（base model 冻结） |
| **Head 的作用** | 训练时：辅助 loss；推理时：draft | 仅在推理时使用 |
| **预测质量** | 更高（backbone 为多步预测优化过） | 较低（backbone 未针对多步预测优化） |
| **部署灵活性** | 需要专门训练的模型 | 可以给任意模型添加 |
| **训练成本** | 很高（需要重新训练或继续训练） | 很低（几小时微调） |

**深层理解**：Medusa 是"推理时的补丁"——在已训练好的模型上加 heads，heads 只能利用 frozen backbone 碰巧编码的信息。MTP 是"训练时的设计"——backbone 本身被优化来支持多步预测，因此 hidden states 天然包含更多前瞻信息。

## 3. DeepSeek-V3 的 MTP 实现

### 3.1 DeepSeek-V3 的 MTP 架构

DeepSeek-V3 是第一批在大规模生产模型中采用 MTP 的模型之一。其 MTP 实现有独特的设计：

```
                        Input Tokens
                            ↓
                    Main Transformer (N layers)
                            ↓
                      Hidden States (h_t)
                            ↓
                  ┌─────────┼─────────┐
                  ↓         ↓         ↓
             MTP Head 1  MTP Head 2  (可选更多)
                  ↓         ↓         ↓
              Predict    Predict    Predict
              x_{t+1}   x_{t+2}   x_{t+3}
```

**DeepSeek-V3 MTP 的特殊设计**：

1. **Sequential MTP module 设计**：不像 Meta 的论文用独立 heads，DeepSeek-V3 的 MTP modules 是**级联**的（sequential）——每个 module 的输入不仅包含主模型的 hidden state，还包含前一个 MTP module 的预测信息

```python
# DeepSeek-V3 MTP Module (简化)
class DeepSeekMTPModule(nn.Module):
    """
    DeepSeek-V3 的 MTP module

    与 Meta MTP 的区别:
    - 级联设计: module k 依赖 module k-1 的输出
    - 共享 embedding 和 output projection
    - 使用 RMSNorm 做特征归一化
    """
    def __init__(self, config):
        super().__init__()
        # 将前一步的预测 embedding 和当前 hidden state 融合
        self.embed_tokens = SharedEmbedding(config)  # 共享主模型的 embedding
        self.norm = RMSNorm(config.hidden_size)

        # 轻量级 Transformer block（通常只有 1 层）
        self.transformer_block = TransformerBlock(config)

        # 共享主模型的 output projection
        self.lm_head = SharedLMHead(config)

    def forward(self, hidden_states, prev_token_embeds):
        """
        Args:
            hidden_states: 主模型（或前一个 MTP module）的 hidden states
            prev_token_embeds: 前一步预测 token 的 embedding
        """
        # 融合 hidden states 和 token embedding
        h = self.norm(hidden_states) + prev_token_embeds
        h = self.transformer_block(h)
        logits = self.lm_head(h)
        return h, logits
```

2. **训练时 MTP loss**：

```python
def compute_mtp_loss(model, input_ids, mtp_modules):
    """
    计算 DeepSeek-V3 的 MTP 训练 loss

    主 loss + 各 MTP module 的辅助 loss
    """
    # 主模型 forward
    hidden_states = model.backbone(input_ids)
    main_logits = model.lm_head(hidden_states)

    # 主 loss
    main_loss = cross_entropy(main_logits[:, :-1], input_ids[:, 1:])

    # MTP losses
    mtp_loss = 0
    prev_hidden = hidden_states
    for k, module in enumerate(mtp_modules):
        # 前一步的 target token embedding (teacher forcing)
        prev_token_embeds = model.embed_tokens(input_ids[:, k+1:])

        # MTP module forward
        new_hidden, mtp_logits = module(
            prev_hidden[:, :-(k+1)],  # 对齐序列长度
            prev_token_embeds
        )
        prev_hidden = new_hidden

        # 第 k+2 个 token 的预测 loss
        targets = input_ids[:, k+2:]
        mtp_loss += cross_entropy(mtp_logits[:, :-1], targets)

    total_loss = main_loss + mtp_weight * mtp_loss
    return total_loss
```

### 3.2 推理时用 MTP heads 做投机解码

训练好的 MTP heads 在推理时直接作为 draft 提议器：

```python
def mtp_speculative_decode(model, mtp_modules, prefix, gamma):
    """
    使用 MTP heads 进行投机解码

    关键优势: MTP heads 是训练时就优化过的，
    draft 质量比 post-hoc 添加的 Medusa heads 更高
    """
    # Step 1: 主模型 forward
    hidden_states = model.backbone(prefix)
    main_logits = model.lm_head(hidden_states)
    token_0 = sample(main_logits[:, -1])

    # Step 2: MTP heads 逐步生成 draft tokens
    draft_tokens = [token_0]
    draft_probs = [softmax(main_logits[:, -1])]

    prev_hidden = hidden_states[:, -1:]
    for k in range(min(gamma, len(mtp_modules))):
        prev_embed = model.embed_tokens(draft_tokens[-1])
        new_hidden, mtp_logits = mtp_modules[k](prev_hidden, prev_embed)
        prev_hidden = new_hidden

        probs = softmax(mtp_logits[:, -1])
        token = sample(probs)
        draft_tokens.append(token)
        draft_probs.append(probs)

    # Step 3: Target model 验证
    verify_input = torch.cat([prefix] + draft_tokens)
    target_logits = model.backbone_and_lm_head(verify_input)

    # Step 4: Rejection sampling
    accepted = rejection_sample(draft_tokens, draft_probs, target_logits)
    return accepted
```

## 4. Qwen3 的 MTP 支持

### 4.1 Qwen3 MTP 架构

Qwen3 系列模型同样原生支持 MTP，其设计与 DeepSeek-V3 类似但有一些差异：

```python
# Qwen3 MTP 的特点:
# 1. MTP heads 数量可配置 (默认 1-2)
# 2. 每个 MTP head 包含一个完整的 Transformer layer
# 3. 与主模型共享 tokenizer 和 embedding

class Qwen3MTPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.transformer_layer = Qwen3DecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size)
        # 共享主模型的 lm_head

    def forward(self, hidden_states, token_embeds):
        x = self.proj(torch.cat([hidden_states, token_embeds], dim=-1))
        x = self.transformer_layer(x)
        x = self.norm(x)
        return x
```

### 4.2 MTP 在不同模型系列中的对比

| 模型 | MTP heads 数量 | Head 类型 | 训练策略 | 推理支持 |
|------|---------------|-----------|---------|---------|
| Meta MTP (论文) | 4 | 独立 linear | 预训练 | 理论验证 |
| DeepSeek-V3 | 1 (可选 2) | Sequential module | 预训练 | vLLM/SGLang |
| Qwen3 | 1-2 | Transformer layer | 继续训练 | vLLM |
| LLaMA-4 | 未公开 | 未公开 | 预训练 | 未公开 |

## 5. vLLM 中的 MTP 实现

### 5.1 MTP 配置与启动

```bash
# 使用 DeepSeek-V3 的 MTP 进行投机解码
vllm serve deepseek-ai/DeepSeek-V3 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'

# 使用 Qwen3 的 MTP
vllm serve Qwen/Qwen3-32B \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

> **注意**：MTP 的 `num_speculative_tokens` 受限于模型训练时的 MTP heads 数量。DeepSeek-V3 默认只有 1 个 MTP head，因此 `num_speculative_tokens` 最大为 1。

### 5.2 模型层面的 MTP 实现

```python
# vllm/model_executor/models/deepseek_mtp.py (简化)
class DeepSeekMTP(nn.Module):
    """
    DeepSeek-V3 MTP Model for speculative decoding

    职责:
    1. 加载 MTP head 权重
    2. 提供 draft token 生成接口
    3. 管理 MTP 相关的 KV Cache
    """

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.norm = RMSNorm(config.hidden_size)

        # MTP 的 Transformer block
        self.layers = nn.ModuleList([
            DeepSeekMTPDecoderLayer(config)
        ])

        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor,       # 前一步采样的 token
        hidden_states: torch.Tensor,    # 主模型的 hidden states
        positions: torch.Tensor,
        **kwargs
    ):
        # 融合 hidden states 和 token embedding
        token_embeds = self.embed_tokens(input_ids)
        h = self.norm(hidden_states) + token_embeds

        # Transformer block
        for layer in self.layers:
            h = layer(h, positions=positions, **kwargs)

        logits = self.lm_head(h)
        return logits, h
```

### 5.3 MTP Proposer

```python
# vllm/v1/spec_decode/ 中的 MTP proposer (简化)
class MTPProposer:
    """
    MTP draft token 提议器

    核心逻辑:
    1. 接收 target model 的 hidden states
    2. 通过 MTP module 生成 draft tokens
    3. 返回 draft tokens 和对应的概率分布
    """

    def __init__(self, mtp_model, config):
        self.mtp_model = mtp_model
        self.num_draft_tokens = config.num_speculative_tokens

    def propose(self, hidden_states, sampled_token_ids, positions):
        """
        Args:
            hidden_states: [batch, hidden_size] - 主模型最后一层输出
            sampled_token_ids: [batch, 1] - 主模型采样的 token
            positions: position ids
        """
        draft_tokens = []
        draft_logits = []

        current_hidden = hidden_states
        current_token = sampled_token_ids

        for step in range(self.num_draft_tokens):
            logits, new_hidden = self.mtp_model(
                input_ids=current_token,
                hidden_states=current_hidden,
                positions=positions + step + 1,
            )

            # 采样 draft token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            draft_tokens.append(next_token)
            draft_logits.append(logits)
            current_hidden = new_hidden
            current_token = next_token

        return draft_tokens, draft_logits
```

## 6. MTP Loss 的设计细节

### 6.1 Loss 权重策略

不同位置的 MTP loss 通常使用不同的权重：

```python
def compute_mtp_loss_weighted(logits_list, targets_list, weights=None):
    """
    加权 MTP loss

    weights[k] 控制第 k 步预测的重要性
    常见策略:
    - 均匀权重: [1, 1, 1, 1]
    - 递减权重: [1, 0.5, 0.25, 0.125]
    - 学习权重: 可训练参数
    """
    if weights is None:
        weights = [1.0 / len(logits_list)] * len(logits_list)

    total_loss = 0
    for k, (logits, targets) in enumerate(zip(logits_list, targets_list)):
        loss_k = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='mean'
        )
        total_loss += weights[k] * loss_k

    return total_loss
```

### 6.2 训练效率优化

MTP 训练相比标准 NTP 有额外的计算开销（多个 head 的 forward + backward）。常见优化：

```python
# 优化 1: 共享 unembedding matrix
# 所有 MTP heads 共享同一个输出投影矩阵
class SharedMTPHeads(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads):
        super().__init__()
        # 共享的输出投影
        self.shared_lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 每个 head 有独立的 hidden state 变换
        self.head_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            for _ in range(num_heads)
        ])

    def forward(self, hidden_states):
        logits = []
        for transform in self.head_transforms:
            h = transform(hidden_states)
            logits.append(self.shared_lm_head(h))
        return logits

# 优化 2: Gradient checkpointing for MTP heads
# 在反向传播时重计算 MTP heads 的激活，节省显存
```

### 6.3 MTP 与 Knowledge Distillation 的结合

一个有趣的训练策略是将 MTP 与 knowledge distillation 结合——用更大的 teacher model 的多步预测来监督 student model 的 MTP heads：

$$
\mathcal{L}_{\text{MTP-KD}} = \sum_{k=1}^{K} \lambda_k \cdot \text{KL}(p_k^{\text{student}} \| p_k^{\text{teacher}})
$$

这在 DeepSeek-V3 的技术报告中有所提及——用 DeepSeek-V2.5 作为 teacher 来辅助 V3 的 MTP 训练。

## 7. MTP 的优势与局限

### 7.1 优势

1. **无额外推理开销**：MTP heads 在推理时直接作为 draft，不需要额外的 draft model
2. **高 draft 质量**：backbone 为多步预测优化过，draft 接受率更高
3. **训练正则化**：MTP loss 本身提升了模型质量
4. **工程简洁**：模型自带 draft 能力，部署配置简单

### 7.2 局限

1. **训练成本高**：需要从头训练或大规模继续训练，不适合已有模型
2. **MTP heads 数量有限**：目前主流模型的 MTP heads 只有 1-2 个，draft 长度受限
3. **不是所有模型都支持**：需要模型架构原生支持 MTP
4. **最优 MTP 策略仍在探索**：loss 权重、head 数量、架构选择等超参数尚无统一结论

### 7.3 MTP 的未来

随着 DeepSeek-V3、Qwen3 等模型的成功，MTP 正在成为新模型架构的标准组件。可以预见：

- 更多模型会原生支持 MTP（2-4 个 heads）
- MTP heads 的架构会进一步优化（从简单 MLP 到更复杂的 module）
- MTP 与其他投机解码方案（如 EAGLE-style tree drafting）的结合
- 推理框架（vLLM、SGLang、TensorRT-LLM）会将 MTP 作为一等公民支持

---

> **下一节**：[05-draft-selection.md](05-draft-selection.md) — Draft Model 选择与 N-gram 方案
