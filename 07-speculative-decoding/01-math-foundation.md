# 投机解码数学基础

> 投机解码（Speculative Decoding）的核心洞察：autoregressive decoding 的瓶颈在于串行化，而非计算量。通过引入一个"便宜的猜测 + 昂贵的验证"范式，我们可以在**数学上保证无损**的前提下实现显著加速。

## 1. 问题背景：为什么自回归解码慢？

LLM 推理中 decode 阶段的每一步只生成一个 token，但每步都需要加载全部模型权重——这是典型的 **memory-bound** 操作。对于一个参数量为 $P$ 的模型，生成 $N$ 个 token 需要 $N$ 次 forward pass，每次都要从 HBM 加载 $O(P)$ 的数据。

核心矛盾在于：GPU 的计算能力（FLOPS）远超内存带宽能供给的数据量。Decode 阶段的 arithmetic intensity 极低（每个 token 的 batch size 通常为 1），GPU 大量计算单元处于空闲状态。

**投机解码的思路**：既然每步 forward pass 的计算资源被严重浪费，能否让 target model 一次 forward pass "同时"验证多个 token？答案是肯定的——prefill 阶段本身就是并行处理多个 token 的。

## 2. 基本框架

投机解码涉及两个模型：

| 角色 | 模型 | 特点 |
|------|------|------|
| **Draft model** $M_q$ | 小模型 / 轻量级 head | 快速生成候选 token，分布为 $q(x)$ |
| **Target model** $M_p$ | 大模型（原始模型） | 准确但慢，分布为 $p(x)$ |

**基本流程**：

```
1. Draft model 自回归生成 γ 个 candidate tokens: x₁, x₂, ..., xᵧ
2. 记录每个位置的 draft 分布: q(x₁), q(x₂|x₁), ..., q(xᵧ|x₁...xᵧ₋₁)
3. Target model 并行前向传播，一次性计算所有位置的分布:
   p(x₁), p(x₂|x₁), ..., p(xᵧ|x₁...xᵧ₋₁), p(xᵧ₊₁|x₁...xᵧ)
4. 从左到右逐个用 rejection sampling 决定是否接受
5. 第一个被拒绝的位置从修正分布重新采样，后续全部丢弃
```

> **关键洞察**：Target model 的 forward pass 可以并行处理 $\gamma$ 个位置（类似 prefill），其延迟与生成单个 token 几乎相同（在 batch size 较小时）。因此，如果 $\gamma$ 个候选 token 中有 $k$ 个被接受，我们在约 1 次 forward pass 的时间内生成了 $k+1$ 个 token。

## 3. 核心定理：Rejection Sampling 保证无损性

### 3.1 Modified Rejection Sampling

对于每个候选位置 $t$，draft model 提议 token $x \sim q(x)$，我们使用如下规则决定是否接受：

$$
\text{accept } x \text{ with probability } \min\left(1, \frac{p(x)}{q(x)}\right)
$$

具体操作：采样一个均匀随机变量 $r \sim U(0,1)$，如果 $r < \min\left(1, \frac{p(x)}{q(x)}\right)$，则接受 $x$。

**两种情况**：
- 当 $p(x) \geq q(x)$ 时：一定接受（draft model 低估了这个 token 的概率）
- 当 $p(x) < q(x)$ 时：以概率 $p(x)/q(x)$ 接受（draft model 高估了）

### 3.2 无损性证明

**定理**：经过上述 rejection sampling，被接受的 token 服从 target model 的分布 $p(x)$。

**证明**：

被接受的 token $x$ 的分布为：

$$
P(\text{output} = x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)
$$

分两种情况讨论：

**Case 1**：$p(x) \geq q(x)$

$$
q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = q(x) \cdot 1 = q(x)
$$

但这不等于 $p(x)$？——注意，这里还没考虑归一化。实际上需要考虑接受的概率：

$$
P(\text{output} = x | \text{accepted}) = \frac{q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)}{\sum_{x'} q(x') \cdot \min\left(1, \frac{p(x')}{q(x')}\right)}
$$

设 $\beta = \sum_{x'} q(x') \cdot \min\left(1, \frac{p(x')}{q(x')}\right)$ 为总接受概率。

$$
\beta = \sum_{x': p(x') \geq q(x')} q(x') + \sum_{x': p(x') < q(x')} p(x')
$$

我们来计算条件分布。对于任意 token $x$：

$$
q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))
$$

但 $\min(q(x), p(x)) \neq p(x)$，那如何保证无损？

**关键**：拒绝时我们从**修正分布**重新采样，而不是简单丢弃。整个过程产出的分布才等于 $p(x)$。

### 3.3 修正分布（Residual Distribution）

当 token $x$ 被拒绝时，我们从以下修正分布重新采样：

$$
p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{x'} \max(0, p(x') - q(x'))}
$$

**完整证明**：输出 token 的分布为（分"被接受"和"被拒绝后重采样"两种情况）：

$$
P(\text{output} = x) = \underbrace{q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right)}_{\text{accepted}} + \underbrace{(1-\beta) \cdot p'(x)}_{\text{resampled}}
$$

其中 $\beta = \sum_{x'} \min(q(x'), p(x'))$ 是接受的总概率。

$$
= \min(q(x), p(x)) + \left(1 - \sum_{x'}\min(q(x'), p(x'))\right) \cdot \frac{\max(0, p(x) - q(x))}{\sum_{x'}\max(0, p(x') - q(x'))}
$$

注意分母 $\sum_{x'}\max(0, p(x') - q(x')) = 1 - \sum_{x'}\min(q(x'), p(x')) = 1 - \beta$（这是因为 $\max(0, a-b) = a - \min(a,b)$，对所有 $x'$ 求和且 $\sum p = \sum q = 1$）。

因此：

$$
P(\text{output} = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x) \quad \blacksquare
$$

这是因为 $\min(a,b) + \max(0, a-b) = a$ 对所有 $a, b \geq 0$ 成立。

> **直觉理解**：接受步骤"消耗"了 $p(x)$ 和 $q(x)$ 的重叠部分（$\min(p,q)$），拒绝后的重采样补充了 $p(x)$ 超出 $q(x)$ 的部分（$\max(0, p-q)$）。两者之和恰好等于 $p(x)$。

## 4. 多步验证：链式 Rejection Sampling

实际中 draft model 生成 $\gamma$ 个 token，验证按顺序进行：

```python
# 伪代码：speculative decoding 验证过程
def verify(draft_tokens, draft_probs, target_probs, gamma):
    accepted = []
    for i in range(gamma):
        x = draft_tokens[i]
        q_x = draft_probs[i][x]   # draft model 对 token x 的概率
        p_x = target_probs[i][x]  # target model 对 token x 的概率

        r = random.uniform(0, 1)
        if r < min(1, p_x / q_x):
            # 接受
            accepted.append(x)
        else:
            # 拒绝：从修正分布采样一个 token
            residual = np.maximum(0, target_probs[i] - draft_probs[i])
            residual /= residual.sum()
            new_token = np.random.choice(vocab_size, p=residual)
            accepted.append(new_token)
            return accepted  # 后续 draft tokens 全部丢弃

    # 全部接受！额外从 target model 的最后一个分布采样 bonus token
    bonus = np.random.choice(vocab_size, p=target_probs[gamma])
    accepted.append(bonus)
    return accepted
```

**关键点**：

1. **验证必须从左到右**：因为位置 $i+1$ 的条件分布依赖于位置 $i$ 的确定结果
2. **第一个拒绝位置**：从修正分布采样替代 token，后续全部丢弃
3. **全部接受时的 bonus token**：target model 在 $\gamma+1$ 位置的分布已经计算出来了，可以直接采样一个额外 token——这保证了每次迭代至少产出 1 个 token（即使第一个就被拒绝）

## 5. 期望加速比分析

### 5.1 接受概率

定义 **token-level 接受率** $\alpha$：

$$
\alpha = \mathbb{E}_{x \sim q}\left[\min\left(1, \frac{p(x)}{q(x)}\right)\right] = \sum_x \min(q(x), p(x)) = 1 - D_{TV}(p, q)
$$

其中 $D_{TV}(p,q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ 是 total variation distance。

$\alpha$ 越大（$p$ 和 $q$ 越相似），接受率越高。

### 5.2 期望接受 token 数

假设每个位置的接受概率独立且均为 $\alpha$（简化假设），在 $\gamma$ 个候选中，期望接受的 token 数为：

$$
E[\text{accepted tokens}] = \sum_{k=0}^{\gamma} P(\text{前 } k \text{ 个全被接受且第 } k+1 \text{ 个被拒绝}) \cdot (k + 1)
$$

这里 $+1$ 是因为拒绝位置会从修正分布采样一个 token（或者全部接受时有 bonus token）。

$$
E[\text{tokens per iteration}] = \sum_{k=0}^{\gamma-1} \alpha^k (1-\alpha)(k+1) + \alpha^\gamma (\gamma+1)
$$

化简得到：

$$
\boxed{E[\text{tokens per iteration}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}}
$$

**推导过程**：

设 $S = \sum_{k=0}^{\gamma-1} \alpha^k(1-\alpha)(k+1) + \alpha^\gamma(\gamma+1)$。

注意到 $\sum_{k=0}^{\gamma-1}\alpha^k(1-\alpha)(k+1) = (1-\alpha)\sum_{k=0}^{\gamma-1}(k+1)\alpha^k$。

利用等差-等比级数求和公式 $\sum_{k=0}^{n}(k+1)x^k = \frac{1-(n+2)x^{n+1}+(n+1)x^{n+2}}{(1-x)^2}$：

$$
(1-\alpha) \cdot \frac{1 - (\gamma+1)\alpha^\gamma + \gamma\alpha^{\gamma+1}}{(1-\alpha)^2} + \alpha^\gamma(\gamma+1)
$$

$$
= \frac{1 - (\gamma+1)\alpha^\gamma + \gamma\alpha^{\gamma+1}}{1-\alpha} + (\gamma+1)\alpha^\gamma
$$

$$
= \frac{1 - (\gamma+1)\alpha^\gamma + \gamma\alpha^{\gamma+1} + (\gamma+1)\alpha^\gamma(1-\alpha)}{1-\alpha}
$$

$$
= \frac{1 - (\gamma+1)\alpha^\gamma + \gamma\alpha^{\gamma+1} + (\gamma+1)\alpha^\gamma - (\gamma+1)\alpha^{\gamma+1}}{1-\alpha}
$$

$$
= \frac{1 - \alpha^{\gamma+1}}{1-\alpha} \quad \blacksquare
$$

### 5.3 数值分析

| $\alpha$ | $\gamma=3$ | $\gamma=5$ | $\gamma=7$ |
|----------|-----------|-----------|-----------|
| 0.5      | 1.88      | 1.97      | 1.99      |
| 0.7      | 2.95      | 3.64      | 3.99      |
| 0.8      | 3.69      | 5.03      | 6.01      |
| 0.9      | 4.69      | 7.10      | 9.17      |
| 0.95     | 5.22      | 8.45      | 11.57     |

**关键观察**：

1. 当 $\alpha$ 较低时（$<0.6$），增大 $\gamma$ 收益递减——因为后面的 token 几乎不可能被接受
2. 当 $\alpha$ 较高时（$>0.8$），增大 $\gamma$ 有显著收益
3. 最优 $\gamma$ 取决于 draft model 的速度比——draft model 越快（相对 target model），$\gamma$ 可以越大

### 5.4 实际加速比

期望 tokens per iteration 并不直接等于加速比，还需要考虑 draft model 的开销：

$$
\text{Speedup} = \frac{E[\text{tokens per iteration}]}{1 + \gamma \cdot c}
$$

其中 $c = T_{\text{draft}} / T_{\text{target}}$ 是 draft model 相对 target model 的时间比。

- 如果 draft model 很快（$c \to 0$，如 EAGLE/Medusa 的轻量级 head）：加速比 $\approx E[\text{tokens}]$
- 如果 draft model 较慢（$c = 0.1$，如独立的小模型）：加速比会打折

最优 $\gamma$ 可以通过求导得到。实践中通常设为 3-7。

## 6. Greedy 模式 vs Sampling 模式

### 6.1 Greedy Decoding (temperature = 0)

当 target model 使用 greedy decoding 时，$p(x)$ 是一个 one-hot 分布（最大概率的 token 概率为 1，其他为 0）。

验证简化为：如果 draft token 等于 target model 的 argmax token，则接受；否则拒绝并用 target 的 argmax 替换。

```python
# Greedy 模式简化
if draft_token == target_argmax:
    accept
else:
    reject, use target_argmax instead
```

这种情况下 rejection sampling 退化为简单的 token 比较，**不存在修正分布采样**——因为 greedy 模式下 target 的选择是确定性的。

### 6.2 Sampling 模式 (temperature > 0)

这是完整 rejection sampling 发挥作用的场景。需要注意的几个细节：

**Temperature 的影响**：温度越低，$p$ 和 $q$ 的分布越尖锐，它们的重叠（$\alpha$）通常越高——因为两个模型大概率在 top token 上达成一致。温度越高，分布越平坦，$\alpha$ 可能降低。

**Top-k / Top-p 采样的处理**：

当使用 top-k 或 top-p 截断时，需要在截断后的分布上做 rejection sampling：

```python
# 1. 对 q 和 p 分别应用 top-k/top-p 截断
q_truncated = apply_topk_topp(q_logits, top_k, top_p)
p_truncated = apply_topk_topp(p_logits, top_k, top_p)

# 2. 重新归一化
q_truncated /= q_truncated.sum()
p_truncated /= p_truncated.sum()

# 3. 在截断分布上做 rejection sampling
ratio = p_truncated[x] / q_truncated[x]
accept_prob = min(1, ratio)
```

**注意**：如果 draft model 提议的 token 不在 target model 的 top-k/top-p 范围内（$p_{\text{truncated}}(x) = 0$），则一定被拒绝。

### 6.3 典型 Acceptance（Typical Acceptance）

Medusa 论文提出了一种放宽的验证策略——典型接受（Typical Acceptance），它不严格保证无损，但在实践中质量损失极小，同时大幅提升接受率：

$$
\text{accept } x \text{ if } p(x) > \epsilon
$$

其中 $\epsilon$ 是一个小阈值。只要 target model 认为这个 token"足够合理"，就接受——不管 $q(x)$ 是多少。这种策略在 temperature 较高的采样场景下特别有效。

## 7. 与 Batch Size 的交互

投机解码在不同 batch size 下的表现差异很大：

**Batch size = 1**（单请求）：
- Decode 阶段严重 memory-bound，GPU 利用率极低
- 投机解码收益最大（将 $\gamma$ 个串行步骤合并为 1 步验证）

**大 Batch size**（continuous batching）：
- Decode 阶段的 arithmetic intensity 提升，GPU 利用率较高
- 验证阶段要为每个请求处理 $\gamma$ 个额外 token 的 attention，计算开销不再可忽略
- 收益递减，甚至可能负优化

这也是为什么投机解码在 **latency-sensitive** 场景（单请求或小 batch）中最有价值，而在 **throughput-oriented** 场景中需要谨慎评估。

## 8. 算法伪代码总结

```python
def speculative_decoding(target_model, draft_model, prefix, gamma, max_tokens):
    """
    Speculative Decoding 完整算法

    Args:
        target_model: target model M_p, 产生分布 p(x)
        draft_model: draft model M_q, 产生分布 q(x)
        prefix: 输入 token 序列
        gamma: draft length（每轮生成的候选 token 数）
        max_tokens: 最大生成 token 数
    """
    generated = []
    context = prefix

    while len(generated) < max_tokens:
        # === Step 1: Draft Phase ===
        draft_tokens = []
        draft_probs = []
        draft_context = context
        for _ in range(gamma):
            q = draft_model.forward(draft_context)  # 得到分布 q(x)
            x = sample(q)
            draft_tokens.append(x)
            draft_probs.append(q)
            draft_context = draft_context + [x]

        # === Step 2: Verification Phase ===
        # Target model 并行计算所有位置的分布
        # 输入: context + draft_tokens (共 gamma+1 个位置需要分布)
        target_probs = target_model.forward(context + draft_tokens)
        # target_probs[i] 是位置 i 的 target 分布, i = 0..gamma

        # === Step 3: Acceptance-Rejection ===
        n_accepted = 0
        for i in range(gamma):
            x = draft_tokens[i]
            r = uniform(0, 1)
            if r < min(1, target_probs[i][x] / draft_probs[i][x]):
                # 接受
                generated.append(x)
                n_accepted += 1
            else:
                # 拒绝：从修正分布采样
                residual = max(0, target_probs[i] - draft_probs[i])
                residual /= residual.sum()
                new_token = sample(residual)
                generated.append(new_token)
                break

        if n_accepted == gamma:
            # 全部接受，bonus token
            bonus = sample(target_probs[gamma])
            generated.append(bonus)

        # 更新 context
        context = prefix + generated

    return generated
```

## 9. 参考文献

| 论文 | 核心贡献 |
|------|----------|
| [Leviathan et al., 2022](https://arxiv.org/abs/2211.17192) | 提出 Speculative Decoding 原始框架和无损性证明 |
| [Chen et al., 2023](https://arxiv.org/abs/2302.01318) | 独立提出类似方法（Accelerating LLM Inference with Staged Speculative Decoding） |
| [Sun et al., 2024](https://arxiv.org/abs/2401.10774) | Medusa: Typical Acceptance 策略 |
| [Xia et al., 2024](https://arxiv.org/abs/2304.04487) | Unlocking Efficiency in LLM Inference（speculative decoding 综述） |

---

> **下一节**：[02-eagle.md](02-eagle.md) — EAGLE 系列：用 target model 的 hidden states 构建高效 draft
