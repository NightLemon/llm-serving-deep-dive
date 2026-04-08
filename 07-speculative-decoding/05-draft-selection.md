# Draft Model 选择与 N-gram 方案

> 投机解码的效果很大程度上取决于 draft 方案的选择。前面我们讨论了 EAGLE、Medusa、MTP 等需要额外训练的方案。本节聚焦于**无需训练**或**低成本**的 draft 策略——包括独立 draft model 的选择、N-gram 匹配、Suffix Decoding、Lookahead Decoding 等。

## 1. 独立 Draft Model 选择

### 1.1 同系列小模型

最直观的方案：用同系列的小模型作为 draft model。

| Target Model | Draft Model | 参数比 | 典型接受率 $\alpha$ | 加速比 |
|-------------|-------------|--------|--------------------|----|
| LLaMA-3.1-70B | LLaMA-3.1-8B | 11.4% | 0.6-0.8 | 1.8-2.5x |
| Qwen2.5-72B | Qwen2.5-7B | 9.7% | 0.6-0.8 | 1.8-2.5x |
| Mixtral-8x22B | Mixtral-8x7B | ~25% | 0.5-0.7 | 1.5-2.2x |
| GPT-4 (推测) | GPT-3.5 (推测) | ~10% | 0.5-0.7 | 1.5-2.0x |

**选择原则**：

```python
# 评估 draft model 质量的关键指标
def evaluate_draft_model(target_model, draft_model, eval_dataset):
    """
    评估 draft model 的质量

    核心指标:
    1. Token-level acceptance rate (α): 越高越好
    2. Draft model latency ratio (c): 越低越好
    3. 综合加速比 = E[accepted] / (1 + γ*c)
    """
    total_tokens = 0
    accepted_tokens = 0

    for prompt in eval_dataset:
        draft_logits = draft_model(prompt)   # draft 分布
        target_logits = target_model(prompt) # target 分布

        for t in range(len(draft_logits)):
            p = softmax(target_logits[t])
            q = softmax(draft_logits[t])
            # α = Σ min(p(x), q(x))
            alpha_t = torch.min(p, q).sum().item()
            total_tokens += 1
            accepted_tokens += alpha_t

    avg_alpha = accepted_tokens / total_tokens

    # 延迟比
    draft_time = measure_latency(draft_model)
    target_time = measure_latency(target_model)
    c = draft_time / target_time

    return avg_alpha, c
```

### 1.2 量化版本作为 Draft

一种巧妙的方案：用 target model 的量化版本作为 draft model。

**优势**：
- 分布对齐好（同一个模型，只是精度不同），接受率 $\alpha$ 高
- 不需要额外加载不同的模型
- 量化模型推理速度快

```bash
# vLLM 中使用量化 draft model
vllm serve meta-llama/Llama-3.1-70B \
    --speculative-model meta-llama/Llama-3.1-70B-FP8 \
    --num-speculative-tokens 5
```

**挑战**：
- 量化模型仍然很大（70B FP8 ≈ 70GB），显存开销高
- 需要同时在 GPU 上放两个版本的模型
- 在小 GPU 上不可行

### 1.3 MLPSpeculator

MLPSpeculator 是一种极轻量级的 draft 方案——用一个小 MLP 网络直接预测多个 token 的 embedding：

```python
class MLPSpeculator(nn.Module):
    """
    MLPSpeculator: 用极小的 MLP 预测未来 token embeddings

    架构特点:
    1. 输入: target model 的最后一层 hidden state
    2. 输出: 未来 K 个位置的 token embeddings
    3. 参数量极小 (< 10M)
    4. 推理速度极快
    """
    def __init__(self, hidden_size, embed_size, num_speculative_tokens):
        super().__init__()
        self.num_tokens = num_speculative_tokens

        # 每个位置有一个独立的小 MLP
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, embed_size),
            )
            for _ in range(num_speculative_tokens)
        ])

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, hidden_size]
        Returns:
            predicted_embeds: list of [batch, embed_size]
        """
        return [head(hidden_states) for head in self.heads]
```

MLPSpeculator 的接受率通常较低（0.3-0.5），但由于推理速度极快（$c \approx 0.01$），在某些场景下仍然有用。IBM 的 `ibm-fms/llama3-8b-speculator` 是一个公开的 MLPSpeculator 示例。

## 2. N-gram Speculation

### 2.1 核心思想

N-gram speculation 是最简单的投机解码方案——**完全不需要额外模型**。

思路：从已有的 context（prompt + 已生成的 token）中匹配 n-gram pattern，用匹配到的后续 token 作为 draft。

```
已有 context:
"The quick brown fox jumps over the lazy dog. The quick brown fox"

当前需要预测 "fox" 之后的 token。
在 context 中搜索以 "fox" 结尾的 n-gram：
  找到 "brown fox jumps over"

Draft: ["jumps", "over"]
```

### 2.2 算法

```python
class NGramSpeculator:
    """
    N-gram based speculation

    从 prompt 和已生成文本中匹配 n-gram pattern
    """
    def __init__(self, n=3, num_speculative_tokens=5):
        self.n = n  # n-gram 的 n
        self.num_speculative_tokens = num_speculative_tokens

    def propose(self, token_ids):
        """
        Args:
            token_ids: 已有的所有 token (prompt + generated)
        Returns:
            draft_tokens: 提议的 token 序列
        """
        if len(token_ids) < self.n:
            return []  # context 太短，无法匹配

        # 当前的 n-gram 后缀
        suffix = tuple(token_ids[-(self.n - 1):])

        # 在 context 中搜索匹配的 n-gram
        matches = []
        for i in range(len(token_ids) - self.n):
            ngram = tuple(token_ids[i:i + self.n - 1])
            if ngram == suffix:
                # 找到匹配！收集后续 token
                continuation = token_ids[i + self.n - 1:]
                if len(continuation) > 0:
                    matches.append(continuation)

        if not matches:
            return []  # 无匹配

        # 选择最长的匹配（或最近的匹配）
        best_match = max(matches, key=len)
        return best_match[:self.num_speculative_tokens]
```

### 2.3 适用场景

N-gram speculation 在以下场景下特别有效：

| 场景 | 原因 | 预期加速 |
|------|------|----------|
| **代码补全** | 代码有大量重复 pattern（函数名、变量名、import 语句） | 2-4x |
| **翻译** | 某些短语会反复出现 | 1.5-2x |
| **对话（有 system prompt）** | 格式化输出中的模板文本 | 1.5-3x |
| **JSON 生成** | 键名、格式反复出现 | 2-3x |
| **文档总结** | 源文本中的术语会在摘要中出现 | 1.5-2x |

**不适用场景**：创意写作、开放问答等 context 中没有可匹配 pattern 的任务。

### 2.4 vLLM 中的配置

```bash
# 使用 N-gram speculation
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4 \
    --ngram-prompt-lookup-min 2
```

**参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--speculative-model "[ngram]"` | 指定使用 n-gram speculation | 固定值 |
| `--num-speculative-tokens` | 最大 draft 长度 | 3-7 |
| `--ngram-prompt-lookup-max` | n-gram 匹配的最大 n | 3-5 |
| `--ngram-prompt-lookup-min` | n-gram 匹配的最小 n | 1-2 |

> **小技巧**：`min` 和 `max` 定义了搜索的 n-gram 范围。先尝试长 n-gram（更精确但可能匹配不到），逐步退回到短 n-gram。

## 3. Suffix Decoding

### 3.1 核心思想

Suffix Decoding 是 N-gram speculation 的增强版——使用 **suffix tree（后缀树）** 或 **suffix array（后缀数组）** 数据结构来高效搜索匹配：

```
传统 N-gram: 只匹配固定长度的 n-gram, O(n * L) 搜索
Suffix Decoding: 使用 suffix tree, O(n) 搜索任意长度的匹配
```

### 3.2 Suffix Tree 构建

```python
class SuffixTree:
    """
    后缀树用于高效匹配 token pattern

    构建: O(L) (Ukkonen's algorithm)
    查询: O(m) (m = query length)
    """
    def __init__(self):
        self.root = {}

    def build(self, token_ids):
        """从 token 序列构建后缀树"""
        for i in range(len(token_ids)):
            node = self.root
            for j in range(i, len(token_ids)):
                token = token_ids[j]
                if token not in node:
                    node[token] = {'_children': {}, '_positions': []}
                node[token]['_positions'].append(i)
                node = node[token]['_children']

    def search(self, query, max_continuation=10):
        """
        搜索 query 在后缀树中的匹配，返回最长的后续 token 序列
        """
        node = self.root
        for token in query:
            if token not in node:
                return []
            node = node[token]['_children']

        # 找到匹配，提取后续 tokens
        return self._extract_continuation(node, max_continuation)
```

### 3.3 动态更新

Suffix decoding 的一个重要优势是可以**动态更新**——随着生成的推进，新生成的 token 也被加入 suffix tree：

```python
class DynamicSuffixSpeculator:
    def __init__(self, prompt_tokens, n=3, max_draft=5):
        self.suffix_tree = SuffixTree()
        self.suffix_tree.build(prompt_tokens)
        self.all_tokens = list(prompt_tokens)
        self.n = n
        self.max_draft = max_draft

    def propose_and_update(self, new_tokens):
        """生成 draft 并更新 suffix tree"""
        # 将新 token 加入 suffix tree
        self.all_tokens.extend(new_tokens)
        self.suffix_tree.update(new_tokens)

        # 搜索匹配
        query = self.all_tokens[-(self.n - 1):]
        continuation = self.suffix_tree.search(query, self.max_draft)
        return continuation
```

## 4. Lookahead Decoding

### 4.1 核心思想：Jacobi 迭代

Lookahead Decoding 使用了一个完全不同的思路——**Jacobi 迭代**。它不使用 draft model，而是利用 target model 本身的"并行猜测-验证"能力：

```
标准自回归:
x₁ → x₂ → x₃ → x₄ (串行, 4 步)

Jacobi 迭代:
Step 0: x₁,  ?,  ?,  ?   (初始化：随机或启发式猜测)
Step 1: x₁, x₂', x₃', x₄'  (并行更新所有位置)
Step 2: x₁, x₂,  x₃', x₄'  (x₂ 收敛了)
Step 3: x₁, x₂,  x₃,  x₄'  (x₃ 收敛了)
Step 4: x₁, x₂,  x₃,  x₄   (全部收敛)
```

**关键洞察**：自回归语言模型的 next-token prediction 可以看作一个不动点迭代问题。如果我们并行地猜测多个位置的 token，然后用模型更新每个位置，这个过程最终会收敛到正确的自回归序列。

### 4.2 Lookahead Decoding 的具体实现

```python
def lookahead_decode(model, prefix, window_size=5, max_steps=20):
    """
    Lookahead Decoding via Jacobi iteration

    Args:
        model: target model
        prefix: 输入序列
        window_size: lookahead 窗口大小 (W)
        max_steps: 最大 Jacobi 迭代步数
    """
    context = list(prefix)
    # 维护一个 lookahead window
    # window[i] = 位置 len(context)+i 的当前猜测
    window = [random_token() for _ in range(window_size)]

    n_gram_pool = {}  # 用于 n-gram matching 的历史池

    while True:
        # 构造输入: context + window
        input_ids = context + window

        # 模型并行 forward（所有 window 位置同时计算）
        logits = model.forward(input_ids)

        # 更新每个 window 位置
        new_window = []
        accepted_prefix = 0

        for i in range(window_size):
            pos = len(context) + i
            new_token = logits[pos - 1].argmax()  # greedy

            if i == 0 or (accepted_prefix == i and new_token == window[i]):
                # 这个位置已经收敛
                accepted_prefix = i + 1

            new_window.append(new_token)

        # 收集 n-grams 用于未来的猜测
        collect_ngrams(n_gram_pool, new_window)

        # 将收敛的 prefix 移入 context
        context.extend(new_window[:accepted_prefix])

        # 用 n-gram pool 初始化新的 window
        window = initialize_window(n_gram_pool, context, window_size)
```

### 4.3 Lookahead 的优缺点

**优势**：
- 不需要任何额外模型或训练
- 利用 target model 自身的能力
- 理论上可以处理任意长度的 lookahead

**劣势**：
- 收敛速度不确定（依赖于输入的"难度"）
- 每步需要处理整个 window 的 attention，计算开销大
- 在 sampling 模式下不直接适用（Jacobi 迭代假设确定性映射）
- 实际加速比通常低于 EAGLE/Medusa

### 4.4 LLMA（Lookahead with Matching and Adaptation）

LLMA 是 Lookahead Decoding 的一个变体，将 Jacobi 迭代与 reference text matching 结合：

- 如果有参考文本（如翻译的源文本、摘要的原文），直接从参考文本中选取候选 token
- 这本质上是一种特化的 N-gram speculation

## 5. 各方案的综合对比

### 5.1 方案选择决策树

```
需要投机解码吗？
├─ 延迟优先 + 小 batch → 是
│   ├─ 模型原生支持 MTP？
│   │   ├─ 是 (DeepSeek-V3, Qwen3) → 使用 MTP
│   │   └─ 否 → 继续
│   ├─ 有对应的 EAGLE head？
│   │   ├─ 是 → 使用 EAGLE-2
│   │   └─ 否 → 继续
│   ├─ 愿意训练 Medusa heads？
│   │   ├─ 是 → 训练并使用 Medusa
│   │   └─ 否 → 继续
│   ├─ 有同系列小模型？
│   │   ├─ 是 → 使用 draft model
│   │   └─ 否 → 继续
│   ├─ 输入有重复 pattern？
│   │   ├─ 是 (代码, JSON, 模板) → N-gram Speculation
│   │   └─ 否 → 考虑 Lookahead 或不用投机解码
│   └─ 吞吐优先 + 大 batch → 通常不需要投机解码
└─ 否
```

### 5.2 综合对比表

| 方案 | 额外模型/训练 | 显存开销 | 接受率 | 加速比 | 适用场景 |
|------|-------------|---------|--------|--------|---------|
| **EAGLE-2** | EAGLE head (训练) | 小 | 高 | 3-4x | 通用 |
| **Medusa** | Medusa heads (训练) | 很小 | 中 | 2-3x | 通用 |
| **MTP** | 模型原生 | 零 | 高 | 2-3x | 支持 MTP 的模型 |
| **Draft Model** | 独立小模型 | 大 | 中 | 1.5-2.5x | 有同系列模型 |
| **量化 Draft** | 量化版模型 | 大 | 较高 | 1.5-2.5x | 显存充足 |
| **MLPSpeculator** | 小 MLP (训练) | 极小 | 低 | 1.3-1.8x | 资源受限 |
| **N-gram** | 无 | 零 | 变动大 | 1-3x | 重复 pattern 场景 |
| **Suffix Decoding** | 无 | 小 | 变动大 | 1-3x | 重复 pattern 场景 |
| **Lookahead** | 无 | 小 | 变动大 | 1-2x | 通用但加速有限 |

### 5.3 关键观察

1. **"免费午餐"不存在**：高接受率方案（EAGLE、MTP）通常需要训练；免训练方案（N-gram、Lookahead）接受率不稳定
2. **场景依赖性强**：N-gram 在代码补全上可能达到 4x 加速，但在创意写作上完全无效
3. **batch size 是关键**：所有方案在大 batch 时效果都会下降
4. **"够用就行"原则**：在已经 compute-bound 的场景（大 batch），投机解码可能是负优化

## 6. vLLM 统一配置

vLLM 对各种投机解码方案提供了统一的配置接口：

```bash
# === 方案 1: EAGLE ===
vllm serve $MODEL \
    --speculative-model $EAGLE_HEAD \
    --num-speculative-tokens 5

# === 方案 2: Medusa ===
vllm serve $MODEL \
    --speculative-model $MEDUSA_HEAD \
    --num-speculative-tokens 5

# === 方案 3: MTP (模型原生) ===
vllm serve deepseek-ai/DeepSeek-V3 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'

# === 方案 4: Draft Model ===
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 5

# === 方案 5: N-gram ===
vllm serve $MODEL \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 4

# === 方案 6: MLPSpeculator ===
vllm serve $MODEL \
    --speculative-model ibm-fms/llama3-8b-speculator \
    --num-speculative-tokens 5
```

**通用参数**：

| 参数 | 说明 |
|------|------|
| `--spec-decoding-acceptance-method` | `rejection_sampler` (默认, 无损) 或 `typical_acceptance` |
| `--num-speculative-tokens` | Draft 长度 $\gamma$ |
| `--speculative-disable-mqa-scorer` | 禁用 MQA scorer 优化 |
| `--speculative-max-model-len` | Draft model 的最大序列长度 |

## 7. 实践建议

### 7.1 性能测试方法

在选择方案前，建议用实际工作负载做 benchmarking：

```python
# 测试脚本框架
import time
from vllm import LLM, SamplingParams

def benchmark_spec_decoding(model_name, spec_config, prompts, warmup=5):
    """
    对比有/无投机解码的延迟
    """
    # Baseline: 无投机解码
    llm_base = LLM(model=model_name)
    params = SamplingParams(max_tokens=256, temperature=0)

    # Warmup
    for p in prompts[:warmup]:
        llm_base.generate([p], params)

    # 计时
    start = time.time()
    outputs_base = llm_base.generate(prompts, params)
    base_time = time.time() - start

    # 投机解码
    llm_spec = LLM(model=model_name, **spec_config)

    for p in prompts[:warmup]:
        llm_spec.generate([p], params)

    start = time.time()
    outputs_spec = llm_spec.generate(prompts, params)
    spec_time = time.time() - start

    # 验证输出一致性 (greedy 模式下应该完全相同)
    for o_base, o_spec in zip(outputs_base, outputs_spec):
        assert o_base.outputs[0].text == o_spec.outputs[0].text, \
            "投机解码输出与 baseline 不一致！"

    speedup = base_time / spec_time
    print(f"Speedup: {speedup:.2f}x")
    return speedup
```

### 7.2 调参指南

| 场景 | 推荐 $\gamma$ | 推荐方案 | 备注 |
|------|---------------|---------|------|
| 单请求延迟优化 | 5-7 | EAGLE-2 / MTP | 加速比最大化 |
| 中等 batch (4-16) | 3-5 | EAGLE / Medusa | 平衡延迟和吞吐 |
| 大 batch (32+) | 不推荐 | 无 | 投机解码可能负优化 |
| 代码补全 | 5-10 | N-gram + EAGLE | N-gram 在代码上很强 |
| JSON 生成 | 5-7 | N-gram | 大量重复键名 |
| 通用对话 | 3-5 | EAGLE-2 / MTP | 依赖模型可用性 |

---

> **下一节**：[06-tree-attention.md](06-tree-attention.md) — Tree Attention 详解
