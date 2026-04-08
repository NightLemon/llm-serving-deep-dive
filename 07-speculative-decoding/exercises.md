# 动手练习：对比不同投机解码方案的加速比

> 本节通过 5 个实践练习，帮助你深入理解投机解码的工作机制、性能特征和参数调优。从数学模拟到实际 vLLM 部署，逐步建立对各方案的直觉。

---

## 练习 1：Rejection Sampling 模拟器

**目标**：通过纯 Python 模拟理解投机解码的数学基础，验证期望加速比公式。

### 任务

实现一个完整的 speculative decoding 模拟器，不涉及任何实际模型，只模拟概率分布的交互。

```python
"""
练习 1: Speculative Decoding 模拟器

实现步骤:
1. 用 Dirichlet 分布随机生成 target 和 draft 的 token 分布
2. 控制两个分布的 "相似度" (对应接受率 α)
3. 实现 rejection sampling 验证
4. 统计平均接受 token 数，与理论公式对比
"""

import numpy as np
from typing import Tuple, List

# ============================================
# TODO 1: 实现分布生成函数
# ============================================

def generate_distributions(
    vocab_size: int = 100,
    similarity: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一对 target 分布 p 和 draft 分布 q

    Args:
        vocab_size: 词表大小
        similarity: p 和 q 的相似度 (0=完全不同, 1=完全相同)

    Returns:
        p: target 分布 [vocab_size]
        q: draft 分布 [vocab_size]

    提示:
    - 可以先用 Dirichlet 生成 p
    - 然后 q = similarity * p + (1 - similarity) * noise
    - 确保 p, q 都是合法的概率分布 (非负, 和为 1)
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 2: 实现 rejection sampling
# ============================================

def rejection_sample_one_token(
    x: int,
    p: np.ndarray,
    q: np.ndarray
) -> Tuple[bool, int]:
    """
    对单个 token 执行 rejection sampling

    Args:
        x: draft model 采样的 token
        p: target 分布
        q: draft 分布

    Returns:
        accepted: 是否接受
        output_token: 输出 token (接受时=x, 拒绝时从修正分布采样)
    """
    # TODO: 你的实现
    # 提示: accept prob = min(1, p(x)/q(x))
    # 拒绝时从修正分布 max(0, p - q) / sum(max(0, p-q)) 采样
    pass


# ============================================
# TODO 3: 实现完整的单轮 speculative decoding
# ============================================

def simulate_one_round(
    gamma: int,
    vocab_size: int,
    similarity: float
) -> int:
    """
    模拟一轮 speculative decoding

    Args:
        gamma: draft 长度
        vocab_size: 词表大小
        similarity: 分布相似度

    Returns:
        num_accepted: 本轮接受的 token 数 (包括修正采样的 token)
    """
    # TODO: 你的实现
    # 提示:
    # 1. 为每个位置生成 p, q 分布
    # 2. 从 q 采样 draft tokens
    # 3. 逐个做 rejection sampling
    # 4. 第一个拒绝位置从修正分布采样 (计入 accepted)
    # 5. 全部接受时有 bonus token
    pass


# ============================================
# TODO 4: 大规模模拟并对比理论值
# ============================================

def run_experiment(
    gammas: List[int] = [1, 3, 5, 7, 10],
    similarities: List[float] = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
    num_trials: int = 10000,
    vocab_size: int = 100,
):
    """
    运行大规模模拟实验

    对每个 (gamma, similarity) 组合:
    1. 计算理论期望值: E = (1 - α^(γ+1)) / (1 - α)
    2. 运行 num_trials 次模拟, 统计实际平均值
    3. 计算理论值与实际值的偏差

    输出一个对比表格

    问题思考:
    - α 与 similarity 的关系是什么? (提示: α = Σ min(p, q))
    - 理论公式假设各位置 α 相同且独立, 实际中这个假设合理吗?
    - 当 γ 很大时, 边际收益如何变化?
    """
    # TODO: 你的实现
    pass


if __name__ == "__main__":
    run_experiment()
```

### 预期输出

```
=== Speculative Decoding 模拟结果 ===

similarity=0.50, α≈0.72:
  γ=1: 理论=1.48, 实际=1.49 (误差: 0.7%)
  γ=3: 理论=2.53, 实际=2.51 (误差: 0.8%)
  γ=5: 理论=3.10, 实际=3.08 (误差: 0.6%)
  γ=7: 理论=3.38, 实际=3.36 (误差: 0.6%)

similarity=0.90, α≈0.95:
  γ=1: 理论=1.95, 实际=1.95 (误差: 0.0%)
  γ=3: 理论=3.72, 实际=3.71 (误差: 0.3%)
  γ=5: 理论=5.33, 实际=5.30 (误差: 0.6%)
  γ=7: 理论=6.77, 实际=6.74 (误差: 0.4%)
```

### 思考题

1. 当 `similarity=1.0` 时，$\alpha$ 是否等于 1.0？为什么？
2. 增大 `vocab_size` 对 $\alpha$ 有什么影响？
3. 如果 draft 分布是均匀分布（$q(x) = 1/V$），$\alpha$ 会是多少？

---

## 练习 2：Tree Attention Mask 构建与可视化

**目标**：手动构建 tree attention mask，理解 tree 结构如何影响验证效率。

### 任务

```python
"""
练习 2: Tree Attention Mask 构建与可视化

实现 tree attention mask 的构建, 并用 matplotlib 可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict

# ============================================
# TODO 1: 定义 Tree 数据结构
# ============================================

class TreeNode:
    def __init__(self, token_id: int, node_id: int, parent: Optional['TreeNode'] = None):
        self.token_id = token_id
        self.node_id = node_id
        self.parent = parent
        self.children: List['TreeNode'] = []
        self.depth = 0 if parent is None else parent.depth + 1

    def add_child(self, token_id: int, node_id: int) -> 'TreeNode':
        child = TreeNode(token_id, node_id, self)
        self.children.append(child)
        return child


# ============================================
# TODO 2: 构建 Tree Attention Mask
# ============================================

def build_tree_attention_mask(
    nodes: List[TreeNode],
    num_prefix_tokens: int
) -> np.ndarray:
    """
    构建 tree attention mask

    Args:
        nodes: 树中所有节点的列表 (按 BFS 顺序)
        num_prefix_tokens: prefix token 数量

    Returns:
        mask: [num_nodes, num_prefix + num_nodes] 的 bool 数组
            mask[i][j] = True 表示 node i 可以 attend to position j

    规则:
    - 所有 node 都可以 attend to 所有 prefix positions
    - 每个 node 可以 attend to 自己
    - 每个 node 可以 attend to 其所有祖先 (parent, grandparent, ...)
    - 不能 attend to 其他分支的 node
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 3: 计算 Position IDs
# ============================================

def compute_position_ids(
    nodes: List[TreeNode],
    prefix_length: int
) -> List[int]:
    """
    计算每个 tree node 的 position ID

    同一深度的 nodes 应该有相同的 position ID
    position_id = prefix_length + depth

    Returns:
        position_ids: 每个 node 的 position ID
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 4: 可视化
# ============================================

def visualize_tree_and_mask(nodes, mask, num_prefix, position_ids):
    """
    创建两个子图:
    1. 树结构的可视化 (用 networkx 或手动画)
    2. Attention mask 的热力图

    提示: 用 matplotlib 的 imshow 画 mask
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 子图 1: 树结构
    # TODO: 画出树结构, 标注 token_id 和 depth

    # 子图 2: Attention mask 热力图
    # TODO: 用 imshow 画 mask, x 轴是 attend-to 的位置, y 轴是 node

    plt.tight_layout()
    plt.savefig("tree_attention_mask.png", dpi=150)
    plt.show()


# ============================================
# TODO 5: 构建测试用例
# ============================================

def create_test_tree():
    """
    构建以下测试树:

          root(0)
         /    \
       A(1)   B(2)
      / \       \
    C(3) D(4)   E(5)
    |
    F(6)

    返回节点列表 (BFS 顺序)
    """
    # TODO: 你的实现
    pass


if __name__ == "__main__":
    nodes = create_test_tree()
    num_prefix = 5  # 假设有 5 个 prefix tokens

    mask = build_tree_attention_mask(nodes, num_prefix)
    position_ids = compute_position_ids(nodes, num_prefix)

    print("Tree Attention Mask:")
    print(mask.astype(int))
    print(f"\nPosition IDs: {position_ids}")

    visualize_tree_and_mask(nodes, mask, num_prefix, position_ids)
```

### 验证清单

- [ ] `root` 行: 只能看到 prefix 和自己
- [ ] `A` 行: 能看到 prefix、root、自己，不能看到 B/E
- [ ] `C` 行: 能看到 prefix、root、A、自己，不能看到 B/D/E
- [ ] `F` 行: 能看到 prefix、root、A、C、自己
- [ ] 同深度节点 (A/B, C/D/E) 有相同的 position ID

---

## 练习 3：vLLM 投机解码 Benchmark

**目标**：在实际环境中对比不同投机解码方案的延迟和吞吐。

### 前置条件

- 一台有 A100 40G/80G（或同级 GPU）的机器
- 安装 vLLM >= 0.8.0
- 至少 80GB 可用显存（用于加载模型和 draft model）

### 任务

```python
"""
练习 3: vLLM 投机解码 Benchmark

对比以下方案:
1. Baseline (无投机解码)
2. N-gram speculation
3. EAGLE (如果有对应的 head)
4. Draft model (同系列小模型)

在不同类型的 prompt 上测试 (代码, 对话, JSON)
"""

import time
import json
from vllm import LLM, SamplingParams

# ============================================
# TODO 1: 准备测试 prompts
# ============================================

TEST_PROMPTS = {
    "code": [
        "Write a Python function that implements binary search on a sorted array. Include proper error handling and type hints.",
        "Implement a LRU cache in Python using OrderedDict. The cache should support get and put operations.",
        "Write a Python class that implements a thread-safe singleton pattern with double-checked locking.",
    ],
    "chat": [
        "Explain the difference between TCP and UDP protocols in networking. When would you use one over the other?",
        "What are the main differences between Python and Rust? Discuss memory management, performance, and use cases.",
        "Describe how garbage collection works in Java, including the different generations and GC algorithms.",
    ],
    "json": [
        'Generate a JSON object representing a user profile with fields: name, email, age, address (nested), hobbies (array), and employment history (array of objects with company, role, years).',
        'Create a JSON API response for a paginated list of products with fields: id, name, price, category, ratings (nested object with average, count), and availability status.',
        'Generate a JSON schema definition for a blog post that includes title, content, author info, tags, comments array, and metadata.',
    ],
}

# ============================================
# TODO 2: 实现 benchmark 函数
# ============================================

def benchmark_config(
    model_name: str,
    spec_config: dict,
    prompts: list,
    sampling_params: SamplingParams,
    warmup: int = 2,
    config_name: str = "unknown",
) -> dict:
    """
    对单个配置进行 benchmark

    Returns:
        {
            "config": config_name,
            "avg_latency_ms": float,
            "avg_tokens_per_sec": float,
            "avg_output_length": float,
            "total_time_s": float,
        }
    """
    # TODO: 你的实现
    # 提示:
    # 1. 创建 LLM 实例 (注意不同配置的参数)
    # 2. Warmup
    # 3. 计时
    # 4. 统计延迟和吞吐
    pass


# ============================================
# TODO 3: 运行完整的对比实验
# ============================================

def run_full_benchmark():
    """
    运行完整的 benchmark 实验

    配置:
    1. baseline: 无投机解码
    2. ngram: --speculative-model "[ngram]" --num-speculative-tokens 5
    3. draft_model: --speculative-model 同系列小模型 --num-speculative-tokens 5
    4. eagle (如果可用): --speculative-model EAGLE_HEAD --num-speculative-tokens 5

    对每个 prompt 类别 (code/chat/json) 分别测试
    """
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    # 如果没有 70B 的显存，用 8B 作为 target model, 但可能加速不明显

    configs = {
        "baseline": {},
        "ngram": {
            "speculative_model": "[ngram]",
            "num_speculative_tokens": 5,
            "ngram_prompt_lookup_max": 4,
        },
        # 取消注释以测试 draft model (需要足够显存)
        # "draft_model": {
        #     "speculative_model": "meta-llama/Llama-3.1-1B-Instruct",
        #     "num_speculative_tokens": 5,
        # },
    }

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0,  # greedy mode, 方便验证一致性
    )

    results = {}
    for prompt_type, prompts in TEST_PROMPTS.items():
        results[prompt_type] = {}
        for config_name, config in configs.items():
            print(f"\n=== {prompt_type} / {config_name} ===")
            result = benchmark_config(
                MODEL, config, prompts, sampling_params,
                config_name=config_name
            )
            results[prompt_type][config_name] = result

    # TODO: 打印对比表格
    print_results_table(results)
    return results


def print_results_table(results):
    """
    打印格式化的结果表格

    格式:
    | Prompt Type | Config | Avg Latency (ms) | Tokens/sec | Speedup |
    |------------|--------|-------------------|------------|---------|
    """
    # TODO: 你的实现
    pass


if __name__ == "__main__":
    run_full_benchmark()
```

### 预期发现

1. **N-gram** 在 JSON 和代码生成上的加速比高于对话
2. **Draft model** 的加速比较为稳定，但有额外的显存开销
3. **Greedy 模式** 的加速比通常高于 sampling 模式
4. **短输出** 的加速比低于长输出（投机解码的 amortization 效果）

---

## 练习 4：修正分布采样的数值验证

**目标**：用大量采样验证 rejection sampling 输出的分布确实等于 target 分布。

### 任务

```python
"""
练习 4: 数值验证 Rejection Sampling 的无损性

核心目标: 用统计方法验证 rejection sampling 后的输出分布 = target 分布
"""

import numpy as np
from scipy import stats
from collections import Counter

# ============================================
# TODO 1: 实现完整的 rejection sampling (含修正分布)
# ============================================

def speculative_sample(p: np.ndarray, q: np.ndarray) -> int:
    """
    执行一次投机采样 (单步)

    流程:
    1. 从 q 采样 token x
    2. 以 min(1, p(x)/q(x)) 概率接受
    3. 接受 → 返回 x
    4. 拒绝 → 从修正分布 max(0, p-q)/Z 采样新 token，返回

    Returns:
        输出 token (应该服从分布 p)
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 2: 大量采样并统计
# ============================================

def verify_distribution(
    p: np.ndarray,
    q: np.ndarray,
    num_samples: int = 100000,
) -> dict:
    """
    通过大量采样验证输出分布 = p

    Returns:
        {
            "empirical_dist": np.ndarray,  # 实际采样得到的分布
            "target_dist": p,               # 理论目标分布
            "kl_divergence": float,         # KL(empirical || target)
            "chi2_pvalue": float,           # 卡方检验 p-value
            "max_abs_error": float,         # 最大绝对误差
        }
    """
    # TODO: 你的实现
    # 提示:
    # 1. 调用 speculative_sample num_samples 次
    # 2. 统计每个 token 出现的频率
    # 3. 与 p 对比
    # 4. 做卡方检验 (chi-squared test)
    pass


# ============================================
# TODO 3: 对比不同场景
# ============================================

def experiment():
    """
    测试以下场景:

    场景 A: p 和 q 非常相似 (α ≈ 0.95)
    场景 B: p 和 q 中等相似 (α ≈ 0.7)
    场景 C: p 和 q 差异很大 (α ≈ 0.3)
    场景 D: q 是均匀分布 (最坏情况 draft)
    场景 E: p 是 one-hot (greedy 模式)

    对每个场景:
    1. 验证输出分布 = p (p-value > 0.05)
    2. 计算接受率
    3. 计算平均采样效率

    思考:
    - 场景 E 中修正分布是什么? (提示: 应该是 undefined/不需要)
    - 如果 q(x)=0 但 p(x)>0, 会发生什么?
    """
    # TODO: 你的实现
    pass

if __name__ == "__main__":
    experiment()
```

### 关键验证点

1. 对所有场景，卡方检验的 p-value 应该 > 0.05（不能拒绝"分布相同"的原假设）
2. KL 散度应该随 `num_samples` 增大而趋近 0
3. 场景 E（one-hot target）中，rejection sampling 退化为确定性比较

---

## 练习 5：EAGLE Head 训练模拟

**目标**：用一个小模型模拟 EAGLE head 的训练过程，理解 feature-level draft 的原理。

### 任务

```python
"""
练习 5: EAGLE Head 训练模拟

使用 GPT-2 (124M) 作为 "target model"，训练一个 EAGLE-style head
在小规模数据上验证 EAGLE 的核心思路

注意: 这是教学目的的简化实现, 不用于生产
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================
# TODO 1: 定义 EAGLE Head
# ============================================

class SimpleEAGLEHead(nn.Module):
    """
    简化版 EAGLE Head

    输入: token embedding (d) + hidden state (d) = 2d
    输出: 预测的下一步 hidden state (d)

    结构: Linear(2d → d) → LayerNorm → Linear(d → d) → GELU → Linear(d → d)
    """
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: 定义网络结构
        pass

    def forward(self, token_embed, hidden_state):
        """
        Args:
            token_embed: [batch, hidden_size]
            hidden_state: [batch, hidden_size]
        Returns:
            predicted_hidden: [batch, hidden_size]
        """
        # TODO: 你的实现
        pass


# ============================================
# TODO 2: 采集训练数据
# ============================================

def collect_training_data(model, tokenizer, texts, device='cuda'):
    """
    从 target model 收集训练数据

    对每个 text:
    1. Tokenize
    2. 前向传播, 收集所有层的 hidden states
    3. 收集 token embeddings

    返回:
    - embeddings: [total_tokens, hidden_size]
    - hidden_states: [total_tokens, hidden_size] (最后一层)
    - next_token_ids: [total_tokens] (下一个 token 的 ID)
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 3: 训练循环
# ============================================

def train_eagle_head(
    target_model,
    eagle_head,
    train_data,
    epochs=10,
    lr=1e-3,
    device='cuda',
):
    """
    训练 EAGLE head

    Loss = cross_entropy(lm_head(predicted_hidden), next_token)

    注意: lm_head 是 target model 的 LM Head, 冻结不训练

    步骤:
    1. 输入 (embed[t], hidden[t]) → EAGLE head → predicted_hidden[t+1]
    2. lm_head(predicted_hidden[t+1]) → logits
    3. loss = CE(logits, next_token[t+1])
    4. 只更新 EAGLE head 的参数
    """
    # TODO: 你的实现
    pass


# ============================================
# TODO 4: 评估 draft 质量
# ============================================

def evaluate_draft_quality(
    target_model,
    eagle_head,
    eval_texts,
    tokenizer,
    num_draft_steps=5,
    device='cuda',
):
    """
    评估 EAGLE head 的 draft 质量

    指标:
    1. 每步的 top-1 准确率
    2. 每步的 top-5 准确率
    3. 平均接受率 α
    4. 模拟的期望加速比

    思考题:
    - 随着 draft 步数增加, 准确率如何变化?
    - 这和 Medusa 的准确率下降模式有什么区别?
    """
    # TODO: 你的实现
    pass


# ============================================
# 主程序
# ============================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载 GPT-2 作为 "target model"
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 冻结 target model
    for param in model.parameters():
        param.requires_grad = False

    # 创建 EAGLE head
    eagle_head = SimpleEAGLEHead(model.config.n_embd).to(device)

    # 训练数据 (用简单文本)
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        # 添加更多训练文本...
    ]

    # 收集训练数据
    train_data = collect_training_data(model, tokenizer, train_texts, device)

    # 训练
    train_eagle_head(model, eagle_head, train_data, epochs=20, device=device)

    # 评估
    eval_texts = [
        "The weather today is sunny and warm.",
        "Deep learning models require large amounts of data.",
    ]
    evaluate_draft_quality(model, eagle_head, eval_texts, tokenizer, device=device)


if __name__ == "__main__":
    main()
```

### 预期观察

1. EAGLE head 在训练数据上的 top-1 准确率应该能达到 50-70%（第 1 步），之后逐步下降
2. 与 Medusa 不同，EAGLE 在后续步骤上的准确率下降应该更平缓（因为是自回归预测）
3. 更多的训练数据和更大的 EAGLE head 会提升 draft 质量

---

## 提交建议

完成练习后，记录以下内容：

1. **练习 1**：理论值与实测值的对比表格，以及对偏差原因的分析
2. **练习 2**：tree attention mask 的可视化截图，以及对不同 tree 形状效率的讨论
3. **练习 3**：benchmark 结果表格，包括不同 prompt 类型和配置的加速比
4. **练习 4**：各场景的卡方检验 p-value 和 KL 散度，验证无损性
5. **练习 5**：EAGLE head 训练曲线和各步骤的准确率图表

### 进阶挑战

- 在练习 1 中加入 **tree 结构** 的模拟，对比线性 draft 和 tree draft 的期望加速比
- 在练习 3 中加入 **不同 batch size** 的对比，观察投机解码在大 batch 下的收益变化
- 在练习 5 中实现完整的 **speculative decoding loop**（EAGLE head draft + GPT-2 验证），测量端到端加速比
