# 动手练习

> 通过实际计算、代码实验和分析，加深对 KV Cache 压缩技术的理解。

---

## 练习 1：KV Cache 显存计算器

### 目标

编写一个 KV Cache 显存计算工具，支持不同的 attention 方案和量化精度，帮助理解各种压缩技术的显存节省效果。

### 任务

实现以下 Python 函数：

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    hidden_dim: int
    num_q_heads: int
    num_kv_heads: int       # GQA: < num_q_heads; MQA: 1; MHA: = num_q_heads
    head_dim: int
    num_layers: int
    attention_type: Literal["mha", "gqa", "mqa", "mla"]
    mla_latent_dim: int = 0      # MLA 专用：latent dimension d_c
    mla_rope_dim: int = 0        # MLA 专用：RoPE key dimension d_h^R

@dataclass
class ServingConfig:
    """推理服务配置"""
    batch_size: int
    seq_len: int
    kv_dtype: Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4", "int2"]

def dtype_bytes(dtype: str) -> float:
    """返回每个元素占用的字节数"""
    # TODO: 实现
    pass

def kv_cache_per_token_per_layer(model: ModelConfig, dtype: str) -> float:
    """
    计算每个 token 每层的 KV Cache 大小（bytes）。
    
    对于 MHA/GQA/MQA:
      KV Cache = 2 × num_kv_heads × head_dim × dtype_bytes
    
    对于 MLA:
      KV Cache = (mla_latent_dim + mla_rope_dim) × dtype_bytes
    """
    # TODO: 实现
    pass

def total_kv_cache(model: ModelConfig, serving: ServingConfig) -> dict:
    """
    计算总 KV Cache 显存占用，返回详细信息。
    
    返回:
    {
        "per_token_per_layer_bytes": float,
        "per_token_all_layers_bytes": float,
        "total_bytes": float,
        "total_gb": float,
        "compression_vs_mha_fp16": float,  # 相对于同规模 MHA FP16 的压缩比
    }
    """
    # TODO: 实现
    pass
```

### 验证数据

用你的实现计算以下模型的 KV Cache 大小，并与参考值对比：

```python
# 定义模型配置
models = {
    "LLaMA-3-8B": ModelConfig(
        name="LLaMA-3-8B",
        hidden_dim=4096, num_q_heads=32, num_kv_heads=8,
        head_dim=128, num_layers=32, attention_type="gqa"
    ),
    "LLaMA-3-70B": ModelConfig(
        name="LLaMA-3-70B",
        hidden_dim=8192, num_q_heads=64, num_kv_heads=8,
        head_dim=128, num_layers=80, attention_type="gqa"
    ),
    "DeepSeek-V3": ModelConfig(
        name="DeepSeek-V3",
        hidden_dim=7168, num_q_heads=128, num_kv_heads=0,
        head_dim=128, num_layers=61, attention_type="mla",
        mla_latent_dim=512, mla_rope_dim=64
    ),
}

serving = ServingConfig(batch_size=64, seq_len=4096, kv_dtype="fp16")

# 期望输出（近似值）：
# LLaMA-3-8B:   ~32 GB total KV Cache
# LLaMA-3-70B:  ~80 GB total KV Cache  
# DeepSeek-V3:  ~17.2 GB total KV Cache
```

### 进阶要求

1. 增加一个函数 `max_batch_size(model, serving, gpu_memory_gb, model_weight_gb)`，计算给定 GPU 显存和模型权重后，能支持的最大 batch_size
2. 生成一个对比表格，展示同一模型在不同量化精度下的 KV Cache 大小和最大 batch_size
3. 对比 GQA-8 + FP8 与 MLA + FP16 的 KV Cache 大小，哪个更小？

---

## 练习 2：FP8 KV Cache 量化误差分析

### 目标

理解 FP8 量化对 attention 计算的数值影响，分析误差分布和累积效应。

### 任务

```python
import torch
import torch.nn.functional as F

def simulate_fp8_quantization(tensor: torch.Tensor, format: str = "e4m3"):
    """
    模拟 FP8 量化（在没有 FP8 硬件的情况下）。
    
    E4M3: 1 sign + 4 exponent + 3 mantissa, max = 240
    E5M2: 1 sign + 5 exponent + 2 mantissa, max = 57344
    
    步骤：
    1. 计算 scale = fp8_max / tensor.abs().max()
    2. 缩放 tensor
    3. 模拟精度损失（round to nearest representable FP8 value）
    4. 反缩放
    """
    # TODO: 实现
    # 提示：可以使用 torch.float8_e4m3fn 类型（PyTorch 2.1+）
    # 或者手动模拟量化过程
    pass

def analyze_quantization_error(
    num_heads: int = 32,
    head_dim: int = 128,
    seq_len: int = 1024,
    num_trials: int = 10,
):
    """
    分析 FP8 量化对 attention 输出的影响。
    
    步骤：
    1. 生成随机的 Q, K, V（模拟真实分布：标准正态 + 少量 outlier）
    2. 计算 FP16 精度的 attention 输出（baseline）
    3. 将 K, V 量化为 FP8，重新计算 attention 输出
    4. 比较两者的差异（MSE、max error、cosine similarity）
    5. 分析误差在不同 head 之间的分布
    """
    results = []
    
    for trial in range(num_trials):
        # 生成带 outlier 的 K, V
        K = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16)
        V = torch.randn(1, num_heads, seq_len, head_dim, dtype=torch.float16)
        Q = torch.randn(1, num_heads, 1, head_dim, dtype=torch.float16)
        
        # 添加 outlier channel (模拟真实 Key 分布)
        outlier_channels = torch.randint(0, head_dim, (3,))
        K[:, :, :, outlier_channels] *= 10.0
        
        # TODO: 
        # 1. 计算 FP16 baseline attention output
        # 2. 量化 K, V 为 FP8
        # 3. 计算 FP8 attention output
        # 4. 计算误差指标
        pass
    
    # TODO: 汇总并可视化结果
    pass
```

### 思考题

1. 对比 per-tensor 和 per-channel 量化 Key 的误差，哪个更小？为什么？
2. 量化 Key 和量化 Value 哪个对 attention 输出的影响更大？请设计实验验证。
3. 当 seq_len 从 1024 增加到 16384 时，量化误差如何变化？为什么？

---

## 练习 3：StreamingLLM Attention Sink 验证

### 目标

通过实验验证 attention sink 效应的存在，并实现一个简单的 StreamingLLM cache 管理器。

### 任务 A：Attention Sink 可视化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def visualize_attention_sink(
    model_name: str = "meta-llama/Llama-3.2-1B",  # 或其他可用的小模型
    text: str = "The quick brown fox jumps over the lazy dog. " * 50,
    layer_indices: list = [0, 8, 16, 24, 31],
):
    """
    可视化指定层的 attention 分布，验证 attention sink 效应。
    
    步骤：
    1. 加载模型和 tokenizer
    2. 对输入文本做 forward pass，记录所有层的 attention weights
    3. 对指定层绘制 attention heatmap
    4. 统计第 0 个 token（BOS）在每一层获得的平均 attention weight
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        output_attentions=True,  # 输出 attention weights
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions  # tuple of [batch, heads, seq, seq]
    
    # TODO:
    # 1. 对每个指定层，计算 BOS token 获得的平均 attention weight
    # 2. 绘制 "BOS attention weight vs layer index" 的曲线
    # 3. 对比 BOS token 和随机中间 token 的 attention weight
    # 4. 用 matplotlib 绘制 2-3 个层的 attention heatmap
    pass
```

### 任务 B：StreamingLLM Cache 实现

```python
class StreamingLLMCache:
    """
    实现 StreamingLLM 的 KV Cache 管理策略。
    
    策略：保留前 sink_size 个 token + 最近 window_size 个 token 的 KV Cache。
    """
    
    def __init__(self, sink_size: int = 4, window_size: int = 252):
        """
        总 budget = sink_size + window_size = 256 tokens
        """
        self.sink_size = sink_size
        self.window_size = window_size
        self.total_budget = sink_size + window_size
        
        self.key_cache = None   # [num_layers, num_heads, budget, head_dim]
        self.value_cache = None
        self.current_len = 0
        self.total_seen = 0     # 已处理的 token 总数
    
    def update(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """
        添加一个新 token 的 KV 到 cache。
        
        new_key: [num_layers, num_heads, 1, head_dim]
        new_value: [num_layers, num_heads, 1, head_dim]
        
        当 cache 满时，使用 FIFO 策略驱逐 window 中最旧的 token。
        Sink tokens 永远不被驱逐。
        """
        # TODO: 实现
        # 提示：
        # 1. 前 sink_size 步：直接追加到 cache
        # 2. 之后：追加到 window 区域，FIFO 驱逐最旧的 window token
        # 3. 注意 position encoding 的处理
        pass
    
    def get_kv_cache(self):
        """
        返回当前的 KV Cache 用于 attention 计算。
        
        返回:
            key_cache: [num_layers, num_heads, current_len, head_dim]
            value_cache: [num_layers, num_heads, current_len, head_dim]
            position_ids: [current_len]  # 用于 RoPE 的位置编码
        """
        # TODO: 实现
        # 注意 position_ids 的处理：
        #   sink tokens 保留原始位置 [0, 1, ..., sink_size-1]
        #   window tokens 位置紧接在 sink 之后
        pass
    
    def memory_usage_bytes(self, dtype_bytes: int = 2) -> int:
        """计算当前 cache 的显存占用"""
        # TODO: 实现
        pass


def test_streaming_cache():
    """测试 StreamingLLM cache 的正确性"""
    cache = StreamingLLMCache(sink_size=4, window_size=12)
    
    num_layers, num_heads, head_dim = 2, 4, 8
    
    # 模拟 20 个 token 的 decoding
    for step in range(20):
        key = torch.randn(num_layers, num_heads, 1, head_dim)
        value = torch.randn(num_layers, num_heads, 1, head_dim)
        cache.update(key, value)
        
        kv_key, kv_value, positions = cache.get_kv_cache()
        
        print(f"Step {step:2d}: cache_len={kv_key.shape[2]:2d}, "
              f"positions={positions.tolist()}")
    
    # 预期输出：
    # Step  0: cache_len= 1, positions=[0]
    # Step  3: cache_len= 4, positions=[0, 1, 2, 3]
    # Step 15: cache_len=16, positions=[0, 1, 2, 3, 4, 5, ..., 15]
    # Step 16: cache_len=16, positions=[0, 1, 2, 3, 5, 6, ..., 16]
    #          (第 4 个 token 被驱逐，window 滑动)
    # Step 19: cache_len=16, positions=[0, 1, 2, 3, 8, 9, ..., 19]

test_streaming_cache()
```

### 思考题

1. 如果将 `sink_size` 从 4 减少到 1 或增加到 16，会如何影响生成质量？
2. StreamingLLM 的位置重编码策略（将 window tokens 的位置紧凑排列）可能会对哪些任务产生负面影响？
3. 设计一个实验来测量 StreamingLLM 在"需要回忆早期信息"任务上的退化程度。

---

## 练习 4：MLA vs GQA 压缩效率对比分析

### 目标

通过数学分析和代码模拟，深入比较 MLA 和 GQA 在不同配置下的 KV Cache 效率。

### 任务 A：理论分析

填写以下表格，计算各种配置下的 KV Cache 大小（FP16，per token per layer，单位 bytes）：

```
| 配置 | 参数 | KV Cache (bytes/token/layer) | 相对 MHA 压缩比 |
|------|------|------------------------------|-----------------|
| MHA-32 | n_h=32, d_h=128 | ? | 1x |
| MHA-64 | n_h=64, d_h=128 | ? | — |
| GQA-8 (32 heads) | n_h=32, n_kv=8, d_h=128 | ? | ? |
| GQA-8 (64 heads) | n_h=64, n_kv=8, d_h=128 | ? | ? |
| GQA-4 (32 heads) | n_h=32, n_kv=4, d_h=128 | ? | ? |
| MQA (32 heads) | n_h=32, n_kv=1, d_h=128 | ? | ? |
| MLA (d_c=512) | d_c=512, d_rope=64 | ? | ?* |
| MLA (d_c=256) | d_c=256, d_rope=64 | ? | ?* |
| MLA (d_c=1024) | d_c=1024, d_rope=64 | ? | ?* |
```

*注：MLA 的压缩比相对于同 hidden_dim 模型的 MHA 来计算。

### 任务 B：找到 GQA 和 MLA 的等效点

```python
def find_equivalent_config():
    """
    找到 GQA 和 MLA 的等效配置：
    给定 MLA (d_c=512, d_rope=64)，
    找到哪个 GQA 分组数能达到相同的 KV Cache 大小。
    
    问题：
    1. 对于 n_h=128, d_h=128 的模型，MLA (d_c=512, d_rope=64) 等效于 GQA-?
    2. 对于 n_h=32, d_h=128 的模型，MLA (d_c=512, d_rope=64) 等效于 GQA-?
    3. 是否存在某个 GQA 分组数能完全匹配 MLA 的 KV Cache 大小？
    """
    # MLA KV Cache per token per layer
    d_c = 512
    d_rope = 64
    mla_cache = (d_c + d_rope) * 2  # bytes (FP16)
    
    print(f"MLA cache: {mla_cache} bytes/token/layer")
    
    for n_h in [32, 64, 128]:
        d_h = 128
        print(f"\n模型 n_h={n_h}, d_h={d_h}:")
        print(f"  MHA cache: {2 * n_h * d_h * 2} bytes")
        
        # 找到使 GQA cache ≈ MLA cache 的 n_kv
        # GQA cache = 2 × n_kv × d_h × 2
        # 令 2 × n_kv × d_h × 2 = mla_cache
        n_kv_equivalent = mla_cache / (2 * d_h * 2)
        print(f"  等效 GQA n_kv = {n_kv_equivalent}")
        
        # 输出附近的整数 GQA 配置
        for n_kv in [1, 2, 4, 8, 16]:
            gqa_cache = 2 * n_kv * d_h * 2
            ratio = gqa_cache / mla_cache
            print(f"  GQA-{n_h//n_kv} (n_kv={n_kv}): {gqa_cache} bytes, "
                  f"ratio to MLA = {ratio:.2f}x")

find_equivalent_config()
```

### 任务 C：权重吸收效率分析

```python
import torch
import time

def benchmark_mla_absorption(
    batch_size: int = 1,
    seq_len: int = 4096,
    n_h: int = 128,
    d_h: int = 128,
    d_c: int = 512,
    d_rope: int = 64,
):
    """
    对比两种 MLA attention 实现的效率：
    1. 朴素实现：先 up-project 再做标准 attention
    2. 吸收实现：在 latent space 直接计算
    
    测量：
    - 内存读取量
    - 计算时间
    - 数值差异（应该为零或极小的浮点误差）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟 weights
    W_UK = torch.randn(d_c, n_h * d_h, device=device, dtype=torch.float16) * 0.01
    W_UV = torch.randn(d_c, n_h * d_h, device=device, dtype=torch.float16) * 0.01
    
    # 模拟 cached latent vectors
    c_kv = torch.randn(batch_size, seq_len, d_c, device=device, dtype=torch.float16) * 0.1
    
    # 模拟 query latent
    c_q = torch.randn(batch_size, 1, n_h * d_h, device=device, dtype=torch.float16) * 0.1
    
    # TODO: 实现方法 1（朴素）和方法 2（吸收），对比性能和结果
    
    # 方法 1: 朴素实现
    # K_full = c_kv @ W_UK  → [B, S, n_h*d_h]
    # V_full = c_kv @ W_UV  → [B, S, n_h*d_h]
    # score = c_q @ K_full.transpose(-1, -2) / sqrt(d_h)  → [B, 1, S]
    # output = softmax(score) @ V_full  → [B, 1, n_h*d_h]
    
    # 方法 2: 吸收实现
    # W_abs = ... (预计算)
    # score = c_q @ W_abs @ c_kv.T / sqrt(d_h)
    # latent_output = softmax(score) @ c_kv
    # output = latent_output @ W_UV
    
    pass

benchmark_mla_absorption()
```

### 思考题

1. MLA 的 latent dimension d_c 越小，KV Cache 越小。但 d_c 太小会怎样？如何确定最优的 d_c？
2. 如果在 GQA-8 的基础上再做 FP8 量化，KV Cache 大小会和 MLA + FP16 相比如何？
3. MLA 的吸收优化在 prefill 阶段和 decode 阶段分别有多大收益？为什么？

---

## 练习 5：综合方案设计

### 场景描述

你需要为一个在线客服系统设计 LLM 推理架构。需求如下：

- 模型：LLaMA-3-70B（GQA-8, n_h=64, n_kv=8, d_h=128, 80 layers）
- 硬件：4x NVIDIA H100 80GB（Tensor Parallelism）
- 模型权重：~140GB FP16（TP-4 后每张卡 ~35GB）
- 每张卡可用于 KV Cache 的显存：~40GB
- 目标：
  - 支持 200 并发用户
  - 平均上下文长度 4096 tokens
  - 最大上下文长度 32768 tokens
  - TTFT < 500ms
  - 解码速度 > 30 tokens/s

### 任务

**Step 1**: 计算基线场景（FP16 KV Cache）能否满足需求

```python
def calculate_baseline():
    """
    计算 FP16 KV Cache 下的显存需求：
    1. 200 个用户 × 平均 4096 tokens 的 KV Cache
    2. 能否放入 4 × 40GB = 160GB 的 KV Cache 空间？
    3. 如果不能，需要什么压缩方案？
    """
    kv_per_token = 2 * 8 * 128 * 80 * 2  # bytes, per token, all layers, FP16
    # TP-4: 每张卡承担 1/4 的 KV heads → 每张卡: n_kv_per_gpu = 2
    kv_per_token_per_gpu = 2 * 2 * 128 * 80 * 2  # bytes
    
    # TODO: 计算总 KV Cache 需求
    # TODO: 判断是否能放入显存
    # TODO: 如果不能，计算需要的压缩比
    pass
```

**Step 2**: 设计压缩方案

请从以下选项中组合出一个方案，满足显存约束和性能目标：

```
可选技术：
□ FP8 KV Cache 量化（2x 压缩）
□ INT8 KV Cache 量化（2x 压缩）
□ INT4 KV Cache 量化（4x 压缩）
□ SnapKV 选择性缓存（可选 2x-8x 压缩）
□ StreamingLLM（仅适合对话场景）
□ Prefix Caching（不压缩，但提高复用率）
```

**Step 3**: 分析方案的 trade-off

对你选择的方案，分析以下维度：

```markdown
| 维度 | 分析 |
|------|------|
| 显存节省 | 具体数字 |
| 精度影响 | 预估的质量损失 |
| 延迟影响 | 对 TTFT 和 decode 速度的影响 |
| 实现复杂度 | 需要修改哪些组件 |
| 风险 | 最大的技术风险是什么 |
```

**Step 4**: 编写 vLLM 启动配置

```bash
# 写出你推荐的 vLLM 启动命令
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    # TODO: 添加你选择的 KV Cache 压缩相关参数
    # --kv-cache-dtype ???
    # --max-model-len ???
    # --gpu-memory-utilization ???
    # 其他参数...
```

### 交付物

1. 一份简短的技术方案文档（不超过 1 页），说明你的选择和理由
2. 计算过程的 Python 代码
3. vLLM 启动命令

---

## 总结

通过这些练习，你应该能够：

1. **量化评估**：准确计算不同模型和量化精度下的 KV Cache 显存占用
2. **理解误差**：直观感受 FP8 量化对 attention 的影响程度
3. **实现验证**：亲手实现 StreamingLLM 的核心逻辑，理解 attention sink
4. **架构对比**：从数学角度比较 MLA 和 GQA 的压缩效率
5. **方案设计**：综合运用所学知识，为实际场景设计 KV Cache 压缩方案

---

> **返回**：[章节概述](README.md)
