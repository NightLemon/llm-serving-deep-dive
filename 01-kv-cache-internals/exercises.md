# 动手练习

> 通过实际计算和代码实验，巩固对 KV Cache 内存布局、Prefill/Decode 差异和显存计算的理解。

---

## 练习 1：KV Cache 显存精确计算

### 题目

你的团队计划部署 **Qwen-2.5-72B** 用于企业内部的 RAG 问答系统。硬件为 **4 张 NVIDIA H100-80GB**（TP=4）。

已知模型参数：
- 参数量：72B
- 层数 $L = 80$
- Q heads: 64, KV heads: 8
- Head dim: 128
- 使用 BF16 推理

系统需求：
- 典型 prompt 长度：8,000 tokens（含检索文档）
- 典型生成长度：2,000 tokens
- `max_model_len` 设为 16384
- `gpu_memory_utilization` = 0.9

**问题**：

**(a)** 计算每个 token 的 KV Cache 大小（每 GPU）。

**(b)** 计算单个请求在最大序列长度（16384 tokens）下的 KV Cache 大小（每 GPU）。

**(c)** 估算每 GPU 可用于 KV Cache 的显存空间。

**(d)** 计算最大可服务的并发请求数。注意：虽然 vLLM 使用分页按需分配，但你需要考虑**最坏情况**（所有请求都用到 `max_model_len`）和**典型情况**（请求平均使用 10,000 tokens）两种场景。

**(e)** 如果启用 FP8 KV Cache 量化，上述数字会如何变化？

### 参考答案框架

```python
# (a) 每 token KV Cache/GPU
L = 80
n_kv = 8
d_h = 128
TP = 4
dtype_bytes = 2  # BF16

kv_per_token_per_gpu = 2 * L * (n_kv / TP) * d_h * dtype_bytes
print(f"(a) KV per token per GPU: {kv_per_token_per_gpu:,} bytes "
      f"= {kv_per_token_per_gpu/1024:.1f} KB")

# (b) 单请求最大 KV Cache/GPU
max_seq_len = 16384
kv_per_request = kv_per_token_per_gpu * max_seq_len
print(f"(b) KV per request per GPU (max): {kv_per_request/1024**3:.3f} GB")

# (c) 可用 KV Cache 空间/GPU
total_gpu_mem = 80  # GB
gpu_util = 0.9
model_weights_per_gpu = 72 * 2 / 4  # 72B params × 2 bytes / 4 GPUs = 36 GB
activation_overhead = 2  # GB 估算

available_kv = total_gpu_mem * gpu_util - model_weights_per_gpu - activation_overhead
print(f"(c) Available KV Cache/GPU: {available_kv:.1f} GB")

# (d) 最大并发
max_concurrent_worst = available_kv / (kv_per_request / 1024**3)
avg_seq_len = 10000
kv_per_avg_request = kv_per_token_per_gpu * avg_seq_len
max_concurrent_typical = available_kv / (kv_per_avg_request / 1024**3)
print(f"(d) Max concurrency (worst case): {int(max_concurrent_worst)}")
print(f"(d) Max concurrency (typical): {int(max_concurrent_typical)}")

# (e) FP8
# 重复上述计算，将 dtype_bytes 改为 1
```

---

## 练习 2：Prefill vs Decode 性能分析

### 题目

使用 HuggingFace Transformers 对一个小型模型进行 Prefill 和 Decode 的性能对比实验。

**步骤**：

**(a)** 使用以下代码框架，测量不同 prompt 长度下 Prefill 和 Decode 的耗时：

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"  # 小模型，单 GPU 即可
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def measure_prefill_decode(prompt_len: int, gen_len: int = 50):
    """测量 Prefill 和 Decode 的耗时。"""
    
    # 构造指定长度的输入
    input_ids = torch.randint(
        100, 30000, (1, prompt_len), device=model.device
    )
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids[:, :10], max_new_tokens=5)
    torch.cuda.synchronize()
    
    # --- 测量 Prefill ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
    
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    prefill_time = t1 - t0
    
    # --- 测量 Decode (逐 token 生成) ---
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    decode_times = []
    
    for _ in range(gen_len):
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        decode_times.append(t3 - t2)
    
    return {
        "prompt_len": prompt_len,
        "prefill_time_ms": prefill_time * 1000,
        "prefill_tok_per_s": prompt_len / prefill_time,
        "avg_decode_time_ms": sum(decode_times) / len(decode_times) * 1000,
        "decode_tok_per_s": 1 / (sum(decode_times) / len(decode_times)),
    }

# 测试不同 prompt 长度
for prompt_len in [128, 512, 1024, 2048, 4096]:
    result = measure_prefill_decode(prompt_len)
    print(f"Prompt={result['prompt_len']:>5}: "
          f"Prefill={result['prefill_time_ms']:>8.2f}ms "
          f"({result['prefill_tok_per_s']:>10,.0f} tok/s) | "
          f"Decode={result['avg_decode_time_ms']:>8.2f}ms/tok "
          f"({result['decode_tok_per_s']:>8,.0f} tok/s)")
```

**(b)** 回答以下问题：
1. Prefill 吞吐量（tokens/s）随 prompt 长度如何变化？是线性增长还是亚线性？为什么？
2. Decode 每步耗时随已生成 token 数如何变化？为什么？
3. 计算 Prefill 和 Decode 的吞吐量比值，与理论分析（约 100-1000x）是否吻合？

**(c)** 修改代码，在 Decode 阶段使用不同的 batch size（1, 4, 16），观察：
- Decode 每步耗时如何变化？
- 总吞吐量（tokens/s × batch_size）如何变化？
- 在什么 batch size 下 Decode 开始接近 compute-bound？

---

## 练习 3：Block Table 模拟实现

### 题目

实现一个简化版的 Block Manager，理解 PagedAttention 的核心数据结构。

```python
from typing import Dict, List, Optional
import random

class SimpleBlockManager:
    """简化版 Block Manager，模拟 vLLM 的 KV Cache 管理。"""
    
    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 总 Block 数量
            block_size: 每个 Block 的 token 容量
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        # TODO: 初始化空闲 Block 池
        self.free_blocks: List[int] = None  # 空闲 Block 编号列表
        
        # TODO: 初始化 Block Table
        # block_tables[req_id] = [physical_block_0, physical_block_1, ...]
        self.block_tables: Dict[str, List[int]] = None
        
        # TODO: 记录每个请求当前的 token 数
        self.seq_lens: Dict[str, int] = None
    
    def allocate(self, req_id: str, num_tokens: int) -> bool:
        """为新请求分配初始 Block。
        
        Args:
            req_id: 请求 ID
            num_tokens: 初始 token 数 (prompt length)
        
        Returns:
            是否成功分配
        """
        # TODO: 
        # 1. 计算需要多少个 Block: ceil(num_tokens / block_size)
        # 2. 检查是否有足够的空闲 Block
        # 3. 从 free_blocks 中取出所需数量的 Block
        # 4. 更新 block_tables 和 seq_lens
        pass
    
    def append_token(self, req_id: str) -> bool:
        """为请求追加 1 个 token (Decode 阶段)。
        
        如果当前最后一个 Block 已满，需要分配新 Block。
        
        Returns:
            是否成功
        """
        # TODO:
        # 1. seq_lens[req_id] += 1
        # 2. 检查当前最后一个 Block 是否已满
        #    if seq_lens[req_id] % block_size == 0 之前的 Block 已满
        # 3. 如果已满，分配新 Block 并追加到 block_tables
        pass
    
    def free(self, req_id: str):
        """释放请求的所有 Block。"""
        # TODO:
        # 1. 将请求的所有 Block 归还到 free_blocks
        # 2. 清理 block_tables 和 seq_lens
        pass
    
    def get_physical_location(self, req_id: str, token_pos: int):
        """获取指定 token 的物理位置。
        
        Returns:
            (physical_block_idx, offset_in_block)
        """
        # TODO:
        # 1. logical_block = token_pos // block_size
        # 2. offset = token_pos % block_size
        # 3. physical_block = block_tables[req_id][logical_block]
        pass
    
    def get_utilization(self) -> float:
        """计算 Block 利用率。"""
        # TODO: 已分配 blocks / 总 blocks
        pass
    
    def get_internal_fragmentation(self) -> float:
        """计算内部碎片率。"""
        # TODO: 
        # 对每个请求，计算 (分配的 slots - 实际使用的 slots) / 分配的 slots
        pass


# ===== 测试代码 =====
def test_block_manager():
    bm = SimpleBlockManager(num_blocks=100, block_size=16)
    
    # 测试 1: 分配
    assert bm.allocate("req_A", num_tokens=50)  # 需要 4 个 block
    assert bm.allocate("req_B", num_tokens=10)  # 需要 1 个 block
    assert bm.allocate("req_C", num_tokens=33)  # 需要 3 个 block
    
    print(f"分配后利用率: {bm.get_utilization():.1%}")
    print(f"内部碎片率: {bm.get_internal_fragmentation():.1%}")
    
    # 测试 2: Decode 追加
    for _ in range(30):  # 为 req_A 生成 30 个 token
        assert bm.append_token("req_A")
    print(f"req_A 追加 30 tokens 后的 block 数: "
          f"{len(bm.block_tables['req_A'])}")
    
    # 测试 3: 释放
    bm.free("req_B")
    print(f"释放 req_B 后利用率: {bm.get_utilization():.1%}")
    
    # 测试 4: 物理位置查询
    block, offset = bm.get_physical_location("req_A", token_pos=35)
    print(f"req_A token 35 → Block {block}, Offset {offset}")
    
    # 测试 5: 大量请求
    for i in range(20):
        success = bm.allocate(f"req_{i}", num_tokens=random.randint(10, 100))
        if not success:
            print(f"Block 用尽! 无法分配第 {i} 个请求")
            break
    
    print(f"最终利用率: {bm.get_utilization():.1%}")
    print(f"最终内部碎片率: {bm.get_internal_fragmentation():.1%}")

test_block_manager()
```

**要求**：
1. 完成所有 `TODO` 部分
2. 运行测试确保通过
3. 思考并回答：当 `block_size` 从 16 改为 1 时，内部碎片如何变化？有什么代价？

---

## 练习 4：MHA vs GQA vs MLA 显存对比可视化

### 题目

编写一个 Python 脚本，可视化不同 Attention 变体在不同序列长度下的 KV Cache 显存占用。

```python
import matplotlib.pyplot as plt
import numpy as np

def calculate_kv_cache(
    seq_lens: np.ndarray,
    num_layers: int,
    attention_type: str,
    # MHA/GQA 参数
    num_kv_heads: int = 0,
    head_dim: int = 128,
    # MLA 参数
    d_c: int = 0,
    d_h_rope: int = 0,
    dtype_bytes: int = 2,  # FP16
) -> np.ndarray:
    """计算不同序列长度下的 KV Cache 大小 (GB)。
    
    Returns:
        各序列长度对应的 KV Cache 大小 (GB)
    """
    if attention_type in ("MHA", "GQA"):
        # TODO: 使用标准公式
        kv_per_token = None
    elif attention_type == "MLA":
        # TODO: 使用 MLA 公式
        kv_per_token = None
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return seq_lens * kv_per_token / (1024**3)


# 定义模型配置
models = {
    "GPT-3 175B (MHA)": {
        "num_layers": 96, "attention_type": "MHA",
        "num_kv_heads": 96, "head_dim": 128,
    },
    "LLaMA-3-70B (GQA-8)": {
        "num_layers": 80, "attention_type": "GQA",
        "num_kv_heads": 8, "head_dim": 128,
    },
    "LLaMA-3-8B (GQA-8)": {
        "num_layers": 32, "attention_type": "GQA",
        "num_kv_heads": 8, "head_dim": 128,
    },
    "DeepSeek-V3 (MLA)": {
        "num_layers": 61, "attention_type": "MLA",
        "d_c": 512, "d_h_rope": 64,
    },
}

seq_lens = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])

# TODO: 
# 1. 对每个模型调用 calculate_kv_cache
# 2. 使用 matplotlib 画折线图
#    - X 轴: 序列长度 (log scale)
#    - Y 轴: KV Cache 大小 (GB)
#    - 每个模型一条线
# 3. 添加水平虚线表示 GPU 显存容量 (80 GB, 40 GB)
# 4. 添加标题、图例、网格

plt.figure(figsize=(12, 7))

for name, config in models.items():
    kv_sizes = calculate_kv_cache(seq_lens, **config)
    plt.plot(seq_lens, kv_sizes, marker='o', label=name, linewidth=2)

plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='H100 80GB')
plt.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='A6000 48GB')

plt.xscale('log', base=2)
plt.xlabel('Sequence Length')
plt.ylabel('KV Cache Size (GB) per Request')
plt.title('KV Cache Memory Usage: MHA vs GQA vs MLA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kv_cache_comparison.png', dpi=150)
plt.show()
```

**要求**：
1. 完成 `calculate_kv_cache` 函数
2. 运行脚本生成对比图
3. 回答以下问题：
   - 在 128K 序列长度下，MHA (GPT-3) 的单请求 KV Cache 是 GQA (LLaMA-3-70B) 的多少倍？
   - DeepSeek-V3 (MLA) 在多长序列时 KV Cache 才会超过 80 GB？
   - 如果只看 KV Cache，哪个模型最适合超长上下文部署？

---

## 练习 5：vLLM 部署实验与显存验证

### 题目

本练习需要一台有 GPU 的机器（至少 1 张 16GB+ 显存的卡）。

**(a)** 使用 vLLM 部署一个小模型，验证显存计算：

```bash
# 安装 vLLM
pip install vllm

# 部署 Qwen2.5-0.5B (足够小，单卡 16GB 即可)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --block-size 16 \
    --dtype bfloat16 \
    2>&1 | tee vllm_startup.log
```

在启动日志中找到以下信息并记录：
1. `# GPU blocks` — GPU 上分配了多少个 Block？
2. `block_size` — 每个 Block 多少个 token？
3. 模型实际使用了多少显存？

**(b)** 手动验证 Block 数量：

```python
# Qwen2.5-0.5B 参数
L = 24       # 层数
n_kv = 2     # KV heads (GQA)
d_h = 64     # Head dim
block_size = 16
dtype_bytes = 2  # BF16

# 1. 计算每个 Block 的 KV Cache 大小 (所有层合计)
kv_per_block = 2 * L * n_kv * d_h * block_size * dtype_bytes
print(f"KV per block: {kv_per_block:,} bytes = {kv_per_block/1024:.1f} KB")

# 2. 查看 GPU 总显存
# nvidia-smi 或 torch.cuda.get_device_properties(0).total_memory

# 3. 估算可用 KV Cache 空间
# 可用 = 总显存 × 0.9 - 模型权重 - 激活开销

# 4. 计算理论 Block 数量
# 理论 blocks = 可用空间 / kv_per_block

# 5. 与 vLLM 日志中的实际 Block 数量对比
```

**(c)** 改变参数，观察 Block 数量变化：

```bash
# 实验 1: 改变 gpu_memory_utilization
--gpu-memory-utilization 0.7   # 记录 Block 数
--gpu-memory-utilization 0.9   # 记录 Block 数
--gpu-memory-utilization 0.95  # 记录 Block 数

# 实验 2: 改变 max_model_len
--max-model-len 2048    # 记录 Block 数
--max-model-len 8192    # 记录 Block 数
--max-model-len 32768   # 记录 Block 数

# 实验 3: 改变 kv-cache-dtype (如果 GPU 支持)
--kv-cache-dtype auto   # BF16, 记录 Block 数
--kv-cache-dtype fp8    # FP8, 记录 Block 数 (需要 H100/L40S)
```

**(d)** 发送并发请求，观察显存使用：

```python
import openai
import concurrent.futures

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def send_request(prompt_len: int, max_tokens: int):
    """发送一个请求并返回 TTFT。"""
    prompt = "Hello " * prompt_len
    import time
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    t1 = time.perf_counter()
    return {
        "prompt_len": prompt_len,
        "completion_tokens": response.usage.completion_tokens,
        "total_time_ms": (t1 - t0) * 1000,
    }

# 逐步增加并发，观察 nvidia-smi 中的显存变化
for concurrency in [1, 4, 8, 16, 32]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, prompt_len=500, max_tokens=200)
            for _ in range(concurrency)
        ]
        results = [f.result() for f in futures]
    
    avg_time = sum(r["total_time_ms"] for r in results) / len(results)
    print(f"并发 {concurrency:>3}: 平均耗时 {avg_time:>8.1f} ms")
    
    # 在另一个终端运行 nvidia-smi 查看显存使用
    # 记录 Used Memory 数值
```

**报告要求**：
1. 将理论计算与实际观测的 Block 数量进行对比，计算误差
2. 分析 `gpu_memory_utilization` 变化对 Block 数量的影响是否符合线性关系
3. 观察并发数增加时显存增长是否与 KV Cache 计算一致
4. 总结你观察到的 Prefill 和 Decode 阶段的显存行为差异

---

## 通用提示

- **计算验证**：对于所有手动计算，建议写 Python 代码验证，避免单位换算错误
- **单位注意**：GB = $1024^3$ bytes（这里我们使用二进制前缀，与 `nvidia-smi` 一致）
- **工程思维**：实际系统中还有 CUDA context（~500MB）、PyTorch 分配器开销等额外显存消耗，估算时应留有余量
- **模型精度**：注意区分模型权重的精度（BF16/INT8/INT4）和 KV Cache 的精度（通常 FP16/BF16/FP8），两者可以独立设置
