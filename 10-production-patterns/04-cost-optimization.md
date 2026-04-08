# 成本优化

> GPU 是 LLM 推理最大的成本项。本节从 TCO 分析框架出发，
> 系统介绍推理服务的成本优化策略和硬件选型决策。

## 1. TCO 分析框架

### 1.1 成本构成

LLM 推理服务的 Total Cost of Ownership (TCO) 包含三大部分：

```
┌────────────────────────────────────────────┐
│            LLM 推理 TCO 构成               │
├────────────────────────────────────────────┤
│                                            │
│  ┌──────────────────────┐                  │
│  │ 硬件/租赁成本  (70-85%) │ ← 主要成本     │
│  └──────────────────────┘                  │
│  ┌──────────────────────┐                  │
│  │ 能耗成本       (8-15%) │                │
│  └──────────────────────┘                  │
│  ┌──────────────────────┐                  │
│  │ 人力/运维成本   (5-15%) │                │
│  └──────────────────────┘                  │
│                                            │
└────────────────────────────────────────────┘
```

### 1.2 成本计算公式

```python
def calculate_inference_tco(
    # 硬件参数
    gpu_type: str,
    num_gpus: int,
    gpu_hourly_cost: float,  # $/hour per GPU (云租赁)
    
    # 工作负载参数
    avg_input_tokens: int,
    avg_output_tokens: int,
    requests_per_day: int,
    
    # 性能参数
    prefill_tokens_per_sec: int,     # 每 GPU prefill 吞吐
    decode_tokens_per_sec: int,      # 每 GPU decode 吞吐
    cache_hit_rate: float = 0.0,     # prompt cache 命中率
    
    # 时间参数
    months: int = 12,
    utilization_rate: float = 0.7,   # GPU 平均利用率
):
    """计算推理服务的 TCO"""
    
    hours = months * 30 * 24
    
    # 硬件成本
    hardware_cost = num_gpus * gpu_hourly_cost * hours
    
    # 能耗成本 (假设 PUE=1.3)
    gpu_power_map = {
        "H100_SXM": 700, "H100_PCIe": 350, "H200": 700,
        "A100_80GB": 300, "A100_40GB": 250, "L40S": 350,
    }
    power_watts = gpu_power_map.get(gpu_type, 350) * num_gpus
    pue = 1.3
    electricity_rate = 0.08  # $/kWh
    energy_cost = power_watts * pue / 1000 * hours * electricity_rate
    
    # 有效处理能力 (考虑利用率和 cache)
    effective_prefill_capacity = (
        prefill_tokens_per_sec * num_gpus * utilization_rate * 3600 * 24
    )
    # Cache hit 的 tokens 不需要 prefill 计算
    daily_prefill_tokens = requests_per_day * avg_input_tokens * (1 - cache_hit_rate)
    daily_decode_tokens = requests_per_day * avg_output_tokens
    
    # 每百万 tokens 的成本
    total_daily_tokens = (daily_prefill_tokens + daily_decode_tokens)
    daily_cost = (hardware_cost + energy_cost) / (months * 30)
    cost_per_million_tokens = (daily_cost / total_daily_tokens) * 1e6
    
    result = {
        "total_tco": hardware_cost + energy_cost,
        "hardware_cost": hardware_cost,
        "energy_cost": energy_cost,
        "monthly_cost": (hardware_cost + energy_cost) / months,
        "cost_per_million_tokens": cost_per_million_tokens,
        "daily_capacity_tokens": effective_prefill_capacity,
    }
    
    return result

# 示例：Llama-3.1-70B on 4x H100 SXM
result = calculate_inference_tco(
    gpu_type="H100_SXM",
    num_gpus=4,
    gpu_hourly_cost=3.50,  # 云厂商典型价格
    avg_input_tokens=1500,
    avg_output_tokens=500,
    requests_per_day=100_000,
    prefill_tokens_per_sec=15000,
    decode_tokens_per_sec=2000,
    cache_hit_rate=0.6,  # 60% prompt cache hit
    months=12,
    utilization_rate=0.65,
)

for k, v in result.items():
    print(f"{k}: ${v:,.2f}" if "cost" in k else f"{k}: {v:,.0f}")
```

### 1.3 成本敏感度分析

不同因素对成本的影响程度：

| 优化手段 | 成本降低幅度 | 实施难度 | 对延迟的影响 |
|----------|------------|---------|------------|
| Prompt Caching (60% hit) | 30-40% | 低 | 改善 TTFT |
| FP8 量化 | 25-35% | 中 | 轻微增加 |
| Speculative Decoding | 15-25% | 高 | 改善 TBT |
| 按需扩缩容 | 20-40% | 中 | 无影响 |
| 优化 batch size | 10-20% | 低 | 需权衡 |
| 换更大显存 GPU | 因场景而异 | 中 | 通常改善 |

## 2. 优化策略详解

### 2.1 Prompt Caching

Prompt Caching 是成本优化中 ROI 最高的策略。

**原理回顾：** 相同 prompt 前缀的 KV Cache 可以复用，避免重复的 prefill 计算。

**成本影响计算：**

```python
def prompt_caching_savings(
    requests_per_month: int,
    avg_prompt_tokens: int,
    avg_cached_tokens: int,
    prefill_cost_per_million: float,  # $/M tokens (无 cache)
    cache_cost_per_million: float,    # $/M tokens (cache hit)
):
    """计算 prompt caching 的月度节省"""
    
    # 无 cache 的成本
    total_prompt_tokens = requests_per_month * avg_prompt_tokens
    no_cache_cost = total_prompt_tokens / 1e6 * prefill_cost_per_million
    
    # 有 cache 的成本
    cached_tokens = requests_per_month * avg_cached_tokens
    uncached_tokens = total_prompt_tokens - cached_tokens
    
    with_cache_cost = (
        uncached_tokens / 1e6 * prefill_cost_per_million +
        cached_tokens / 1e6 * cache_cost_per_million
    )
    
    savings = no_cache_cost - with_cache_cost
    savings_pct = savings / no_cache_cost * 100
    
    return {
        "monthly_without_cache": no_cache_cost,
        "monthly_with_cache": with_cache_cost,
        "monthly_savings": savings,
        "savings_percentage": savings_pct,
    }

# OpenAI GPT-4o 定价示例
result = prompt_caching_savings(
    requests_per_month=1_000_000,
    avg_prompt_tokens=3000,
    avg_cached_tokens=2000,   # 2000/3000 = 66% cache hit
    prefill_cost_per_million=2.50,   # $2.50/M input tokens
    cache_cost_per_million=1.25,     # $1.25/M cached tokens (50% 折扣)
)
# 月度节省约 $2,500 (33%)
```

**最大化 Cache 命中率的设计模式：**

```python
# Pattern 1: 将稳定内容放在 prompt 最前面
messages = [
    # 不变的部分（会被缓存）
    {"role": "system", "content": LONG_SYSTEM_PROMPT},        # 2000 tokens
    {"role": "user", "content": FEW_SHOT_EXAMPLES},           # 1000 tokens
    # 变化的部分（不会被缓存）
    {"role": "user", "content": user_specific_query},          # 200 tokens
]

# Pattern 2: 使用 cache_control 显式标记（Anthropic API）
messages_with_cache = [
    {
        "role": "system",
        "content": LONG_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"}  # 明确标记缓存
    },
    {"role": "user", "content": user_query}
]
```

### 2.2 量化推理

量化通过降低精度减少显存占用和计算量，是最直接的成本优化。

**不同量化精度的成本影响：**

| 精度 | 模型大小 (70B) | 每 GPU 可容纳 | 吞吐提升 | 质量影响 |
|------|--------------|-------------|---------|---------|
| FP16 | 140 GB | 需要 2x H100 | 基准 | 无 |
| FP8 (W8A8) | 70 GB | 1x H100 | 1.5-2x | 极小 |
| INT4 (W4A16) | 35 GB | 1x H100 (更多 KV Cache) | 1.2-1.5x | 小 |
| INT4 (GPTQ/AWQ) | 35 GB | 1x A100 80GB | 1.3x | 可接受 |

**vLLM 量化配置：**

```bash
# FP8 量化 (推荐，H100/H200)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --quantization fp8 \
    --tensor-parallel-size 2   # FP8 只需要 2 卡而非 4 卡

# GPTQ INT4 量化
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-3.1-70B-GPTQ \
    --quantization gptq \
    --tensor-parallel-size 2

# KV Cache 量化 (独立于权重量化)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8   # KV Cache 使用 FP8，节省一半显存
```

**量化的成本收益分析：**

```python
def quantization_roi(
    fp16_gpus: int,
    fp16_cost_per_gpu_hour: float,
    quantized_gpus: int,
    quantized_cost_per_gpu_hour: float,
    quality_loss_pct: float,  # 质量损失 (benchmark 衡量)
    hours_per_month: int = 720,
):
    """量化的投资回报分析"""
    fp16_monthly = fp16_gpus * fp16_cost_per_gpu_hour * hours_per_month
    quant_monthly = quantized_gpus * quantized_cost_per_gpu_hour * hours_per_month
    
    savings = fp16_monthly - quant_monthly
    savings_pct = savings / fp16_monthly * 100
    
    # 质量-成本权衡
    cost_per_quality_point = savings / max(quality_loss_pct, 0.01)
    
    print(f"FP16 月成本: ${fp16_monthly:,.0f} ({fp16_gpus} GPUs)")
    print(f"量化月成本: ${quant_monthly:,.0f} ({quantized_gpus} GPUs)")
    print(f"月度节省: ${savings:,.0f} ({savings_pct:.1f}%)")
    print(f"质量损失: {quality_loss_pct:.1f}%")

# 70B 模型: FP16 需要 4x A100 vs FP8 需要 2x H100
quantization_roi(
    fp16_gpus=4, fp16_cost_per_gpu_hour=2.20,         # 4x A100
    quantized_gpus=2, quantized_cost_per_gpu_hour=3.50, # 2x H100 FP8
    quality_loss_pct=0.3,  # FP8 质量损失约 0.3%
)
```

### 2.3 Speculative Decoding

Speculative Decoding 通过小模型草稿 + 大模型验证的方式减少 decode 步数。

**成本影响：**

```
传统 Decode (生成 100 tokens):
  = 100 步 × 每步读取全部模型权重
  = 100 × 140GB (70B FP16) = 14TB 数据读取

Speculative Decoding (accept rate = 0.7, draft length = 5):
  每轮: draft 5 tokens + verify 5 tokens
  有效每轮生成: 1 + 5 × 0.7 ≈ 4.5 tokens
  生成 100 tokens: 约 22 轮
  = 22 × (draft 读取 + target 读取)
  ≈ 22 × (14GB + 140GB) = 3.4TB
  
  节省: (14TB - 3.4TB) / 14TB = 75% 的数据读取！
```

但 speculative decoding 的成本优势取决于：
- Draft model 的额外 GPU 显存开销
- Accept rate（接受率越高越划算）
- 是否能利用同一组 GPU（MTP 方式更优）

### 2.4 Batch 优化

增大 batch size 是提高 GPU 利用率最直接的方法：

```
GPU 利用率 vs Batch Size (70B FP16, H100):

Batch Size  |  GPU Util  |  Tokens/s/GPU  |  Cost/M tokens
     1      |    15%     |       24       |     $40.50
     4      |    35%     |       88       |     $11.05
    16      |    65%     |      280       |      $3.47
    64      |    85%     |      750       |      $1.30
   128      |    90%     |      920       |      $1.06
   
成本随 batch 增大而非线性下降，因为更大 batch 需要更多 KV Cache 显存
```

**优化 batch 的配置：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --max-num-seqs 128        # 最大并发序列数（增大 batch）
    --max-num-batched-tokens 8192  # 每步最大 tokens
    --gpu-memory-utilization 0.92  # 更多显存给 KV Cache
```

### 2.5 按需扩缩容

大多数服务的流量有明显的时间模式：

```
请求量
│
│    ╱╲
│   ╱  ╲        ╱╲
│  ╱    ╲      ╱  ╲
│ ╱      ╲    ╱    ╲
│╱        ╲  ╱      ╲
│          ╲╱        ╲___________
│
└────────────────────────────────── 时间
 0:00  6:00  12:00  18:00  24:00

高峰: 10:00-14:00, 19:00-22:00  → 4 replicas
平峰: 其他时间                   → 2 replicas
低谷: 02:00-06:00               → 1 replica
```

**Kubernetes HPA 配置：**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 1
  maxReplicas: 8
  metrics:
    # 基于自定义 Prometheus 指标扩缩容
    - type: Pods
      pods:
        metric:
          name: vllm_gpu_cache_usage_perc
        target:
          type: AverageValue
          averageValue: "0.8"   # KV Cache 使用率 > 80% 时扩容
    - type: Pods
      pods:
        metric:
          name: vllm_num_requests_waiting
        target:
          type: AverageValue
          averageValue: "10"    # 等待队列 > 10 时扩容
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # 1 分钟内持续高负载才扩容
      policies:
        - type: Pods
          value: 2              # 每次最多加 2 个 replica
          periodSeconds: 120
    scaleDown:
      stabilizationWindowSeconds: 300  # 5 分钟低负载才缩容
      policies:
        - type: Pods
          value: 1              # 每次最多减 1 个 replica
          periodSeconds: 300
```

**注意事项：**
- LLM 推理服务的启动时间较长（模型加载需要数分钟），扩容不能太激进
- 缩容时需要 graceful shutdown，等待正在处理的请求完成
- 考虑使用预热池（warm pool）：保持 1-2 个预加载好模型的备用实例

## 3. 硬件选型指南

### 3.1 主流 GPU 对比

| 规格 | H100 SXM | H100 PCIe | H200 | A100 80GB | L40S |
|------|----------|-----------|------|-----------|------|
| FP16 TFLOPS | 989 | 756 | 989 | 312 | 362 |
| FP8 TFLOPS | 1,979 | 1,513 | 1,979 | N/A | 724 |
| HBM 容量 | 80 GB | 80 GB | 141 GB | 80 GB | 48 GB (GDDR6X) |
| HBM 带宽 | 3.35 TB/s | 2.0 TB/s | 4.8 TB/s | 2.0 TB/s | 0.86 TB/s |
| NVLink | 900 GB/s | N/A | 900 GB/s | 600 GB/s | N/A |
| TDP | 700W | 350W | 700W | 300W | 350W |
| 云价格 (参考) | $3.50/h | $2.50/h | $4.50/h | $2.20/h | $1.20/h |

### 3.2 场景化硬件推荐

| 场景 | 推荐硬件 | 理由 | 参考配置 |
|------|---------|------|---------|
| **低延迟、高并发** (在线客服) | H100 SXM / H200 | 高带宽降低 TBT，大显存支持高并发 | 4x H100 SXM, TP=4 |
| **高吞吐、成本敏感** (批量处理) | A100 80GB | 性价比最高，吞吐/$ 最优 | 8x A100, TP=4 × 2 replicas |
| **超长上下文** (128K+) | H200 (141GB) | 超大显存容纳长序列 KV Cache | 4x H200, TP=4 |
| **小模型、高并发** (7B-13B) | L40S | 成本最低，小模型不需要高带宽 | 2x L40S, TP=2 |
| **Mixture of Experts** (DeepSeek) | H100 SXM + NVLink | Expert 需要大显存 + 快速通信 | 8x H100 SXM, EP=8 |
| **边缘部署** | L40S / L4 | 低功耗、低成本 | 1x L40S |

### 3.3 自建 vs 云租赁决策

```python
def build_vs_rent(
    num_gpus: int,
    gpu_purchase_price: float,  # 购买单价
    gpu_rent_hourly: float,     # 租赁时价
    server_cost: float,         # 服务器其他硬件成本
    data_center_monthly: float, # 机房/电费/网络月费
    it_staff_monthly: float,    # 运维人力成本/月
    gpu_lifecycle_years: int = 3,  # GPU 生命周期
    utilization_rate: float = 0.7,  # 平均利用率
):
    """比较自建和云租赁的成本"""
    months = gpu_lifecycle_years * 12
    
    # 自建成本
    build_hardware = num_gpus * gpu_purchase_price + server_cost
    build_monthly_opex = data_center_monthly + it_staff_monthly
    build_total = build_hardware + build_monthly_opex * months
    build_effective_hourly = build_total / (months * 30 * 24 * num_gpus)
    
    # 云租赁成本 (只为实际使用付费)
    rent_total = (
        num_gpus * gpu_rent_hourly * 24 * 30 * months * utilization_rate
    )
    rent_effective_hourly = gpu_rent_hourly * utilization_rate
    
    # 盈亏平衡点
    # build_hardware + build_monthly * M = rent_hourly * num_gpus * 24 * 30 * utilization * M
    monthly_rent = num_gpus * gpu_rent_hourly * 24 * 30 * utilization_rate
    if monthly_rent > build_monthly_opex:
        breakeven_months = build_hardware / (monthly_rent - build_monthly_opex)
    else:
        breakeven_months = float('inf')
    
    print(f"=== 自建 ({gpu_lifecycle_years} 年) ===")
    print(f"  硬件一次性投入: ${build_hardware:,.0f}")
    print(f"  月运营成本: ${build_monthly_opex:,.0f}")
    print(f"  总成本: ${build_total:,.0f}")
    print(f"  等效时价/GPU: ${build_effective_hourly:.2f}")
    print()
    print(f"=== 云租赁 ({gpu_lifecycle_years} 年) ===")
    print(f"  月成本: ${monthly_rent:,.0f}")
    print(f"  总成本: ${rent_total:,.0f}")
    print(f"  等效时价/GPU (含利用率): ${rent_effective_hourly:.2f}")
    print()
    print(f"盈亏平衡点: {breakeven_months:.0f} 个月")
    
    if build_total < rent_total:
        print(f"结论: 自建更划算，{gpu_lifecycle_years}年节省 ${rent_total - build_total:,.0f}")
    else:
        print(f"结论: 云租赁更划算，{gpu_lifecycle_years}年节省 ${build_total - rent_total:,.0f}")

# 示例：8x H100 集群
build_vs_rent(
    num_gpus=8,
    gpu_purchase_price=30_000,     # $30K per H100
    gpu_rent_hourly=3.50,          # $3.50/h cloud
    server_cost=50_000,            # 服务器其他硬件
    data_center_monthly=5_000,     # 机房电费网络
    it_staff_monthly=8_000,        # 半个全职运维
    gpu_lifecycle_years=3,
    utilization_rate=0.70,
)
```

## 4. Spot Instance 策略

### 4.1 适用场景

Spot/Preemptible 实例价格通常是按需实例的 30-70%，但可能随时被回收。

| 场景 | 是否适合 Spot | 原因 |
|------|-------------|------|
| 批量推理 / 离线评估 | 适合 | 可以接受中断和重试 |
| 非关键在线服务 | 部分适合 | 需要 fallback 机制 |
| 核心在线服务 | 不适合 | 延迟和可用性要求高 |
| 模型评测 / Benchmarking | 非常适合 | 可中断可重试 |

### 4.2 混合部署策略

```
┌─────────────────────────────────────────────┐
│            混合 Spot + On-Demand            │
├─────────────────────────────────────────────┤
│                                             │
│  On-Demand (2 replicas, 始终运行):          │
│  ├── replica-0: 保证基线容量               │
│  └── replica-1: 保证基线容量               │
│                                             │
│  Spot (0-4 replicas, 按需):                │
│  ├── replica-2: 弹性扩展 (Spot)            │
│  ├── replica-3: 弹性扩展 (Spot)            │
│  ├── replica-4: 弹性扩展 (Spot)            │
│  └── replica-5: 弹性扩展 (Spot)            │
│                                             │
│  策略:                                      │
│  - 基线流量: On-Demand 处理                │
│  - 高峰流量: Spot 实例分担                 │
│  - Spot 被回收: 流量回退到 On-Demand       │
│  - On-Demand 过载: 降级或排队               │
└─────────────────────────────────────────────┘
```

## 5. 成本计算实例

### 5.1 场景：SaaS 产品的 AI 助手

```
业务参数:
- DAU: 10,000 用户
- 每用户每天: 20 次对话
- 每次对话: 平均 2000 input tokens + 500 output tokens
- SLA: TTFT < 2s, TBT < 80ms

日请求量: 200,000 requests/day
日 token 量: 200K × 2000 = 400M input + 200K × 500 = 100M output

模型: Llama-3.1-70B-Instruct (FP8)
硬件: 2x H100 SXM (TP=2), 3 replicas
```

```python
# 方案 A: 无 Prompt Caching
monthly_cost_a = 6 * 3.50 * 24 * 30  # 6 GPUs × $3.50/h × 720h
# = $15,120/month

# 方案 B: 有 Prompt Caching (system prompt 1500 tokens, 75% hit rate)
# 有效 prefill tokens = 400M × (1 - 0.75 × 1500/2000) = 400M × 0.4375 = 175M
# 减少了 56% 的 prefill 计算，可以减少到 2 replicas
monthly_cost_b = 4 * 3.50 * 24 * 30  # 4 GPUs × $3.50/h × 720h
# = $10,080/month
# 节省: $5,040/month (33%)

# 方案 C: B + 按需缩容 (夜间流量仅 20%)
# 高峰 16h: 4 GPUs, 低谷 8h: 2 GPUs
monthly_cost_c = (4 * 3.50 * 16 + 2 * 3.50 * 8) * 30
# = $8,400/month
# 节省: $6,720/month (44%)

# 方案 D: C + FP8 量化 (从 TP=2 降到 TP=1, 但保持 3 replicas)
# FP8 下 70B 模型可以放入单卡 H100
# 高峰 3 GPUs, 低谷 1 GPU
monthly_cost_d = (3 * 3.50 * 16 + 1 * 3.50 * 8) * 30
# = $5,880/month
# 节省: $9,240/month (61%)

print(f"""
成本优化对比:
方案 A (基准):     ${15120:>8,}/month
方案 B (+缓存):    ${10080:>8,}/month  (-33%)
方案 C (+缩容):    ${8400:>8,}/month   (-44%)  
方案 D (+FP8):     ${5880:>8,}/month   (-61%)
""")
```

### 5.2 每用户每月成本

```python
monthly_cost = 5880  # 方案 D
mau = 10000
cost_per_user_month = monthly_cost / mau
print(f"每用户每月 AI 推理成本: ${cost_per_user_month:.2f}")
# ≈ $0.59/user/month
```

## 6. 成本监控

### 6.1 成本相关指标

在 Grafana 中添加成本监控面板：

```promql
# 每小时 GPU 成本 (假设 $3.50/GPU/hour)
count(up{job="vllm"}) * 3.50

# 每百万 tokens 成本
(count(up{job="vllm"}) * 3.50) / 
(rate(vllm:generation_tokens_total[1h]) * 3600 / 1e6)

# 每请求平均成本
(count(up{job="vllm"}) * 3.50) / 
(rate(vllm:request_success_total[1h]) * 3600)

# GPU 利用率（成本效率指标）
avg(DCGM_FI_DEV_GPU_UTIL{job="dcgm"}) / 100
```

## 7. 总结

成本优化是一个系统工程，需要在多个维度上同时发力：

| 优化层面 | 关键策略 | 典型收益 |
|----------|---------|---------|
| **算法层** | Prompt Caching, Speculative Decoding | 30-50% |
| **精度层** | FP8/INT4 量化, KV Cache 量化 | 25-50% |
| **调度层** | Batch 优化, Continuous Batching | 10-30% |
| **基础设施层** | 按需扩缩容, Spot Instance | 20-40% |
| **架构层** | 硬件选型, 自建 vs 租赁 | 因场景而异 |

**核心原则：** 先用低成本、低风险的手段（Prompt Caching、调参），再考虑需要工程投入的优化（量化、Speculative Decoding），最后考虑架构级变更（硬件更换、自建集群）。

---

> **延伸阅读：**
> - [Anthropic Prompt Caching 定价](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
> - [vLLM 量化指南](https://docs.vllm.ai/en/latest/quantization/)
> - 各云厂商 GPU 实例定价页面
