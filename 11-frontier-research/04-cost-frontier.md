# 推理成本前沿

> 从 $60/M tokens 到 $0.10/M tokens——LLM 推理成本的下降之路与经济学分析

## 1. 成本下降的历史回顾

### 1.1 GPT-4 级模型的成本演变

LLM 推理成本在短短三年内经历了惊人的下降。以下是 GPT-4 级别（frontier model）输入 token 价格的变化：

```
GPT-4 级模型 每百万输入 token 价格变化:

2023.03  GPT-4 launch          $30.00 /M tokens (input)
2023.06  Claude 2              $11.02 /M tokens
2023.11  GPT-4 Turbo           $10.00 /M tokens
2024.05  GPT-4o                 $5.00 /M tokens
2024.06  Claude 3.5 Sonnet      $3.00 /M tokens
2024.07  GPT-4o-mini            $0.15 /M tokens (小模型, 但接近 GPT-4 水平)
2024.09  o1-preview            $15.00 /M tokens (reasoning model, 价格回升)
2024.12  DeepSeek-V3 API        $0.27 /M tokens (input, cache miss)
2025.01  DeepSeek-R1 API        $0.55 /M tokens (reasoning, 极低价格)
2025.03  GPT-4o                 $2.50 /M tokens (降价 50%)
2025.06  Claude 3.5 Haiku       $0.80 /M tokens
2025.12  Gemini 2.0 Flash       $0.10 /M tokens (预估)

→ 3 年内 frontier model 价格下降 ~100-300×!
→ 平均每年下降 ~5-7×
→ 远超摩尔定律 (~2×/2年)
```

### 1.2 成本下降的驱动因素

```
成本下降 = 硬件进步 × 软件优化 × 模型效率 × 规模效应

1. 硬件进步 (~2-3×):
   A100 (2020) → H100 (2022) → B200 (2024)
   - FP16 TFLOPS: 312 → 989 → 2250
   - HBM 带宽: 2TB/s → 3.35TB/s → 8TB/s
   - 价格: ~$15K → ~$30K → ~$40K (但 FLOPS/$ 提升 3-4×)

2. 软件优化 (~3-5×):
   - FlashAttention: 2-4× attention 加速
   - PagedAttention: ~2× 显存效率 → 更大 batch
   - Continuous batching: 2-3× throughput
   - Quantization (FP16→FP8→INT4): 2-4× 显存节省
   - Speculative decoding: 2-3× decode 加速
   - torch.compile + CUDA Graph: 1.3-1.6×

3. 模型效率 (~3-10×):
   - GPT-4 (推测 ~1.8T 参数) → GPT-4o (~200B 参数?): 同等质量, 更小模型
   - MoE 架构: 激活参数只有总参数的 1/8
   - MLA: 4× KV Cache 压缩
   - 更好的训练数据: 小模型也能达到大模型的质量

4. 规模效应 (~2-5×):
   - 大规模 GPU 集群: 更高利用率
   - Prompt Caching: 减少重复计算
   - 多租户共享: prefix sharing
```

## 2. Flex Inference / Priority Inference (Google)

### 2.1 设计理念

Google Cloud 于 2024 年推出的 Flex Inference（也称 Priority Inference）基于一个简单但重要的经济学洞察：**GPU 集群的利用率永远不会持续 100%——总有波谷时段的闲置资源**。

```
GPU 集群利用率的典型模式:

利用率%
100|     ____
   |    /    \      ____
 80|   /      \    /    \
   |  /        \  /      \
 60| /          \/        \
   |/                      \
 40|                        \____
   |
 20|
   |________________________________
    0  4  8  12  16  20  24  时间

→ 峰值利用率 ~90%, 低谷 ~40%
→ 平均利用率可能只有 ~60-70%
→ 30-40% 的计算能力被浪费!

Flex Inference 的做法:
- 高优先级请求: 即时处理, 保证 SLA (正常价格)
- 低优先级请求: 排队等待闲置资源, 延迟不保证 (折扣价格)
→ 填满波谷, 提高整体利用率
→ 降低低优先级任务的成本
```

### 2.2 技术实现

```
Flex Inference 的调度架构:

┌─────────────────────────────────────┐
│           Request Router            │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ Priority │  │   Flex (Low      │ │
│  │ Queue    │  │   Priority Queue)│ │
│  └────┬─────┘  └────┬─────────────┘ │
│       │              │               │
│  ┌────▼─────────────▼────┐          │
│  │     GPU Scheduler     │          │
│  │  ┌──────────────────┐ │          │
│  │  │ 优先分配给       │ │          │
│  │  │ Priority 请求    │ │          │
│  │  │                  │ │          │
│  │  │ 空闲时处理       │ │          │
│  │  │ Flex 请求        │ │          │
│  │  └──────────────────┘ │          │
│  └───────────────────────┘          │
└─────────────────────────────────────┘

关键特性:
- Flex 请求可能被抢占 (当高优先级请求到来时)
- Flex 请求的 KV Cache 可以被驱逐到 CPU/SSD
- 完成时间不确定 (分钟到小时)
- 定价通常为标准价格的 30-50%
```

### 2.3 适用场景

```
Flex Inference 最适合的场景:

✅ 适合:
- 大规模数据处理 (文档分类, 摘要生成)
- 离线评估和测试 (模型评估, A/B test)
- 训练数据生成 (合成数据)
- 非实时的内容生成 (SEO, 营销文案)
- 研究实验 (大量 prompt 测试)

❌ 不适合:
- 用户交互式对话 (需要低延迟)
- 实时推荐 (SLA 敏感)
- Streaming 输出 (用户在等待)
- Agentic 工作流 (多步推理, 中间延迟会累积)
```

## 3. Batch API (OpenAI / Anthropic)

### 3.1 OpenAI Batch API

OpenAI 于 2024 年推出的 Batch API 提供了更结构化的异步推理方式：

```python
# OpenAI Batch API 使用示例

# Step 1: 准备 JSONL 文件
# batch_input.jsonl:
# {"custom_id": "req_1", "method": "POST", "url": "/v1/chat/completions", 
#  "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}}
# {"custom_id": "req_2", ...}
# ...

# Step 2: 上传文件
import openai
client = openai.OpenAI()

batch_input_file = client.files.create(
    file=open("batch_input.jsonl", "rb"),
    purpose="batch"
)

# Step 3: 创建 batch
batch = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"  # 24 小时内完成
)

# Step 4: 查询状态
status = client.batches.retrieve(batch.id)
# status.status: "validating" → "in_progress" → "completed"

# Step 5: 获取结果
if status.status == "completed":
    result_file = client.files.content(status.output_file_id)
```

**定价**：

```
OpenAI Batch API 定价 (截至 2025):
  GPT-4o:       $1.25 / M input tokens (vs $2.50 standard → 50% 折扣)
  GPT-4o-mini:  $0.075 / M input tokens (vs $0.15 standard → 50% 折扣)

Anthropic Message Batches API:
  Claude 3.5 Sonnet: $1.50 / M input tokens (vs $3.00 → 50% 折扣)
  Claude 3.5 Haiku:  $0.40 / M input tokens (vs $0.80 → 50% 折扣)

Google Batch Prediction:
  Gemini 1.5 Pro: 50% 折扣 (通过 Vertex AI)
```

### 3.2 Batch API 背后的技术

```
为什么 Batch API 可以提供 50% 折扣？

1. 调度灵活性
   - 不需要立即处理, 可以等待最优时机
   - 可以利用波谷资源 (类似 Flex Inference)
   - 可以跨区域调度 (利用时区差异)

2. Batching 效率
   - 预知所有 request, 可以做全局最优 batching
   - 相似 prompt 可以聚合做 prefix sharing
   - 不受延迟约束, batch size 可以设得更大

3. KV Cache 管理
   - 可以按 prompt 相似度排序, 最大化 cache hit
   - 不需要维护长时间的 KV Cache (request 完成即释放)
   - 可以使用更激进的量化 (不需要 streaming output)

4. 计算资源复用
   - 与实时请求共享 GPU (低优先级)
   - 可以使用不同精度/量化版本的模型
   - 出错可以重试 (24h 窗口足够)
```

### 3.3 Anthropic Message Batches API 特点

```python
# Anthropic 的 Message Batches API

import anthropic

client = anthropic.Anthropic()

# 创建 batch
batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": "req_1",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "分析这段代码..."}]
            }
        },
        # ... 最多 100,000 个 request
    ]
)

# 特点:
# - 单 batch 最多 100K 个 request
# - 通常在几分钟到几小时内完成 (大多数情况远快于 24h)
# - 支持 streaming results (部分完成的 request 可以先获取)
# - 与标准 API 完全相同的输出质量
```

## 4. 2023-2026 成本变化深度分析

### 4.1 按模型级别的成本对比

```
每百万 output token 价格 (按模型级别):

              2023 Q1    2024 Q1    2025 Q1    2026 Q1(预估)
─────────────────────────────────────────────────────────────
Frontier      $60.00     $30.00      $10.00     $3.00
(GPT-4级)

Mid-tier      $16.00      $4.00       $0.60     $0.15
(GPT-3.5级)

Lightweight    $2.00      $0.60       $0.08     $0.02
(小模型)

开源自部署     $8.00*     $2.00*      $0.30*    $0.08*
(70B级)

* 自部署成本包含 GPU 租赁/折旧、运维、电力
  按 H100 $2/hr, 80% 利用率估算
```

### 4.2 成本下降的分解

```
从 2023 到 2026, Frontier 模型成本下降 ~20×, 拆解如下:

因素                 | 贡献   | 具体措施
───────────────────────────────────────────────
模型架构进步         | ~4×    | GPT-4 (~1.8T?) → GPT-4o (~200B?)
                     |        | Dense → MoE, 更好的训练
硬件代际升级         | ~2×    | A100 → H100/H200 → B200
软件优化             | ~2×    | FlashAttention, PagedAttention,
                     |        | Continuous batching, CUDA Graph
量化                 | ~1.5×  | FP16 → FP8/INT8 → INT4
规模化运营           | ~1.5×  | 更大集群, 更高利用率,
                     |        | Prompt Caching, 跨区域调度
───────────────────────────────────────────────
总计                 | ~18×   | (4 × 2 × 2 × 1.5 × 1.5 ≈ 36×, 
                     |        |  实际因部分重叠约 18-20×)
```

### 4.3 Input vs Output Token 价格差异

```
为什么 output token 比 input token 贵 2-5×？

技术原因:
1. Input (Prefill): 所有 token 并行处理, compute-bound
   → 高效利用 GPU Tensor Core
   → 吞吐量高 (tokens/s/GPU)

2. Output (Decode): 逐 token 生成, memory-bound
   → GPU 利用率低 (大量时间在读 weight 和 KV Cache)
   → 吞吐量低 (每个 token 都需要读全部参数)

具体对比:
  Prefill throughput:  ~10,000 tokens/s/GPU (H100, 70B model)
  Decode throughput:   ~200 tokens/s/GPU   (same setup)
  → Decode 慢 ~50×!
  → 所以 output token 应该贵 ~50×?
  → 实际只贵 2-5×, 因为:
     - Batching 显著提高 decode 吞吐
     - batch=128 时: ~5,000 tokens/s/GPU
     - Speculative decoding 进一步提升
```

## 5. 开源 vs 闭源的成本对比

### 5.1 自部署开源模型的成本分析

```python
# 自部署成本计算 (以 LLaMA-3 70B 为例)

# 硬件配置
num_gpus = 2  # 2× H100 80GB (TP=2)
gpu_hourly_cost = 2.0  # $/hr per GPU (云端租赁)
total_hourly_cost = num_gpus * gpu_hourly_cost  # $4/hr

# 推理吞吐量
# 使用 vLLM, FP8 量化, batch 优化
decode_throughput = 6000  # tokens/s (batch=64, TP=2)

# 每百万 output token 成本
tokens_per_hour = decode_throughput * 3600  # 21.6M tokens/hr
cost_per_million = total_hourly_cost / (tokens_per_hour / 1e6)
# = $4.0 / 21.6 = $0.185 / M output tokens

# 加上运维、网络、存储等开销 (~30%):
total_cost_per_million = cost_per_million * 1.3
# = $0.24 / M output tokens
```

### 5.2 自部署 vs API 的盈亏平衡

```
每月调用量 vs 推荐方案:

月调用量 (M tokens)  | 推荐方案          | 理由
─────────────────────────────────────────────────
< 10M               | API (按量付费)     | 自部署的固定成本太高
10M - 100M           | API (承诺量折扣)  | 可以谈到更好价格
100M - 1B            | 取决于模型        | 开源自部署开始有优势
> 1B                 | 自部署开源模型    | 成本显著低于 API
> 10B                | 自建集群 + 开源   | 最大成本优势

关键考虑:
- API 价格持续下降, 盈亏平衡点在不断上移
- 自部署需要团队维护, 人力成本不可忽视
- API 通常提供更好的质量 (最新模型, 持续优化)
- 自部署可以做定制化优化 (微调, 自定义量化)
```

### 5.3 不同部署方式的成本对比

```
70B 级模型, 每百万 output token 成本 (2025 年中):

部署方式                    | 成本/M tokens | 说明
────────────────────────────────────────────────────
OpenAI GPT-4o              | $10.00        | 闭源 frontier
Anthropic Claude Sonnet    | $15.00        | 闭源 frontier
Google Gemini 1.5 Pro      |  $7.00        | 闭源 frontier
DeepSeek-V3 API            |  $1.10        | 开源模型, API 服务
自部署 LLaMA-3 70B (FP16)  |  $0.35        | 2×H100, vLLM
自部署 LLaMA-3 70B (FP8)   |  $0.24        | 2×H100, vLLM, FP8
自部署 Qwen-2.5 72B (INT4) |  $0.15        | 1×H100, vLLM, GPTQ
Groq (LPU, LLaMA-3 70B)   |  $0.79        | 专用硬件
Together.ai (LLaMA-3 70B)  |  $0.88        | 托管开源

注: 自部署成本不含人力运维成本 (~$0.05-0.15/M tokens)
```

## 6. 端侧推理的成本分析

### 6.1 端侧推理的经济学

```
端侧推理 (On-device Inference) 的成本结构:

成本构成:
  1. 硬件成本 (已沉没):
     - iPhone 16 Pro: A18 Pro chip, 8GB RAM → 不额外花钱
     - MacBook Pro M4: 可跑 ~30B 模型 → 不额外花钱
     - 高端 Android: Snapdragon 8 Gen 3 → 不额外花钱

  2. 电力成本:
     - 手机推理功耗: ~5-10W
     - 笔记本推理功耗: ~20-50W
     - 1 小时推理电费: ~$0.001-0.005
     → 几乎可以忽略!

  3. 推理性能 (tokens/s):
     - iPhone 16 Pro, 3B model: ~30 tok/s
     - MacBook Pro M4 Max, 14B model: ~40 tok/s
     - MacBook Pro M4 Max, 70B (INT4): ~10 tok/s

  4. 等效成本:
     - 如果按云端价格算, 端侧推理几乎是免费的
     - 但性能受限: 只能跑小模型, 速度较慢
```

### 6.2 端侧 vs 云端的权衡

```
| 维度 | 端侧推理 | 云端推理 |
|------|----------|----------|
| 成本 | ~$0 (硬件已有) | $0.10-15/M tokens |
| 模型大小 | ≤14B (手机), ≤70B (笔记本) | 无限制 |
| 质量 | 中 (受模型大小限制) | 高 (可用最大模型) |
| 延迟 | TTFT 很低 (无网络) | 受网络延迟影响 |
| 吞吐 | 10-40 tok/s | 50-200 tok/s |
| 隐私 | 数据不离设备 | 数据上传到云端 |
| 离线 | 支持 | 不支持 |
| 长上下文 | 受内存限制 (≤32K) | 支持 1M+ |
```

### 6.3 端云协同

```
端云协同的推理模式:

模式 1: 端侧 draft + 云端 verify
  小模型在本地快速生成 draft tokens
  大模型在云端验证和修正
  → 结合端侧低延迟 + 云端高质量

模式 2: 简单任务端侧, 复杂任务云端
  Router 判断任务难度:
  - 简单问答、翻译 → 端侧 3-7B 模型
  - 复杂推理、代码生成 → 云端 frontier 模型
  → 降低 70-80% 的云端调用成本

模式 3: KV Cache 流动
  端侧预计算 system prompt 的 KV Cache
  上传 KV Cache 到云端 (避免重复 prefill)
  → 需要端云模型兼容 (同架构, 同权重)
```

## 7. 推理成本的未来趋势

### 7.1 继续下降的驱动力

```
2026-2028 成本继续下降的预期因素:

1. 新硬件:
   - NVIDIA B200/B300: 预期 2-3× perf/$ vs H100
   - AMD MI350/MI400: 竞争压力压低 GPU 价格
   - 推理专用芯片 (Groq LPU, Cerebras): 特定场景 10×+ 效率

2. 模型蒸馏:
   - 更小更强的模型 (7B 模型接近今天的 70B 质量)
   - 任务特化模型 (coding agent 不需要通用能力)
   - MoE 持续进步 (更稀疏的激活)

3. 推理算法:
   - Speculative decoding 成熟化
   - KV Cache 压缩标配 (FP4/INT4)
   - 更好的 batching 和调度

4. 竞争:
   - 更多 API 提供商 → 价格战
   - 开源模型质量提升 → API 不得不降价
```

### 7.2 成本下降可能放缓的因素

```
成本下降不是无限的, 限制因素:

1. 能源成本: 电费不会大幅下降
   - GPU 功耗: 700W (B200)
   - 电费: $0.05-0.15/kWh
   - 推理最低电力成本: 有个下限

2. 冷却和基础设施: 数据中心建设成本
   - 液冷成本上升
   - 稀缺资源: 电力供应、冷却水

3. 模型质量 vs 效率的权衡:
   - 用户对质量的要求持续提高
   - Reasoning model (o1, R1) 需要更多计算
   - Agent 场景: 单次交互需要多轮推理

4. 需求增长可能超过供给:
   - AI agent 普及 → 推理需求爆炸式增长
   - 需求增长可能推高价格
```

### 7.3 长期成本预测

```
推理成本预测 (frontier model, 每百万 output tokens):

2023:   $60.00
2024:   $10.00
2025:    $3.00
2026:    $1.00  (预测)
2027:    $0.30  (预测)
2028:    $0.10  (预测)
2030:    $0.03  (乐观预测)

注意: 这是 "等效能力" 的成本下降
实际上每一代 frontier model 的能力都在提升
→ 如果保持同等能力, 成本下降更快
→ 如果追求最新能力, 成本下降较慢

类比:
  2010 年的超级计算机 ≈ 今天的手机
  2023 年的 GPT-4 ≈ 2026 年的端侧模型?
```

## 8. 成本优化的实践策略

### 8.1 立即可用的优化

```
成本优化清单 (按投入产出比排序):

ROI: ★★★★★ (几乎零成本)
1. Prompt 优化: 减少冗余 token, 可降低 20-50% 成本
2. Prompt Caching: system prompt 不变 → 90% 折扣 (Anthropic)
3. max_tokens 设置: 避免不必要的长输出
4. 模型选择: 简单任务用小模型 (4o-mini vs 4o)

ROI: ★★★★ (低投入)
5. Batch API: 非实时任务用 batch → 50% 折扣
6. 缓存响应: 相同/相似输入不重复调用
7. Streaming + 提前终止: 检测到足够信息就停止生成

ROI: ★★★ (中等投入)
8. 自部署开源模型: 大规模场景 (>1B tokens/月)
9. 量化: FP8/INT4 量化降低硬件需求
10. Fine-tuning: 更小的微调模型替代大通用模型

ROI: ★★ (高投入)
11. 分离架构: Prefill-decode disaggregation
12. 自建推理集群: 极大规模场景
13. 定制推理芯片: 特定场景的极致优化
```

### 8.2 混合部署策略

```python
# 生产环境的混合部署策略

class InferenceRouter:
    """根据任务特征路由到最优推理端点"""
    
    def route(self, request: Request) -> str:
        # 1. 判断任务复杂度
        complexity = self.estimate_complexity(request)
        
        # 2. 判断延迟要求
        is_realtime = request.requires_streaming
        
        # 3. 路由决策
        if complexity == "simple" and not is_realtime:
            return "batch_api_small_model"    # 最便宜: batch + 小模型
        elif complexity == "simple" and is_realtime:
            return "realtime_small_model"     # 便宜: 实时 + 小模型
        elif complexity == "complex" and not is_realtime:
            return "batch_api_large_model"    # 中等: batch + 大模型
        else:
            return "realtime_large_model"     # 最贵: 实时 + 大模型
    
    # 这种路由策略通常可以降低 50-70% 的总成本
    # 因为大多数请求其实不需要 frontier model
```

## 9. 小结

LLM 推理成本的下降是多种因素共同作用的结果：

1. **价格趋势**：Frontier model 价格 3 年下降约 100-300 倍，且仍在加速
2. **Flex/Batch API**：利用闲置资源和异步处理，提供 50% 成本折扣
3. **开源 vs 闭源**：大规模场景下自部署开源模型成本更低，但需要工程能力
4. **端侧推理**：硬件成本已沉没，推理近乎免费，但模型大小和质量受限
5. **端云协同**：结合端侧低延迟和云端高质量，是未来的重要方向
6. **优化策略**：从 prompt 优化到混合部署，不同投入水平都有对应的成本优化手段

对于大多数团队，建议**从 API + Prompt Caching + Batch API 开始**，随着规模增长逐步引入自部署和混合策略。追求极致成本效率的团队应关注开源模型 + 量化 + 分离架构的组合。
