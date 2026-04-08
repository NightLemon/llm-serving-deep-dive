# 监控与可观测性

> LLM 推理服务的可观测性是生产环境的基础。
> 本节详细介绍如何基于 vLLM 内置指标构建完整的监控体系。

## 1. 为什么 LLM 推理的监控与众不同

传统 Web 服务的监控围绕 QPS、延迟、错误率三大核心指标。LLM 推理服务有几个独特的监控需求：

1. **双阶段延迟**：TTFT（首 token 延迟）和 TBT（每 token 延迟）需要分别监控
2. **资源有限且昂贵**：GPU 显存中的 KV Cache 是核心资源，用尽则无法接收新请求
3. **吞吐量按 tokens 计算**：不是简单的 requests/s，而是 tokens/s
4. **动态 batching**：batch size 随时间变化，影响延迟和吞吐的权衡
5. **长尾请求**：生成 2000 tokens 的请求和生成 10 tokens 的请求差异巨大

## 2. vLLM 内置 Prometheus 指标

vLLM 从 v0.4 开始内置了丰富的 Prometheus 指标，通过 `/metrics` endpoint 暴露。

### 2.1 启用方式

```bash
# 启动 vLLM 时会自动在同一端口暴露 /metrics
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --disable-log-requests  # 减少日志噪音，但保留 metrics
```

访问 `http://localhost:8000/metrics` 即可获取 Prometheus 格式的指标。

### 2.2 核心指标详解

#### 请求级指标

| 指标名 | 类型 | 说明 | 监控意义 |
|--------|------|------|---------|
| `vllm:num_requests_running` | Gauge | 当前正在处理的请求数 | 反映系统当前并发 |
| `vllm:num_requests_waiting` | Gauge | 等待队列中的请求数 | 等待 > 0 说明系统饱和 |
| `vllm:num_requests_swapped` | Gauge | 被 swap 到 CPU 的请求数 | > 0 说明 GPU 显存不足 |
| `vllm:request_success_total` | Counter | 成功完成的请求总数 | 计算成功率 |
| `vllm:request_failure_total` | Counter | 失败的请求总数（含原因标签） | 按失败原因分类告警 |

#### 延迟指标

| 指标名 | 类型 | 说明 | 关注分位数 |
|--------|------|------|-----------|
| `vllm:e2e_request_latency_seconds` | Histogram | 端到端请求延迟 | P50, P90, P99 |
| `vllm:time_to_first_token_seconds` | Histogram | 首 token 延迟 (TTFT) | P50, P99 |
| `vllm:time_per_output_token_seconds` | Histogram | 每个输出 token 的间隔 (TBT/ITL) | P50, P99 |
| `vllm:request_queue_time_seconds` | Histogram | 请求在队列中的等待时间 | P99 |

#### 吞吐量指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm:avg_prompt_throughput_toks_per_s` | Gauge | 最近窗口的 prefill 吞吐量 |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | 最近窗口的 decode 吞吐量 |
| `vllm:prompt_tokens_total` | Counter | 处理的 prompt tokens 总数 |
| `vllm:generation_tokens_total` | Counter | 生成的 output tokens 总数 |

#### 资源指标

| 指标名 | 类型 | 说明 | 告警阈值 |
|--------|------|------|---------|
| `vllm:gpu_cache_usage_perc` | Gauge | GPU KV Cache 使用率 | > 95% 告警 |
| `vllm:cpu_cache_usage_perc` | Gauge | CPU KV Cache 使用率 (swap) | > 80% 告警 |
| `vllm:num_preemptions_total` | Counter | preemption 发生次数 | 速率 > 0 关注 |
| `vllm:gpu_prefix_cache_hit_rate` | Gauge | Prefix cache 命中率 | 越高越好 |

#### 调度指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm:num_running_requests_per_step` | Histogram | 每个调度步骤的 running 请求数 |
| `vllm:num_waiting_requests_per_step` | Histogram | 每个调度步骤的 waiting 请求数 |
| `vllm:scheduler_running_requests` | Gauge | 调度器中 running 状态的请求数 |

### 2.3 自定义指标

vLLM 支持通过插件或修改代码添加自定义指标。以下是添加"按模型分类的请求计数"的示例：

```python
# 在 vLLM 的 metrics 模块中添加自定义指标
from prometheus_client import Counter, Histogram, Gauge

# 自定义：按 API key 分类的请求计数
requests_by_api_key = Counter(
    "vllm_custom:requests_by_api_key_total",
    "Total requests by API key",
    labelnames=["api_key_hash"]  # 注意不要暴露真实 key
)

# 自定义：请求的 prompt 长度分布
prompt_length_histogram = Histogram(
    "vllm_custom:prompt_length_tokens",
    "Distribution of prompt lengths in tokens",
    buckets=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
)

# 自定义：活跃的 LoRA adapter 数量
active_lora_adapters = Gauge(
    "vllm_custom:active_lora_adapters",
    "Number of currently loaded LoRA adapters"
)
```

## 3. Grafana Dashboard 设计

### 3.1 Dashboard 布局

推荐将 dashboard 分为 5 个区域（row）：

```
┌─────────────────────────────────────────────────────┐
│ Row 1: 概览 (Overview)                               │
│  [请求速率] [成功率] [活跃请求] [等待队列]              │
├─────────────────────────────────────────────────────┤
│ Row 2: 延迟 (Latency)                                │
│  [TTFT P50/P90/P99] [TBT P50/P90/P99] [E2E 延迟]    │
├─────────────────────────────────────────────────────┤
│ Row 3: 吞吐量 (Throughput)                            │
│  [Prefill tokens/s] [Decode tokens/s] [总 tokens/s]  │
├─────────────────────────────────────────────────────┤
│ Row 4: 资源 (Resources)                              │
│  [GPU Cache %] [CPU Cache %] [Prefix Cache Hit Rate] │
├─────────────────────────────────────────────────────┤
│ Row 5: 调度 (Scheduling)                             │
│  [Batch Size] [Preemptions] [Queue Time]             │
└─────────────────────────────────────────────────────┘
```

### 3.2 关键面板的 PromQL 查询

**请求速率：**

```promql
# 每秒请求数 (成功)
rate(vllm:request_success_total[5m])

# 每秒请求数 (失败)
rate(vllm:request_failure_total[5m])

# 请求成功率
rate(vllm:request_success_total[5m]) / 
(rate(vllm:request_success_total[5m]) + rate(vllm:request_failure_total[5m]))
```

**TTFT 分位数：**

```promql
# TTFT P50
histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# TTFT P90
histogram_quantile(0.90, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# TTFT P99
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))
```

**TBT (Inter-Token Latency) 分位数：**

```promql
# TBT P50
histogram_quantile(0.50, rate(vllm:time_per_output_token_seconds_bucket[5m]))

# TBT P99
histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]))
```

**吞吐量：**

```promql
# Prefill 吞吐量
vllm:avg_prompt_throughput_toks_per_s

# Decode 吞吐量  
vllm:avg_generation_throughput_toks_per_s

# 也可以用 Counter 计算精确速率
rate(vllm:prompt_tokens_total[5m])
rate(vllm:generation_tokens_total[5m])
```

**KV Cache 使用率：**

```promql
# GPU KV Cache 使用率 (按实例)
vllm:gpu_cache_usage_perc{instance=~".*"}

# 所有实例的平均值
avg(vllm:gpu_cache_usage_perc)

# 最高使用率的实例（用于告警）
max(vllm:gpu_cache_usage_perc)
```

**Prefix Cache 命中率：**

```promql
# 命中率
vllm:gpu_prefix_cache_hit_rate

# 如果没有直接指标，用 Counter 计算
rate(vllm:prefix_cache_hit_total[5m]) / 
(rate(vllm:prefix_cache_hit_total[5m]) + rate(vllm:prefix_cache_miss_total[5m]))
```

### 3.3 Grafana Dashboard JSON 片段

以下是一个 TTFT 面板的完整 JSON 配置示例：

```json
{
  "panels": [
    {
      "title": "Time to First Token (TTFT)",
      "type": "timeseries",
      "datasource": "Prometheus",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
      "fieldConfig": {
        "defaults": {
          "unit": "s",
          "custom": {
            "drawStyle": "line",
            "lineWidth": 2
          }
        }
      },
      "targets": [
        {
          "expr": "histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket{instance=~\"$instance\"}[5m]))",
          "legendFormat": "P50 - {{instance}}"
        },
        {
          "expr": "histogram_quantile(0.90, rate(vllm:time_to_first_token_seconds_bucket{instance=~\"$instance\"}[5m]))",
          "legendFormat": "P90 - {{instance}}"
        },
        {
          "expr": "histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket{instance=~\"$instance\"}[5m]))",
          "legendFormat": "P99 - {{instance}}"
        }
      ]
    },
    {
      "title": "GPU KV Cache Usage",
      "type": "gauge",
      "datasource": "Prometheus",
      "gridPos": { "h": 8, "w": 6, "x": 0, "y": 24 },
      "fieldConfig": {
        "defaults": {
          "unit": "percentunit",
          "min": 0,
          "max": 1,
          "thresholds": {
            "steps": [
              { "color": "green", "value": 0 },
              { "color": "yellow", "value": 0.8 },
              { "color": "red", "value": 0.95 }
            ]
          }
        }
      },
      "targets": [
        {
          "expr": "vllm:gpu_cache_usage_perc{instance=~\"$instance\"}",
          "legendFormat": "{{instance}}"
        }
      ]
    },
    {
      "title": "Throughput (tokens/s)",
      "type": "timeseries",
      "datasource": "Prometheus",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 16 },
      "fieldConfig": {
        "defaults": {
          "unit": "tokens/s",
          "custom": {
            "drawStyle": "line",
            "fillOpacity": 20
          }
        }
      },
      "targets": [
        {
          "expr": "vllm:avg_prompt_throughput_toks_per_s{instance=~\"$instance\"}",
          "legendFormat": "Prefill - {{instance}}"
        },
        {
          "expr": "vllm:avg_generation_throughput_toks_per_s{instance=~\"$instance\"}",
          "legendFormat": "Decode - {{instance}}"
        }
      ]
    }
  ],
  "templating": {
    "list": [
      {
        "name": "instance",
        "type": "query",
        "query": "label_values(vllm:num_requests_running, instance)",
        "datasource": "Prometheus",
        "multi": true,
        "includeAll": true
      }
    ]
  }
}
```

## 4. Prometheus + Docker Compose 部署

### 4.1 完整的 docker-compose 示例

```yaml
# docker-compose.monitoring.yaml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:v2.50.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=30d"
    
  grafana:
    image: grafana/grafana:10.3.0
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
      - ./grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/vllm-overview.json
  
  alertmanager:
    image: prom/alertmanager:v0.27.0
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

volumes:
  prometheus_data:
  grafana_data:
```

### 4.2 Prometheus 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: "vllm"
    scrape_interval: 5s  # LLM 推理指标变化快，用更短的采集间隔
    static_configs:
      - targets:
          - "vllm-replica-0:8000"
          - "vllm-replica-1:8000"
          - "vllm-replica-2:8000"
        labels:
          cluster: "production"
    # 如果使用 Kubernetes
    # kubernetes_sd_configs:
    #   - role: pod
    #     selectors:
    #       - role: pod
    #         label: "app=vllm"
```

## 5. 告警规则

### 5.1 推荐的告警规则

```yaml
# alert_rules.yml
groups:
  - name: vllm_alerts
    rules:
      # KV Cache 使用率过高
      - alert: KVCacheUsageHigh
        expr: vllm:gpu_cache_usage_perc > 0.95
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "GPU KV Cache usage above 95% on {{ $labels.instance }}"
          description: >
            KV Cache 使用率超过 95%，可能触发 preemption 或拒绝新请求。
            当前值: {{ $value | humanizePercentage }}
          runbook: "考虑扩容、启用 KV Cache 量化或减小 max_model_len"

      # KV Cache 满载 - 严重
      - alert: KVCacheFull
        expr: vllm:gpu_cache_usage_perc > 0.99
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "GPU KV Cache 接近满载 on {{ $labels.instance }}"
          description: "立即扩容或降低并发。当前值: {{ $value | humanizePercentage }}"

      # 等待队列过长
      - alert: RequestQueueTooLong
        expr: vllm:num_requests_waiting > 20
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "请求等待队列超过 20 on {{ $labels.instance }}"
          description: >
            队列长度: {{ $value }}。
            持续排队说明当前实例处理能力不足，需要扩容。

      # TTFT 超过 SLA
      - alert: TTFTTooHigh
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 TTFT 超过 5 秒"
          description: >
            首 token 延迟 P99 = {{ $value | humanizeDuration }}。
            可能原因：长 prompt 阻塞、KV Cache 不足、prefill 资源不够。

      # TBT 抖动
      - alert: TBTUnstable
        expr: histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 TBT 超过 200ms"
          description: >
            用户感知到明显的生成卡顿。
            可能原因：preemption、chunked prefill 干扰、GPU 热节流。

      # Preemption 频繁发生
      - alert: FrequentPreemptions
        expr: rate(vllm:num_preemptions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Preemption 频繁发生"
          description: >
            Preemption 速率: {{ $value }}/s。
            请求被抢占会导致延迟大幅增加，需要增加 GPU 显存或减少并发。

      # 实例不健康
      - alert: VLLMInstanceDown
        expr: up{job="vllm"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "vLLM 实例 {{ $labels.instance }} 不可达"
          description: "实例可能崩溃或网络不通，需要立即检查。"

      # 错误率突增
      - alert: HighErrorRate
        expr: >
          rate(vllm:request_failure_total[5m]) / 
          (rate(vllm:request_success_total[5m]) + rate(vllm:request_failure_total[5m])) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "请求错误率超过 5%"
          description: "错误率: {{ $value | humanizePercentage }}。检查 OOM、模型加载错误等。"

      # Prefix cache 命中率下降
      - alert: LowPrefixCacheHitRate
        expr: vllm:gpu_prefix_cache_hit_rate < 0.3
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "Prefix cache 命中率低于 30%"
          description: >
            命中率: {{ $value | humanizePercentage }}。
            检查路由策略是否正确配置 cache-aware routing。
```

## 6. 分布式推理的监控挑战

### 6.1 多节点指标聚合

使用 Tensor Parallelism (TP) 或 Pipeline Parallelism (PP) 时，一个逻辑推理实例由多个 GPU 组成。监控需要注意：

```
逻辑实例 "model-70b-instance-0":
  ├── GPU 0 (TP rank 0) → node-0:8000/metrics
  ├── GPU 1 (TP rank 1) → 无独立 metrics endpoint
  ├── GPU 2 (TP rank 2) → 无独立 metrics endpoint
  └── GPU 3 (TP rank 3) → 无独立 metrics endpoint
```

**关键问题：**
- vLLM 只在 rank 0 暴露 metrics endpoint
- GPU 级别的指标（显存、利用率、温度）需要通过 DCGM Exporter 单独采集
- 通信指标（NCCL AllReduce 延迟）默认不暴露

### 6.2 DCGM Exporter 集成

```yaml
# DCGM Exporter 用于采集 GPU 硬件指标
# docker-compose 中添加
dcgm-exporter:
  image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [gpu]
  ports:
    - "9400:9400"
```

DCGM 提供的关键指标：

| 指标 | 说明 | 与 LLM 推理的关系 |
|------|------|------------------|
| `DCGM_FI_DEV_GPU_UTIL` | GPU SM 利用率 | 低 → batch 太小或 memory-bound |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | 显存带宽利用率 | decode 阶段应接近上限 |
| `DCGM_FI_DEV_FB_USED` | 已用显存 (bytes) | 与 KV Cache 使用关联 |
| `DCGM_FI_DEV_GPU_TEMP` | GPU 温度 | > 80°C 会触发降频 |
| `DCGM_FI_DEV_POWER_USAGE` | GPU 功耗 | 成本计算 |
| `DCGM_FI_DEV_NVLINK_BANDWIDTH_TX` | NVLink 发送带宽 | TP 通信监控 |

### 6.3 端到端追踪

对于需要跨多个服务的追踪（例如 Router → vLLM → Postprocessing），可以集成 OpenTelemetry：

```python
# 在 router 中注入 trace context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# 配置 tracer
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-router")

async def route_request(request):
    with tracer.start_as_current_span("llm_inference") as span:
        span.set_attribute("llm.model", request.model)
        span.set_attribute("llm.prompt_tokens", len(request.prompt_tokens))
        
        # 路由决策
        with tracer.start_as_current_span("route_decision"):
            replica = router.select(request)
            span.set_attribute("llm.replica", replica)
        
        # 转发请求
        with tracer.start_as_current_span("inference"):
            response = await forward_to_replica(replica, request)
            span.set_attribute("llm.completion_tokens", response.completion_tokens)
        
        return response
```

## 7. 监控最佳实践

### 7.1 SLI/SLO 定义

建议为 LLM 推理服务定义以下 SLI/SLO：

| SLI | SLO 目标 | 测量方式 |
|-----|---------|---------|
| 可用性 | 99.9% | `up{job="vllm"} == 1` |
| TTFT P99 | < 3s（短 prompt）| `histogram_quantile(0.99, ...)` |
| TBT P99 | < 100ms | `histogram_quantile(0.99, ...)` |
| 请求成功率 | > 99.5% | success / (success + failure) |
| 吞吐量 | > X tokens/s/GPU | `avg_generation_throughput_toks_per_s` |

### 7.2 Capacity Planning

基于历史监控数据做容量规划：

```promql
# 预测未来 7 天的 GPU Cache 使用趋势
predict_linear(vllm:gpu_cache_usage_perc[7d], 7*24*3600)

# 计算每个 GPU 的有效吞吐量（用于决定需要多少 GPU）
avg_over_time(vllm:avg_generation_throughput_toks_per_s[1h])
```

### 7.3 Dashboard 使用技巧

1. **设置合理的时间窗口**：默认看最近 1 小时，调查问题时切换到 5 分钟
2. **使用 Grafana Variables**：按 instance、cluster、model 过滤
3. **关联 GPU 指标**：将 vLLM 指标和 DCGM 指标放在同一 dashboard 中交叉分析
4. **标注事件**：在 Grafana 中标注部署、扩容、配置变更等事件，方便排查

## 8. 总结

构建 LLM 推理服务的监控体系，需要关注三个层面：

| 层面 | 关注点 | 工具 |
|------|--------|------|
| **应用层** | TTFT、TBT、吞吐量、队列深度 | vLLM Prometheus metrics |
| **资源层** | KV Cache、GPU 利用率、显存 | vLLM metrics + DCGM |
| **基础设施层** | 节点健康、网络、存储 | Node Exporter + DCGM |

**核心原则：** 监控的目的不仅是发现问题，更是为优化提供数据支撑。每一个告警都应该有明确的 runbook（处置手册），告诉 oncall 工程师应该做什么。

---

> **延伸阅读：**
> - [vLLM Metrics 文档](https://docs.vllm.ai/en/latest/serving/metrics.html)
> - [DCGM Exporter GitHub](https://github.com/NVIDIA/dcgm-exporter)
> - Grafana 官方 LLM 推理 Dashboard 模板
