# 故障恢复与高可用

> GPU 硬件昂贵且故障率不低，网络通信复杂，LLM 推理服务的高可用设计至关重要。
> 本节系统介绍故障场景、检测机制和恢复策略。

## 1. 故障场景分析

### 1.1 故障分类

LLM 推理服务面临的故障可以分为以下几类：

```
┌────────────────────────────────────────────────────────┐
│                   故障场景全景                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  硬件故障                                              │
│  ├── GPU 故障: 单卡 hang、ECC 错误、GPU 坏死            │
│  ├── 显存故障: bit flip、显存条损坏                     │
│  ├── NVLink 故障: 链路降级或断开                        │
│  ├── 网络故障: InfiniBand/RoCE 中断                    │
│  └── 主机故障: 电源、主板、CPU 故障                     │
│                                                        │
│  软件故障                                              │
│  ├── OOM: KV Cache 分配失败                            │
│  ├── CUDA Error: kernel 执行失败                       │
│  ├── NCCL Timeout: 分布式通信超时                      │
│  ├── 死锁: 调度器逻辑错误                              │
│  └── 内存泄漏: 长时间运行后 GPU 显存逐渐增长            │
│                                                        │
│  负载故障                                              │
│  ├── 过载: 请求量超过处理能力                           │
│  ├── 恶意请求: 超长 prompt 或输出                      │
│  └── 慢请求: 单个请求占用过多资源                       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 1.2 故障概率

根据大规模 GPU 集群的公开数据（Meta、Google 等的论文）：

| 故障类型 | 频率 (每千 GPU·天) | 影响范围 | 恢复时间 |
|----------|-------------------|---------|---------|
| GPU 硬件故障 | 0.1-0.5 | 单卡/单节点 | 分钟级（替换） |
| NCCL 通信超时 | 1-5 | TP/PP 组 | 秒级（重试） |
| OOM | 5-20 | 单实例 | 秒级（自动恢复） |
| CUDA Error | 0.5-2 | 单实例 | 秒级（重启） |
| 网络中断 | 0.1-1 | 节点组 | 分钟级 |
| 主机宕机 | 0.01-0.1 | 单节点 | 分钟级 |

## 2. Health Check 与故障检测

### 2.1 多层 Health Check

```python
from fastapi import FastAPI, Response
import asyncio
import time

app = FastAPI()

class HealthChecker:
    def __init__(self, vllm_engine):
        self.engine = vllm_engine
        self.last_successful_inference = time.time()
        self.consecutive_failures = 0
        self.max_failures = 3
    
    async def liveness_check(self) -> dict:
        """
        存活检查: 进程是否还活着？
        - Kubernetes liveness probe 使用
        - 失败 → 重启 Pod
        """
        return {"status": "alive", "pid": os.getpid()}
    
    async def readiness_check(self) -> dict:
        """
        就绪检查: 能否接收新请求？
        - Kubernetes readiness probe 使用
        - 失败 → 从 Service 中摘除，不再接收流量
        """
        checks = {}
        
        # 1. 模型是否加载完成
        checks["model_loaded"] = self.engine.is_model_loaded()
        
        # 2. GPU 是否可用
        try:
            import torch
            torch.cuda.synchronize()
            checks["gpu_available"] = True
        except Exception as e:
            checks["gpu_available"] = False
            checks["gpu_error"] = str(e)
        
        # 3. KV Cache 是否有余量 (使用率 > 99% 不再接受新请求)
        cache_usage = self.engine.get_cache_usage()
        checks["cache_available"] = cache_usage < 0.99
        checks["cache_usage"] = cache_usage
        
        # 4. 最近是否成功处理过请求
        time_since_last = time.time() - self.last_successful_inference
        checks["recent_success"] = time_since_last < 300  # 5 分钟内有成功推理
        
        is_ready = all([
            checks["model_loaded"],
            checks["gpu_available"],
            checks["cache_available"],
        ])
        
        return {"ready": is_ready, "checks": checks}
    
    async def deep_health_check(self) -> dict:
        """
        深度健康检查: 能否实际完成推理？
        - 定期执行 (每 30 秒)
        - 发送一个简单的推理请求验证整个链路
        """
        try:
            start = time.time()
            # 发送一个简单的推理请求
            result = await asyncio.wait_for(
                self.engine.generate("Hello", max_tokens=5),
                timeout=30.0  # 30 秒超时
            )
            latency = time.time() - start
            
            self.last_successful_inference = time.time()
            self.consecutive_failures = 0
            
            return {
                "healthy": True,
                "inference_latency_ms": latency * 1000,
                "generated_tokens": len(result.outputs[0].token_ids),
            }
        except asyncio.TimeoutError:
            self.consecutive_failures += 1
            return {
                "healthy": False,
                "error": "inference_timeout",
                "consecutive_failures": self.consecutive_failures,
            }
        except Exception as e:
            self.consecutive_failures += 1
            return {
                "healthy": False,
                "error": str(e),
                "consecutive_failures": self.consecutive_failures,
            }

# Kubernetes Probe 配置
"""
apiVersion: v1
kind: Pod
spec:
  containers:
    - name: vllm
      livenessProbe:
        httpGet:
          path: /health/live
          port: 8000
        initialDelaySeconds: 120   # 模型加载需要时间
        periodSeconds: 10
        failureThreshold: 3        # 连续 3 次失败才重启
        
      readinessProbe:
        httpGet:
          path: /health/ready
          port: 8000
        initialDelaySeconds: 120
        periodSeconds: 5
        failureThreshold: 2
        
      startupProbe:
        httpGet:
          path: /health/ready
          port: 8000
        initialDelaySeconds: 30
        periodSeconds: 10
        failureThreshold: 30       # 最多等待 5 分钟启动
"""
```

### 2.2 GPU 级别的健康监控

```python
import subprocess
import json

def check_gpu_health():
    """使用 nvidia-smi 检查 GPU 健康状态"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,gpu_uuid,temperature.gpu,"
         "utilization.gpu,memory.used,memory.total,ecc.errors.corrected.volatile.total,"
         "ecc.errors.uncorrected.volatile.total,power.draw",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    
    gpu_status = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        status = {
            "index": int(parts[0]),
            "uuid": parts[1],
            "temperature": int(parts[2]),
            "utilization": int(parts[3]),
            "memory_used_mb": int(parts[4]),
            "memory_total_mb": int(parts[5]),
            "ecc_corrected": int(parts[6]) if parts[6] != "N/A" else 0,
            "ecc_uncorrected": int(parts[7]) if parts[7] != "N/A" else 0,
            "power_watts": float(parts[8]),
        }
        
        # 异常检测
        issues = []
        if status["temperature"] > 85:
            issues.append("HIGH_TEMPERATURE")
        if status["ecc_uncorrected"] > 0:
            issues.append("UNCORRECTED_ECC_ERRORS")
        if status["utilization"] == 0 and status["memory_used_mb"] > 1000:
            issues.append("GPU_HUNG")  # 有内存占用但利用率为 0
        
        status["issues"] = issues
        status["healthy"] = len(issues) == 0
        gpu_status.append(status)
    
    return gpu_status
```

## 3. 自动恢复策略

### 3.1 OOM 恢复

OOM 是最常见的运行时故障。vLLM 有内置的 OOM 处理机制：

```
OOM 恢复流程:
1. KV Cache 分配失败
2. 触发 preemption: 按优先级驱逐请求
   - recompute 模式: 释放 KV Cache, 之后重新 prefill
   - swap 模式: 将 KV Cache swap 到 CPU 内存
3. 如果 preemption 后仍然不够:
   - 拒绝新请求 (返回 503)
   - 等待现有请求完成释放资源
4. 如果所有请求都被驱逐仍然 OOM:
   - CUDA OOM → 需要重启进程
```

**预防 OOM 的配置：**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096
```

### 3.2 进程级自动重启

使用 supervisor 或 systemd 管理 vLLM 进程：

```ini
# /etc/supervisor/conf.d/vllm.conf
[program:vllm]
command=python -m vllm.entrypoints.openai.api_server 
    --model meta-llama/Llama-3.1-70B-Instruct
    --port 8000
directory=/opt/vllm
autostart=true
autorestart=true
startretries=5
startsecs=120          ; 120 秒内不崩溃视为启动成功
stopwaitsecs=60        ; 给 60 秒做 graceful shutdown
redirect_stderr=true
stdout_logfile=/var/log/vllm/vllm.log
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=5
environment=CUDA_VISIBLE_DEVICES="0,1,2,3"
```

```yaml
# systemd service file
# /etc/systemd/system/vllm.service
[Unit]
Description=vLLM Inference Server
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=vllm
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3
ExecStart=/opt/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --port 8000
Restart=on-failure
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=600
ExecStop=/bin/kill -SIGTERM $MAINPID
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

### 3.3 请求重试与幂等性

```python
import httpx
import asyncio
from typing import Optional

class ResilientLLMClient:
    def __init__(
        self,
        endpoints: list[str],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ):
        self.endpoints = endpoints
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.endpoint_health = {ep: True for ep in endpoints}
    
    def _get_healthy_endpoint(self, exclude: Optional[str] = None) -> str:
        """获取一个健康的 endpoint，排除指定的"""
        healthy = [
            ep for ep in self.endpoints 
            if self.endpoint_health[ep] and ep != exclude
        ]
        if not healthy:
            # 所有节点都不健康，重置并返回任意一个
            for ep in self.endpoints:
                self.endpoint_health[ep] = True
            healthy = [ep for ep in self.endpoints if ep != exclude]
        
        return healthy[0] if healthy else self.endpoints[0]
    
    async def chat_completion(self, messages: list, **kwargs) -> dict:
        """
        带重试的 chat completion 请求
        
        幂等性说明:
        - LLM 推理本身是幂等的(相同输入+相同 seed → 相同输出)
        - 但由于随机采样，默认行为下重试可能得到不同结果
        - 如果需要严格幂等，需要固定 seed
        """
        last_error = None
        tried_endpoints = set()
        
        for attempt in range(self.max_retries):
            # 选择 endpoint (避免重试到同一个失败节点)
            endpoint = self._get_healthy_endpoint(
                exclude=list(tried_endpoints)[-1] if tried_endpoints else None
            )
            tried_endpoints.add(endpoint)
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{endpoint}/v1/chat/completions",
                        json={"messages": messages, **kwargs},
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 503:
                        # 服务不可用，标记不健康
                        self.endpoint_health[endpoint] = False
                        last_error = f"503 from {endpoint}"
                    elif response.status_code == 429:
                        # 限流，等待后重试同一 endpoint
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        last_error = f"429 rate limited from {endpoint}"
                    else:
                        last_error = f"{response.status_code}: {response.text}"
                        
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                self.endpoint_health[endpoint] = False
                last_error = f"Connection error to {endpoint}: {e}"
            
            # 重试前等待 (指数退避)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
        
        raise Exception(f"All retries failed. Last error: {last_error}")
```

## 4. Graceful Degradation

### 4.1 降级策略

当系统过载或部分故障时，不应直接拒绝所有请求，而是逐级降级：

```python
class DegradationManager:
    """
    降级等级:
    Level 0: 正常运行
    Level 1: 限制最大输出长度
    Level 2: 限制并发 + 降低优先级请求
    Level 3: 仅处理高优先级请求
    Level 4: 熔断 (返回预设回复)
    """
    
    def __init__(self):
        self.current_level = 0
        self.level_configs = {
            0: {"max_tokens": 4096, "max_concurrent": 128, "accept_low_priority": True},
            1: {"max_tokens": 1024, "max_concurrent": 128, "accept_low_priority": True},
            2: {"max_tokens": 512,  "max_concurrent": 64,  "accept_low_priority": False},
            3: {"max_tokens": 256,  "max_concurrent": 32,  "accept_low_priority": False},
            4: {"max_tokens": 0,    "max_concurrent": 0,   "accept_low_priority": False},
        }
    
    def evaluate_level(self, metrics: dict) -> int:
        """根据系统指标自动判断降级等级"""
        cache_usage = metrics["gpu_cache_usage_perc"]
        queue_length = metrics["num_requests_waiting"]
        error_rate = metrics.get("error_rate", 0)
        
        if error_rate > 0.1:  # 错误率 > 10%
            return 4
        elif cache_usage > 0.99 or queue_length > 100:
            return 3
        elif cache_usage > 0.95 or queue_length > 50:
            return 2
        elif cache_usage > 0.90 or queue_length > 20:
            return 1
        else:
            return 0
    
    def should_accept(self, request, metrics: dict) -> tuple[bool, dict]:
        """判断是否接受请求，以及应用的限制"""
        level = self.evaluate_level(metrics)
        config = self.level_configs[level]
        
        if level == 4:
            return False, {"reason": "circuit_breaker", "fallback": True}
        
        if not config["accept_low_priority"] and request.priority == "low":
            return False, {"reason": "low_priority_rejected"}
        
        # 应用限制
        adjustments = {}
        original_max_tokens = request.max_tokens or 4096
        if original_max_tokens > config["max_tokens"]:
            adjustments["max_tokens"] = config["max_tokens"]
        
        return True, adjustments

    def get_fallback_response(self, request) -> dict:
        """熔断时返回预设回复"""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "服务当前繁忙，请稍后重试。"
                },
                "finish_reason": "degraded"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x-degradation-level": 4
        }
```

### 4.2 请求优先级

```python
from enum import IntEnum

class RequestPriority(IntEnum):
    CRITICAL = 0    # 付费用户的实时请求
    HIGH = 1        # 付费用户的批量请求
    NORMAL = 2      # 免费用户的实时请求
    LOW = 3         # 后台任务（摘要、分类等）
    BEST_EFFORT = 4 # 可丢弃的请求

class PriorityScheduler:
    """优先级调度：高优先级请求优先处理"""
    
    def __init__(self, max_concurrent: int = 128):
        self.max_concurrent = max_concurrent
        self.running = 0
        # 每个优先级的预留容量
        self.priority_quotas = {
            RequestPriority.CRITICAL: 0.4,    # 40% 预留给关键请求
            RequestPriority.HIGH: 0.25,
            RequestPriority.NORMAL: 0.25,
            RequestPriority.LOW: 0.1,
            RequestPriority.BEST_EFFORT: 0.0,  # 没有保证
        }
    
    def can_accept(self, priority: RequestPriority) -> bool:
        """判断是否有容量接受该优先级的请求"""
        available = self.max_concurrent - self.running
        
        # 高优先级请求可以使用所有空闲容量
        if priority <= RequestPriority.HIGH:
            return available > 0
        
        # 低优先级请求只能使用未被预留的容量
        reserved = sum(
            self.max_concurrent * quota
            for p, quota in self.priority_quotas.items()
            if p < priority
        )
        return self.running < (self.max_concurrent - reserved)
```

## 5. Sleep Mode 与 Pause/Resume

### 5.1 Sleep Mode

vLLM 的 Sleep Mode 允许在空闲时释放 GPU 显存，需要时重新加载：

```bash
# 启用 sleep mode
# 注意: 这是 vLLM 较新版本的特性
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --enable-sleep-mode \
    --sleep-timeout 300   # 300 秒无请求后进入 sleep
```

**Sleep Mode 工作流程：**

```
正常运行 ──(idle 5 min)──→ Sleep 模式
                             │
                             ├── 释放 KV Cache 显存
                             ├── 模型权重保留在 GPU
                             └── 对外标记为 not-ready
                             
Sleep 模式 ──(新请求到达)──→ 唤醒
                             │
                             ├── 重新分配 KV Cache 显存
                             ├── 重建 CUDA graphs (如需)
                             ├── 标记为 ready
                             └── 处理请求（首次稍慢）

预计唤醒延迟: 数秒 (vs 重启的数分钟)
```

### 5.2 Pause/Resume

对于需要临时暂停接收请求的场景（例如模型更新、配置变更）：

```python
# 通过 API 暂停/恢复
import httpx

# 暂停: 停止接收新请求，等待当前请求完成
async def pause_server(endpoint: str):
    resp = await httpx.post(f"{endpoint}/pause")
    # 此后 readiness probe 返回 false
    # 但不会中断正在处理的请求
    
# 等待所有请求完成 (drain)
async def wait_drain(endpoint: str, timeout: int = 120):
    for _ in range(timeout):
        resp = await httpx.get(f"{endpoint}/health/ready")
        data = resp.json()
        if data["checks"]["active_requests"] == 0:
            return True
        await asyncio.sleep(1)
    return False

# 恢复
async def resume_server(endpoint: str):
    resp = await httpx.post(f"{endpoint}/resume")
    # readiness probe 恢复返回 true
```

## 6. 多副本部署策略

### 6.1 部署拓扑

```
方案 1: 对称多副本 (推荐大多数场景)
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Replica 0│  │ Replica 1│  │ Replica 2│
│ GPU 0,1  │  │ GPU 2,3  │  │ GPU 4,5  │
│ TP=2     │  │ TP=2     │  │ TP=2     │
└──────────┘  └──────────┘  └──────────┘
     ↑              ↑              ↑
     └──────── Load Balancer ──────┘

方案 2: 主备模式 (高可用优先)
┌──────────┐  ┌──────────┐
│ Primary  │  │ Standby  │
│ GPU 0-3  │  │ GPU 4-7  │
│ TP=4     │  │ TP=4     │
│ 活跃     │  │ 热备     │
└──────────┘  └──────────┘
     ↑              │
     └── 故障切换 ──┘

方案 3: 异构副本 (不同优先级)
┌──────────────┐  ┌──────────┐
│ H100 Replicas│  │ A100 Rep │
│ 低延迟优先   │  │ 高吞吐   │
│ 实时请求     │  │ 批量请求 │
└──────────────┘  └──────────┘
```

### 6.2 跨可用区部署

```yaml
# Kubernetes 跨 AZ 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
spec:
  replicas: 3
  template:
    spec:
      affinity:
        # 确保 replicas 分散在不同可用区
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values: ["vllm"]
                topologyKey: topology.kubernetes.io/zone
      
      # 确保调度到有 GPU 的节点
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-H100-80GB-HBM3"
      
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: 2   # 每个 Pod 使用 2 GPU (TP=2)
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
            failureThreshold: 3
```

## 7. 故障演练

### 7.1 Chaos Engineering

定期进行故障演练，验证高可用机制：

```python
# 故障注入脚本
import random
import subprocess
import asyncio

class ChaosMonkey:
    """LLM 推理服务故障注入工具"""
    
    def __init__(self, replicas: list[str]):
        self.replicas = replicas
    
    async def inject_gpu_hang(self, replica: str):
        """模拟 GPU hang (通过发送一个无限循环的 CUDA kernel)"""
        print(f"[CHAOS] Injecting GPU hang on {replica}")
        # 实际实现需要在目标节点上执行
        # 这里用 kill 进程来模拟
        subprocess.run(["ssh", replica, "kill -STOP $(pgrep -f vllm)"])
        await asyncio.sleep(60)
        subprocess.run(["ssh", replica, "kill -CONT $(pgrep -f vllm)"])
    
    async def inject_network_partition(self, replica: str, duration: int = 30):
        """模拟网络分区"""
        print(f"[CHAOS] Network partition on {replica} for {duration}s")
        # 使用 iptables 阻断流量
        subprocess.run(["ssh", replica, 
            f"iptables -A INPUT -p tcp --dport 8000 -j DROP"])
        await asyncio.sleep(duration)
        subprocess.run(["ssh", replica, 
            f"iptables -D INPUT -p tcp --dport 8000 -j DROP"])
    
    async def inject_oom(self, replica: str):
        """模拟 OOM (发送大量超长请求)"""
        print(f"[CHAOS] Injecting OOM on {replica}")
        tasks = []
        for _ in range(100):
            tasks.append(self._send_long_request(replica))
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_long_request(self, endpoint: str):
        """发送超长请求消耗 KV Cache"""
        async with httpx.AsyncClient(timeout=300) as client:
            await client.post(f"{endpoint}/v1/chat/completions", json={
                "model": "default",
                "messages": [{"role": "user", "content": "x " * 10000}],
                "max_tokens": 4096,
            })

# 演练检查清单
"""
□ 杀死一个 replica → 其他 replica 接管流量？
□ GPU hang → health check 检测到？自动重启？
□ 网络分区 → 请求重试到其他 replica？
□ OOM → 自动恢复？是否丢失所有请求？
□ 全部 replica 宕机 → 是否返回合理错误？
□ 恢复后 → 自动重新加入集群？
"""
```

## 8. 总结

LLM 推理服务的高可用设计要点：

| 层面 | 策略 | 实现工具 |
|------|------|---------|
| **检测** | 多层 Health Check (liveness/readiness/deep) | Kubernetes probes + 自定义检查 |
| **预防** | 资源限制、过载保护、优先级调度 | vLLM 配置 + 自定义中间件 |
| **恢复** | 自动重启、请求重试、failover | systemd/supervisor + 客户端重试 |
| **降级** | 分级降级、熔断、fallback | 自定义 DegradationManager |
| **冗余** | 多副本、跨 AZ、主备 | Kubernetes Deployment + Service |
| **验证** | Chaos Engineering、故障演练 | 定期执行故障注入 |

**核心原则：** 对 GPU 密集型服务，恢复时间(MTTR) 比故障间隔(MTBF) 更重要。模型加载通常需要数分钟，因此预防 OOM 和保持进程存活比快速重启更关键。

---

> **延伸阅读：**
> - [vLLM Sleep Mode 文档](https://docs.vllm.ai/en/latest/)
> - Meta "Reliability at Scale" 论文（GPU 集群故障分析）
> - Google "ML Infra Reliability" 最佳实践
