# 模型更新与灰度发布

> 在不中断服务的前提下更新模型是生产环境的核心挑战。
> 本节介绍 LLM 推理服务的模型更新策略、KV Cache 兼容性问题和零停机发布流程。

## 1. 为什么模型更新特别困难

LLM 推理服务的模型更新比传统 Web 服务复杂得多，原因有三：

### 1.1 启动时间长

```
传统 Web 服务:
  启动 → 就绪    约 1-10 秒

LLM 推理服务:
  启动 → 加载模型权重 → 分配 KV Cache → Warm up → 就绪
  
  时间分解 (70B 模型, 4x H100):
  ├── 模型加载 (从磁盘/网络读取 140GB):      60-180 秒
  ├── KV Cache 预分配:                        5-10 秒
  ├── CUDA Graph Capture (可选):              30-60 秒
  └── Warm-up 推理:                           10-20 秒
  
  总计: 约 2-5 分钟
```

### 1.2 有状态的 KV Cache

- 每个正在处理的请求都有对应的 KV Cache
- 模型更新后，旧 KV Cache 通常不兼容新模型
- 不能简单地"切换"——需要等待旧请求完成

### 1.3 GPU 资源稀缺

- 不像 CPU 服务可以轻松多开几个实例
- 新旧模型同时运行需要双倍 GPU（或预留资源）

## 2. 部署策略

### 2.1 蓝绿部署 (Blue-Green Deployment)

蓝绿部署是最安全的策略：完全准备好新环境后一次性切换。

```
Phase 1: 准备新环境 (Green)
┌─────────────────────────────────────────────┐
│                                             │
│  ┌──────────┐           ┌──────────┐        │
│  │ Blue     │ ← 流量 ← │ Router   │        │
│  │ Model v1 │           └──────────┘        │
│  │ 4x H100  │                               │
│  └──────────┘                               │
│                                             │
│  ┌──────────┐                               │
│  │ Green    │  (加载中, 无流量)               │
│  │ Model v2 │                               │
│  │ 4x H100  │  ← 需要额外 4x H100!         │
│  └──────────┘                               │
│                                             │
└─────────────────────────────────────────────┘

Phase 2: 切换流量
┌─────────────────────────────────────────────┐
│                                             │
│  ┌──────────┐                               │
│  │ Blue     │  (drain 中, 等待旧请求完成)    │
│  │ Model v1 │                               │
│  │ 4x H100  │                               │
│  └──────────┘                               │
│                                             │
│  ┌──────────┐           ┌──────────┐        │
│  │ Green    │ ← 流量 ← │ Router   │        │
│  │ Model v2 │           └──────────┘        │
│  │ 4x H100  │                               │
│  └──────────┘                               │
│                                             │
└─────────────────────────────────────────────┘

Phase 3: 清理
- Blue 所有请求完成后，释放 Blue 的 GPU
- 或保留 Blue 用于快速回滚
```

**实现：**

```python
import httpx
import asyncio
from datetime import datetime

class BlueGreenDeployer:
    def __init__(self, router_endpoint: str):
        self.router = router_endpoint
        self.blue = None
        self.green = None
    
    async def deploy(self, new_model_config: dict, new_endpoints: list[str]):
        """执行蓝绿部署"""
        
        # Step 1: 部署新模型到 Green 环境
        print(f"[{datetime.now()}] Starting Green deployment...")
        self.green = new_endpoints
        
        # Step 2: 等待 Green 就绪
        for endpoint in self.green:
            await self._wait_ready(endpoint, timeout=600)
        print(f"[{datetime.now()}] Green is ready")
        
        # Step 3: 验证 Green (发送测试请求)
        if not await self._validate(self.green):
            print("Green validation FAILED, aborting")
            return False
        
        # Step 4: 切换流量
        print(f"[{datetime.now()}] Switching traffic to Green...")
        await self._update_router(self.green)
        
        # Step 5: Drain Blue (等待旧请求完成)
        if self.blue:
            print(f"[{datetime.now()}] Draining Blue...")
            for endpoint in self.blue:
                await self._drain(endpoint, timeout=120)
            print(f"[{datetime.now()}] Blue drained")
        
        # Step 6: 角色交换
        old_blue = self.blue
        self.blue = self.green
        self.green = None
        
        print(f"[{datetime.now()}] Deployment complete!")
        return True
    
    async def rollback(self):
        """快速回滚到 Blue"""
        if self.blue:
            print("Rolling back to previous version...")
            await self._update_router(self.blue)
    
    async def _wait_ready(self, endpoint: str, timeout: int = 600):
        """等待 endpoint 就绪"""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(f"{endpoint}/health")
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(5)
        raise TimeoutError(f"{endpoint} not ready after {timeout}s")
    
    async def _validate(self, endpoints: list[str]) -> bool:
        """发送测试请求验证新模型"""
        test_prompts = [
            "What is 2+2?",
            "Hello, how are you?",
            "Translate 'hello' to Chinese.",
        ]
        
        for endpoint in endpoints:
            for prompt in test_prompts:
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.post(
                            f"{endpoint}/v1/chat/completions",
                            json={
                                "model": "default",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 50,
                            }
                        )
                        if resp.status_code != 200:
                            print(f"Validation failed: {resp.status_code}")
                            return False
                except Exception as e:
                    print(f"Validation error: {e}")
                    return False
        return True
    
    async def _drain(self, endpoint: str, timeout: int = 120):
        """停止接收新请求，等待现有请求完成"""
        async with httpx.AsyncClient() as client:
            # 暂停接收新请求
            await client.post(f"{endpoint}/pause")
            
            # 等待所有请求完成
            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout:
                resp = await client.get(f"{endpoint}/health")
                data = resp.json()
                if data.get("active_requests", 0) == 0:
                    return
                await asyncio.sleep(2)
    
    async def _update_router(self, endpoints: list[str]):
        """更新路由指向新的 endpoints"""
        async with httpx.AsyncClient() as client:
            await client.post(f"{self.router}/update_backends", 
                            json={"backends": endpoints})
```

### 2.2 金丝雀发布 (Canary Deployment)

逐步将流量从旧模型迁移到新模型，降低风险。

```
Stage 1 (5% 流量):
┌──────────┐
│ Router   │──── 95% ──→ [Model v1 × 3 replicas]
│          │────  5% ──→ [Model v2 × 1 replica ]
└──────────┘

Stage 2 (25% 流量, 观察 30 分钟):
┌──────────┐
│ Router   │──── 75% ──→ [Model v1 × 3 replicas]
│          │──── 25% ──→ [Model v2 × 1 replica ]
└──────────┘

Stage 3 (50% 流量, 观察 1 小时):
┌──────────┐
│ Router   │──── 50% ──→ [Model v1 × 2 replicas]
│          │──── 50% ──→ [Model v2 × 2 replicas]
└──────────┘

Stage 4 (100% 流量):
┌──────────┐
│ Router   │──── 100% ─→ [Model v2 × 3 replicas]
└──────────┘
```

**自动化金丝雀发布：**

```python
class CanaryDeployer:
    def __init__(self, router, monitor):
        self.router = router
        self.monitor = monitor
        self.stages = [
            {"canary_weight": 5,  "duration_min": 15, "description": "5% canary"},
            {"canary_weight": 25, "duration_min": 30, "description": "25% canary"},
            {"canary_weight": 50, "duration_min": 60, "description": "50% canary"},
            {"canary_weight": 100, "duration_min": 0,  "description": "Full rollout"},
        ]
    
    async def deploy(self, canary_endpoints: list[str], stable_endpoints: list[str]):
        """执行金丝雀发布"""
        
        for i, stage in enumerate(self.stages):
            print(f"Stage {i+1}: {stage['description']}")
            
            # 设置流量权重
            await self.router.set_weights(
                stable={"endpoints": stable_endpoints, "weight": 100 - stage["canary_weight"]},
                canary={"endpoints": canary_endpoints, "weight": stage["canary_weight"]},
            )
            
            if stage["duration_min"] == 0:
                break
            
            # 观察期：监控关键指标
            healthy = await self._observe(
                duration_min=stage["duration_min"],
                canary_endpoints=canary_endpoints,
            )
            
            if not healthy:
                print(f"Canary unhealthy at stage {i+1}, rolling back!")
                await self.router.set_weights(
                    stable={"endpoints": stable_endpoints, "weight": 100},
                    canary={"endpoints": canary_endpoints, "weight": 0},
                )
                return False
        
        print("Canary deployment completed successfully!")
        return True
    
    async def _observe(self, duration_min: int, canary_endpoints: list[str]) -> bool:
        """观察金丝雀实例的健康状态"""
        check_interval = 30  # 每 30 秒检查一次
        checks = duration_min * 60 // check_interval
        
        for _ in range(checks):
            await asyncio.sleep(check_interval)
            
            metrics = await self.monitor.get_metrics(canary_endpoints)
            
            # 检查告警条件
            if metrics["error_rate"] > 0.05:
                print(f"Error rate too high: {metrics['error_rate']:.1%}")
                return False
            
            if metrics["p99_ttft"] > 5.0:
                print(f"P99 TTFT too high: {metrics['p99_ttft']:.2f}s")
                return False
            
            if metrics["p99_tbt"] > 0.2:
                print(f"P99 TBT too high: {metrics['p99_tbt']:.3f}s")
                return False
            
            print(f"  Canary healthy - error_rate={metrics['error_rate']:.3%}, "
                  f"ttft_p99={metrics['p99_ttft']:.2f}s, "
                  f"tbt_p99={metrics['p99_tbt']:.3f}s")
        
        return True
```

### 2.3 A/B 测试

A/B 测试用于评估新模型的业务效果（不仅是技术指标）。

```python
class ABTestRouter:
    """
    基于用户 ID 的 A/B 测试路由
    确保同一用户始终看到同一版本（一致性）
    """
    
    def __init__(self, control_endpoints: list, treatment_endpoints: list, 
                 treatment_ratio: float = 0.1):
        self.control = control_endpoints
        self.treatment = treatment_endpoints
        self.treatment_ratio = treatment_ratio
    
    def assign_group(self, user_id: str) -> str:
        """
        基于 user_id 的确定性分组
        - 同一 user_id 始终分配到同一组
        - 支持调整 treatment_ratio 时部分用户切换
        """
        # 使用 hash 确保确定性
        hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        bucket = hash_val % 100
        
        if bucket < self.treatment_ratio * 100:
            return "treatment"
        return "control"
    
    def get_endpoint(self, user_id: str) -> str:
        """获取该用户应该使用的 endpoint"""
        group = self.assign_group(user_id)
        
        if group == "treatment":
            endpoints = self.treatment
        else:
            endpoints = self.control
        
        # 在组内 round-robin
        idx = hash(user_id) % len(endpoints)
        return endpoints[idx]
    
    def log_assignment(self, user_id: str, request_id: str, group: str):
        """记录分组信息，用于后续分析"""
        # 写入分析数据库
        analytics_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "request_id": request_id,
            "ab_group": group,
            "model_version": "v2" if group == "treatment" else "v1",
        }
        # send_to_analytics(analytics_event)
```

**A/B 测试的关键指标：**

| 维度 | 指标 | 说明 |
|------|------|------|
| 质量 | 用户满意度评分 | thumbs up/down |
| 质量 | 任务完成率 | 用户是否得到了想要的答案 |
| 性能 | TTFT / TBT | 新模型是否更快/更慢 |
| 业务 | 会话长度 | 用户与 AI 的交互轮数 |
| 业务 | 重试率 | 用户是否需要多次提问 |
| 成本 | tokens/请求 | 新模型是否更啰嗦/简洁 |

## 3. KV Cache 兼容性问题

### 3.1 什么时候 KV Cache 不兼容

模型更新后，KV Cache 是否还能复用取决于变化的类型：

| 变化类型 | KV Cache 兼容 | 说明 |
|----------|-------------|------|
| System prompt 变更 | 不兼容 | KV Cache 内容完全不同 |
| LoRA adapter 更换 | 不兼容 | Attention 权重变化 |
| 模型权重微调 (Fine-tune) | 不兼容 | 所有层权重变化 |
| 量化精度变更 (FP16→FP8) | 不兼容 | 数值精度差异 |
| vLLM 版本升级 | 可能兼容 | 取决于 KV Cache 格式是否变化 |
| 推理参数变更 (temperature 等) | 兼容 | KV Cache 只依赖输入，不依赖采样参数 |
| 增加 LoRA adapter (不换现有) | 兼容 | 新 adapter 的请求创建新 cache |

### 3.2 Cache 失效处理策略

```python
class CacheInvalidationManager:
    """模型更新时的 KV Cache 失效管理"""
    
    def __init__(self, cache_store):
        self.cache_store = cache_store
        self.model_version = None
    
    def on_model_update(self, new_version: str, update_type: str):
        """
        模型更新时决定 cache 处理策略
        """
        if update_type in ["fine_tune", "lora_swap", "quantization_change"]:
            # 完全不兼容 → 清除所有 cache
            self._invalidate_all(reason=f"model_update_{update_type}")
            
        elif update_type == "system_prompt_change":
            # 仅清除受影响的 prefix cache
            self._invalidate_prefix_cache(reason="system_prompt_change")
            
        elif update_type == "vllm_upgrade":
            # 检查 cache 格式兼容性
            if not self._check_cache_format_compatible(new_version):
                self._invalidate_all(reason="cache_format_incompatible")
        
        self.model_version = new_version
    
    def _invalidate_all(self, reason: str):
        """清除所有 KV Cache"""
        print(f"Invalidating all KV Cache: {reason}")
        self.cache_store.clear()
        # 注意：这不会中断正在运行的请求
        # 它们的 KV Cache 仍在 GPU 上，只是新请求无法复用
    
    def _invalidate_prefix_cache(self, reason: str):
        """仅清除 prefix cache (保留运行中请求的 cache)"""
        print(f"Invalidating prefix cache: {reason}")
        self.cache_store.clear_prefix_cache()
    
    def _check_cache_format_compatible(self, new_version: str) -> bool:
        """检查新版 vLLM 的 cache 格式是否兼容"""
        # 这里需要比较 cache 元数据
        # 例如 block_size, num_heads, head_dim 等
        return True  # 简化
```

### 3.3 预热策略

模型更新后 prefix cache 全部失效，需要重新预热：

```python
async def warm_up_cache(endpoint: str, common_prefixes: list[str]):
    """
    模型更新后预热 prefix cache
    发送常见前缀的请求，让新实例构建 KV Cache
    """
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for prefix in common_prefixes:
            task = client.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": "Hi"}
                    ],
                    "max_tokens": 1,  # 只需要 1 个 token 来触发 cache
                }
            )
            tasks.append(task)
        
        # 并发发送所有预热请求
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success = sum(1 for r in results if not isinstance(r, Exception))
        print(f"Cache warm-up: {success}/{len(common_prefixes)} prefixes cached")

# 常见前缀列表（从生产日志中提取）
COMMON_PREFIXES = [
    "You are a helpful customer service agent for ...",
    "You are a code review assistant. Review the following ...",
    "You are a translation expert. Translate the following ...",
    # ... 更多高频 system prompts
]
```

## 4. 零停机更新策略

### 4.1 Rolling Update (滚动更新)

Kubernetes 原生的滚动更新策略，适合有多副本的场景：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-serving
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1         # 最多多出 1 个 Pod (需要额外 GPU)
      maxUnavailable: 0   # 不允许减少可用 Pod (零停机)
  
  template:
    spec:
      terminationGracePeriodSeconds: 300  # 5 分钟 graceful shutdown
      
      containers:
        - name: vllm
          image: vllm-serving:v2.0  # 更新镜像
          
          lifecycle:
            preStop:
              exec:
                command:
                  - /bin/sh
                  - -c
                  - |
                    # 停止接收新请求
                    curl -X POST localhost:8000/pause
                    # 等待正在处理的请求完成
                    while [ $(curl -s localhost:8000/metrics | grep 'vllm:num_requests_running' | awk '{print $2}') -gt 0 ]; do
                      sleep 5
                    done
                    # 预热新实例的 cache (可选)
                    # python /opt/warmup.py
```

**滚动更新的时间线：**

```
t=0min   Pod-0(v1) ✓   Pod-1(v1) ✓   Pod-2(v1) ✓
         [serving]      [serving]      [serving]

t=0min   新建 Pod-3(v2) → 加载模型中...
         Pod-0(v1) ✓   Pod-1(v1) ✓   Pod-2(v1) ✓

t=3min   Pod-3(v2) ✓  就绪
         Pod-0(v1) → drain (停止接收新请求，完成旧请求)
         
t=5min   Pod-0(v1) drain 完成 → 删除
         Pod-3(v2) ✓   Pod-1(v1) ✓   Pod-2(v1) ✓
         
t=5min   新建 Pod-4(v2) → 加载模型中...
         ...重复直到所有 Pod 更新完毕

t=15min  Pod-3(v2) ✓   Pod-4(v2) ✓   Pod-5(v2) ✓
         更新完成!
```

### 4.2 GPU 资源受限时的更新策略

当没有额外 GPU 做蓝绿或 rolling update 时：

```python
class InPlaceUpdater:
    """
    原地更新策略：在同一组 GPU 上更新模型
    牺牲短暂可用性，换取不需要额外 GPU
    """
    
    async def update(self, replicas: list[str], new_model: str):
        """逐个 replica 原地更新"""
        
        for i, replica in enumerate(replicas):
            print(f"Updating replica {i+1}/{len(replicas)}: {replica}")
            
            # 1. 将 replica 从负载均衡器摘除
            await self.router.remove_backend(replica)
            
            # 2. Drain：等待正在处理的请求完成
            await self._drain(replica, timeout=120)
            
            # 3. 停止旧进程
            await self._stop_vllm(replica)
            
            # 4. 启动新模型
            await self._start_vllm(replica, new_model)
            
            # 5. 等待就绪
            await self._wait_ready(replica, timeout=600)
            
            # 6. 预热 cache
            await warm_up_cache(replica, COMMON_PREFIXES)
            
            # 7. 重新加入负载均衡器
            await self.router.add_backend(replica)
            
            print(f"Replica {i+1} updated successfully")
            
            # 8. 观察一段时间确认稳定
            await asyncio.sleep(60)
            if not await self._check_healthy(replica):
                print(f"WARNING: Replica {i+1} unhealthy after update!")
                # 决定是否继续或回滚
```

## 5. 实际部署流程

### 5.1 完整的模型更新 Checklist

```markdown
## 模型更新前
- [ ] 在测试环境验证新模型
  - [ ] 推理质量评估 (benchmark scores)
  - [ ] 性能指标对比 (throughput, latency)
  - [ ] 内存使用对比 (KV Cache 需求)
  - [ ] 兼容性测试 (API 格式, tokenizer)
- [ ] 准备回滚方案
  - [ ] 确认旧模型镜像/文件仍然可用
  - [ ] 确认回滚流程经过测试
- [ ] 通知相关团队 (SRE, 产品, 客户)
- [ ] 选择低流量时段执行

## 模型更新中
- [ ] 部署新模型到 staging 环境
- [ ] 执行 smoke test
- [ ] 启动金丝雀发布 (5% → 25% → 50% → 100%)
- [ ] 每个阶段监控:
  - [ ] 错误率
  - [ ] TTFT / TBT
  - [ ] 用户反馈
  - [ ] GPU 资源使用
- [ ] 确认 prefix cache warm-up 完成

## 模型更新后
- [ ] 确认所有 replica 已更新
- [ ] 确认 prefix cache hit rate 恢复到正常水平
- [ ] 确认无异常告警
- [ ] 更新监控 dashboard 的版本标签
- [ ] 清理旧模型文件 (如有)
- [ ] 更新文档和 changelog
```

### 5.2 CI/CD Pipeline

```yaml
# .github/workflows/model-deploy.yml
name: Model Deployment

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "Model version to deploy"
        required: true
      strategy:
        description: "Deployment strategy"
        required: true
        type: choice
        options:
          - canary
          - blue-green
          - rolling
      target_env:
        description: "Target environment"
        required: true
        type: choice
        options:
          - staging
          - production

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - name: Run quality benchmarks
        run: |
          python run_benchmarks.py \
            --model ${{ inputs.model_version }} \
            --benchmarks "mmlu,humaneval,mt_bench"
      
      - name: Run performance benchmarks
        run: |
          python run_perf_test.py \
            --model ${{ inputs.model_version }} \
            --metrics "ttft,tbt,throughput"
      
      - name: Compare with current production
        run: |
          python compare_results.py \
            --new ${{ inputs.model_version }} \
            --current production \
            --fail-if-regression

  deploy-staging:
    needs: validate
    if: inputs.target_env == 'staging' || inputs.target_env == 'production'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/vllm-staging \
            vllm=vllm-serving:${{ inputs.model_version }}
          kubectl rollout status deployment/vllm-staging --timeout=600s
      
      - name: Run integration tests
        run: python run_integration_tests.py --env staging

  deploy-production:
    needs: deploy-staging
    if: inputs.target_env == 'production'
    runs-on: ubuntu-latest
    environment: production  # 需要审批
    steps:
      - name: Deploy canary
        if: inputs.strategy == 'canary'
        run: |
          python deploy.py canary \
            --model ${{ inputs.model_version }} \
            --stages "5,25,50,100" \
            --observe-minutes "15,30,60,0"
      
      - name: Deploy blue-green
        if: inputs.strategy == 'blue-green'
        run: |
          python deploy.py blue-green \
            --model ${{ inputs.model_version }}
```

## 6. LoRA Adapter 的热更新

LoRA adapter 的更新比全模型更新轻量得多：

```python
# vLLM 支持动态加载/卸载 LoRA adapter
# 无需重启服务

# 加载新的 LoRA adapter
async def load_lora(endpoint: str, lora_name: str, lora_path: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{endpoint}/v1/load_lora_adapter",
            json={
                "lora_name": lora_name,
                "lora_path": lora_path,
            }
        )
        return resp.json()

# 卸载 LoRA adapter
async def unload_lora(endpoint: str, lora_name: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{endpoint}/v1/unload_lora_adapter",
            json={"lora_name": lora_name}
        )
        return resp.json()

# 使用特定 LoRA adapter 推理
async def inference_with_lora(endpoint: str, messages: list, lora_name: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": lora_name,  # 指定 LoRA adapter 名称
                "messages": messages,
                "max_tokens": 512,
            }
        )
        return resp.json()
```

**LoRA 热更新的优势：**
- 加载时间：秒级（vs 全模型的分钟级）
- 显存开销：几十 MB（vs 几十 GB）
- 不影响基础模型的 KV Cache
- 支持同时加载多个 LoRA adapter

## 7. 总结

| 部署策略 | 额外 GPU | 停机时间 | 风险 | 适用场景 |
|----------|---------|---------|------|---------|
| 蓝绿部署 | 2x | 零 | 最低 | 资源充裕，追求安全 |
| 金丝雀 | 1个 replica | 零 | 低 | 需要验证新模型效果 |
| 滚动更新 | 1个 replica | 零 | 中 | Kubernetes 环境 |
| 原地更新 | 无 | 数分钟/replica | 高 | GPU 资源紧张 |
| A/B 测试 | 1+ replicas | 零 | 低 | 评估业务影响 |
| LoRA 热更新 | 极少 | 零 | 低 | 仅更新 adapter |

**核心原则：** 模型更新的首要目标是零停机和可回滚。在资源允许的情况下，优先选择蓝绿或金丝雀策略。无论使用哪种策略，都要有经过验证的回滚方案。

---

> **延伸阅读：**
> - [vLLM LoRA 支持文档](https://docs.vllm.ai/en/latest/models/lora.html)
> - Kubernetes Rolling Update 策略详解
> - Martin Fowler: Blue-Green Deployment 原始文章
