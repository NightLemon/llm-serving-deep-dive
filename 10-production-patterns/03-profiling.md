# 性能 Profiling

> 当监控告诉你"系统慢了"，profiling 告诉你"为什么慢"。
> 本节介绍 LLM 推理性能诊断的完整方法论和工具链。

## 1. 诊断方法论

### 1.1 LLM 推理的三种瓶颈

LLM 推理的性能瓶颈可以归为三类，每类的症状和对策完全不同：

```
┌──────────────────────────────────────────────────────────┐
│                  LLM 推理瓶颈诊断树                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  GPU SM 利用率高 + 显存带宽有余?                           │
│  ├── 是 → Compute-Bound（计算瓶颈）                      │
│  │   └── 典型场景：Prefill 阶段，大 batch decode          │
│  │                                                       │
│  显存带宽接近上限 + GPU SM 空闲?                           │
│  ├── 是 → Memory-Bound（带宽瓶颈）                       │
│  │   └── 典型场景：Decode 阶段 (batch=1)                  │
│  │                                                       │
│  GPU 在等待 NCCL 通信?                                    │
│  ├── 是 → Communication-Bound（通信瓶颈）                │
│  │   └── 典型场景：TP 跨节点，PP bubble                   │
│  │                                                       │
│  以上都不是?                                              │
│  └── 检查 CPU overhead、Python GIL、I/O 等               │
└──────────────────────────────────────────────────────────┘
```

### 1.2 各阶段的瓶颈特征

| 阶段 | 主要瓶颈 | Arithmetic Intensity | GPU 利用率 | 显存带宽利用率 |
|------|---------|---------------------|-----------|--------------|
| Prefill (长 prompt) | Compute-bound | 高 (>100 FLOPs/byte) | 高 (>80%) | 中 |
| Decode (小 batch) | Memory-bound | 低 (<10 FLOPs/byte) | 低 (<30%) | 高 (>80%) |
| Decode (大 batch) | Compute-bound | 中-高 | 中-高 | 中-高 |
| TP AllReduce | Communication | N/A | 低（等待中） | 低 |

### 1.3 Roofline Model 应用

Roofline Model 是判断 compute-bound 还是 memory-bound 的经典工具。

```
Performance (FLOPS)
│
│                    ╱ Compute Ceiling (H100: 989 TFLOPS FP16)
│                   ╱─────────────────────────────────
│                  ╱
│                 ╱
│                ╱    ★ Prefill (large batch)
│               ╱
│              ╱
│             ╱  ★ Decode (large batch)
│            ╱
│           ╱
│          ╱  ★ Decode (batch=1)
│         ╱
│        ╱ Memory BW Ceiling (H100: 3.35 TB/s)
│       ╱
└───────────────────────────────────────────────
         Arithmetic Intensity (FLOPs/Byte)
         
         拐点 = Peak FLOPS / Peak BW = 989T / 3.35T ≈ 295
```

**Attention 层的 Arithmetic Intensity 分析：**

```python
def compute_attention_ai(batch_size, seq_len, head_dim, num_heads):
    """
    计算 Attention 的 Arithmetic Intensity
    
    QKV projection: FLOPs = 2 * B * S * 3 * H * D
                    Bytes = (weight: 3*H*D*D*2 + activation: B*S*H*D*2) 
    
    Attention: FLOPs = 2 * B * num_heads * S * S * head_dim (prefill)
               或 2 * B * num_heads * 1 * S * head_dim (decode)
    """
    # Decode 阶段 (生成 1 个 token)
    # FLOPs: 2 * B * num_heads * 1 * S * head_dim  
    # Bytes: 读取 KV cache = 2 * B * num_heads * S * head_dim * 2 (FP16)
    
    decode_flops = 2 * batch_size * num_heads * 1 * seq_len * head_dim
    decode_bytes = 2 * num_heads * seq_len * head_dim * 2  # 读取 KV cache
    
    decode_ai = decode_flops / decode_bytes
    # 对于 head_dim=128: AI = 2*B*1*128 / (2*S*128*2) = B/S (非常小!)
    
    # Prefill 阶段 
    prefill_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    prefill_bytes = 2 * num_heads * seq_len * head_dim * 2  # 简化
    
    prefill_ai = prefill_flops / prefill_bytes
    # AI ≈ B * S (可以非常大)
    
    return decode_ai, prefill_ai

# Llama-3.1-70B, batch=1, seq=2048
d_ai, p_ai = compute_attention_ai(1, 2048, 128, 64)
print(f"Decode AI: {d_ai:.1f}")   # ≈ 0.5 → 极度 memory-bound
print(f"Prefill AI: {p_ai:.1f}")  # ≈ 1024 → compute-bound
```

## 2. Profiling 工具

### 2.1 NVIDIA Nsight Systems (nsys)

nsys 提供全局时间线视图，是定位宏观瓶颈的首选工具。

**基本用法：**

```bash
# Profile vLLM 推理
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=vllm_profile \
    --duration=30 \
    --gpu-metrics-device=0 \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 4096

# 生成报告
nsys stats vllm_profile.nsys-rep --report cuda_kern_exec_sum
```

**关键分析点：**

```
nsys 时间线分析清单:
1. Kernel 执行时间占比
   - 找到最耗时的 top-10 CUDA kernels
   - 区分 compute kernel (GEMM) 和 memory kernel (copy, attention)

2. GPU 空闲间隙 (gaps)
   - 连续 kernel 之间的空隙 = CPU overhead
   - 大间隙 → Python/调度开销过大

3. NCCL 通信
   - AllReduce 的占比
   - 通信和计算是否重叠 (overlap)

4. Memory 操作
   - H2D / D2H copy 频率和耗时
   - 是否有不必要的数据搬运
```

**nsys 报告解读示例：**

```
CUDA Kernel Statistics:
  Kernel Name                              Time(%)   Avg(us)   Calls
  ampere_fp16_s16816gemm_fp16_...           35.2%     125.3     1024   ← GEMM (compute)
  flash_fwd_kernel                          28.1%      89.7      512   ← FlashAttention
  void at::native::vectorized_elementwise   12.3%      15.2     2048   ← elementwise ops
  ncclKernel_AllReduce_RING_LL_Sum          8.5%       45.6      256   ← TP 通信
  void at::native::copy_kernel              3.2%       22.1      128   ← memory copy
```

### 2.2 NVIDIA Nsight Compute (ncu)

ncu 用于单个 kernel 的深度分析，适合优化热点 kernel。

```bash
# 只 profile attention kernel
ncu --set full \
    --kernel-name "flash_fwd_kernel" \
    --launch-count 10 \
    --output attention_profile \
    python benchmark_attention.py

# 查看 roofline 分析
ncu --import attention_profile.ncu-rep --page roofline
```

**ncu 关键指标：**

| 指标 | 含义 | 理想值 |
|------|------|-------|
| SM Throughput | SM 计算利用率 | > 80% (compute-bound kernel) |
| Memory Throughput | 显存带宽利用率 | > 80% (memory-bound kernel) |
| Achieved Occupancy | 实际占用率 | > 50% |
| Warp Stall Reasons | warp 阻塞原因 | 不应被 memory 大量阻塞 |

### 2.3 PyTorch Profiler

PyTorch Profiler 适合分析 Python 层面的性能：

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 对 vLLM 的一次推理做 profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    schedule=torch.profiler.schedule(
        wait=2,     # 跳过前 2 步
        warmup=3,   # warmup 3 步
        active=5,   # 实际采集 5 步
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs")
) as prof:
    for step in range(10):
        with record_function("inference_step"):
            # 执行推理
            model_runner.execute_model(...)
        prof.step()

# 打印 kernel 统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**输出示例：**

```
Name                                    CPU Total   CUDA Total   Calls
--------------------------------------  ----------  -----------  -----
aten::mm                                 125.3ms     98.7ms       48
flash_attn::flash_fwd                    89.2ms      85.1ms       24
aten::copy_                              45.6ms      12.3ms       96
nccl:all_reduce                          38.4ms      35.2ms       24
aten::layer_norm                         22.1ms      18.5ms       48
```

### 2.4 vLLM 内置 Profiling

vLLM 提供了内置的 profiling 工具，可以直接收集详细的执行追踪：

```bash
# 方法 1：使用环境变量启用
VLLM_TORCH_PROFILER_DIR=./vllm_traces \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct

# 方法 2：通过 API 动态启停 profiling
# 开始 profiling
curl -X POST http://localhost:8000/start_profile

# 发送一些请求...

# 停止并保存
curl -X POST http://localhost:8000/stop_profile
# 结果保存在 VLLM_TORCH_PROFILER_DIR 中
```

**分析 vLLM trace：**

```bash
# 使用 Chrome Trace Viewer 或 Perfetto 查看
# 在浏览器中打开 chrome://tracing 或 https://ui.perfetto.dev/
# 加载生成的 .json trace 文件
```

trace 文件中可以看到：
- 每个 scheduler step 的耗时
- prefill 和 decode 的执行时间
- model forward 中各层的耗时
- CUDA kernel 的 launch 和执行

## 3. 常见性能问题诊断

### 3.1 问题诊断表

| 症状 | 可能原因 | 诊断方法 | 对策 |
|------|---------|---------|------|
| TTFT 高且不稳定 | 长 prompt 阻塞 decode 调度 | 检查 `num_running_requests`，看是否有长 prefill | 启用 chunked prefill，设置 `max_num_batched_tokens` |
| TBT 周期性飙高 | Preemption 导致重算 | 检查 `num_preemptions_total` 增长速率 | 增大 `gpu_memory_utilization`，减小 `max_num_seqs` |
| 吞吐量低于预期 | Batch size 太小 | nsys 查看 GEMM kernel 利用率 | 增大 `max_num_seqs`，检查是否有足够的并发请求 |
| GPU 利用率低 | CPU 调度开销大 | nsys 查看 kernel 间隙 | 检查 tokenizer 性能，减少 Python overhead |
| TP 扩展效率低 | 通信瓶颈 | nsys 查看 NCCL 占比 | 确认 NVLink 连接，减小 TP degree |
| 显存 OOM | KV Cache 分配过大 | 检查 `gpu_cache_usage_perc` | 降低 `gpu_memory_utilization`，启用 KV 量化 |
| Prefix cache 命中率低 | 路由策略不当 | 检查 `gpu_prefix_cache_hit_rate` | 启用 cache-aware routing |
| 首次请求特别慢 | CUDA graph capture / 模型编译 | 检查启动日志 | 使用 warm-up 请求 |

### 3.2 Prefill 性能分析

Prefill 阶段是计算密集型的，优化目标是最大化 GPU 计算利用率。

```python
def analyze_prefill_performance(
    model_params_B: float,  # 模型参数量 (billions)
    prompt_tokens: int,
    gpu_tflops: float,  # GPU 峰值 TFLOPS (FP16)
    measured_time_ms: float  # 实际测量的 prefill 时间
):
    """
    分析 prefill 是否达到了理论峰值
    
    FLOPs 估算: 2 * params * seq_len (每个 token 的前向传播)
    """
    total_flops = 2 * model_params_B * 1e9 * prompt_tokens
    theoretical_time_ms = (total_flops / (gpu_tflops * 1e12)) * 1000
    
    mfu = theoretical_time_ms / measured_time_ms  # Model FLOPs Utilization
    
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Total FLOPs: {total_flops/1e12:.2f} TFLOPS")
    print(f"Theoretical time: {theoretical_time_ms:.2f} ms")
    print(f"Measured time: {measured_time_ms:.2f} ms")
    print(f"MFU: {mfu:.1%}")
    
    if mfu > 0.7:
        print("✓ 接近计算峰值，优化空间有限")
    elif mfu > 0.4:
        print("△ 有一定优化空间，检查 attention 实现和 batch 策略")
    else:
        print("✗ 远低于峰值，可能有 CPU overhead 或内存瓶颈")

# 示例：Llama-3.1-70B on H100, 2048 tokens prefill
analyze_prefill_performance(
    model_params_B=70,
    prompt_tokens=2048,
    gpu_tflops=989,  # H100 SXM FP16
    measured_time_ms=350  # 假设测量值
)
```

### 3.3 Decode 性能分析

Decode 阶段是带宽密集型的，优化目标是最大化显存带宽利用率。

```python
def analyze_decode_performance(
    model_size_bytes: float,  # 模型大小 (bytes, 含 KV cache)
    batch_size: int,
    gpu_bw_tb_s: float,  # GPU 显存带宽 (TB/s)
    measured_tbt_ms: float  # 实际测量的 TBT
):
    """
    Decode 阶段每生成一个 token，需要读取整个模型权重一次
    (batch 内的多个请求共享模型权重读取)
    
    理论 TBT = model_size / (gpu_bandwidth * batch_size_factor)
    """
    # 简化模型：每步读取一次模型权重
    bytes_per_step = model_size_bytes  # 读取权重 + KV cache read/write
    theoretical_tbt_ms = (bytes_per_step / (gpu_bw_tb_s * 1e12)) * 1000
    
    bandwidth_utilization = theoretical_tbt_ms / measured_tbt_ms
    
    print(f"Batch size: {batch_size}")
    print(f"Bytes per step: {bytes_per_step/1e9:.2f} GB")
    print(f"Theoretical TBT: {theoretical_tbt_ms:.2f} ms")
    print(f"Measured TBT: {measured_tbt_ms:.2f} ms")
    print(f"Bandwidth utilization: {bandwidth_utilization:.1%}")
    
    if bandwidth_utilization > 0.8:
        print("✓ 接近带宽上限，考虑量化（FP8/INT4）减少读取量")
    elif bandwidth_utilization > 0.5:
        print("△ 有优化空间，检查 attention kernel 和内存访问模式")
    else:
        print("✗ 带宽利用率低，可能有 CPU overhead 或 kernel launch 开销")

# 示例：Llama-3.1-70B (FP16) on H100, batch=1
analyze_decode_performance(
    model_size_bytes=70 * 2 * 1e9,  # 70B params × 2 bytes (FP16)
    batch_size=1,
    gpu_bw_tb_s=3.35,  # H100 SXM
    measured_tbt_ms=50
)
```

## 4. 实际 Profiling 案例

### 4.1 案例：诊断 TBT 抖动

**症状：** Llama-3.1-70B 服务的 P99 TBT 间歇性飙升到 500ms+，正常时 TBT 约 40ms。

**诊断过程：**

```bash
# Step 1: 查看 Prometheus 指标确认问题
# 发现 TBT P99 与 num_preemptions_total 增长高度相关

# Step 2: nsys profile 捕获异常时段
nsys profile --output=tbt_spike \
    --trigger=cuda-graph-start \
    --duration=60 \
    python -m vllm.entrypoints.openai.api_server ...

# Step 3: 分析 trace
nsys stats tbt_spike.nsys-rep --report cuda_kern_exec_sum
```

**发现：**

```
时间线分析:
t=0ms    : decode step (正常, 40ms)
t=40ms   : decode step (正常, 40ms)
t=80ms   : 新的长 prompt 请求到达 (4096 tokens)
t=80ms   : scheduler 决定做 prefill (4096 tokens)
t=80ms   : 正在 decode 的请求被暂停
t=280ms  : prefill 完成 (耗时 200ms)
t=280ms  : 恢复 decode (但 KV Cache 满了)
t=280ms  : preemption! 部分请求的 KV Cache 被换出
t=380ms  : decode step (因为重算, 耗时 100ms)
          → 这段时间用户看到 TBT 飙升
```

**对策：**

```bash
# 启用 chunked prefill，限制每步的 prefill tokens
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048  # 每步最多处理 2048 tokens
```

### 4.2 案例：诊断 TP 通信瓶颈

**症状：** 4 卡 TP 部署 Llama-3.1-70B，吞吐量只有单卡理论值的 2.5x（预期接近 4x）。

**诊断过程：**

```bash
# 使用 nsys 查看 NCCL 通信占比
nsys profile --trace=cuda,nvtx,nccl \
    --output=tp4_profile \
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --tensor-parallel-size 4
```

**nsys 结果分析：**

```
NCCL Communication Analysis:
  Total CUDA time:          1000ms (100%)
  Compute (GEMM + Attn):     600ms (60%)
  NCCL AllReduce:             320ms (32%)  ← 过高!
  Other:                       80ms (8%)

Per-AllReduce breakdown:
  Avg AllReduce size: 16 MB
  Avg AllReduce time: 2.5ms
  Expected (NVLink): 0.5ms   ← 5x 差距!
```

**根因：** 4 张 GPU 分布在 2 个 NUMA 节点上，跨 NUMA 走的是 PCIe 而不是 NVLink。

**对策：**

```bash
# 确认 GPU 拓扑
nvidia-smi topo -m

# 绑定 GPU 到同一 NVLink 域
# 例如 GPU 0,1,2,3 都在同一 NVLink switch 下
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

### 4.3 案例：Memory-Bound → Compute-Bound 的转折点

**目标：** 找到 decode batch size 的最优值，使其从 memory-bound 转为 compute-bound。

```python
import torch
import time

def benchmark_decode_batch(model, batch_sizes, seq_len=2048, num_warmup=5, num_iter=20):
    """
    测量不同 batch size 下的 decode 性能
    找到 memory-bound → compute-bound 的转折点
    """
    results = []
    
    for bs in batch_sizes:
        # 构造输入
        input_ids = torch.randint(0, 32000, (bs, 1), device="cuda")
        # 模拟 KV cache (已有 seq_len 个 token 的 cache)
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                model(input_ids)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iter):
            with torch.no_grad():
                model(input_ids)
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_time_ms = (end - start) / num_iter * 1000
        tokens_per_sec = bs / (avg_time_ms / 1000)
        
        results.append({
            "batch_size": bs,
            "time_ms": avg_time_ms,
            "tokens/s": tokens_per_sec,
            "time_per_token_ms": avg_time_ms / bs
        })
        
        print(f"BS={bs:4d}  Time={avg_time_ms:8.2f}ms  "
              f"Tokens/s={tokens_per_sec:8.0f}  "
              f"Per-token={avg_time_ms/bs:6.2f}ms")
    
    return results

# 预期结果模式:
# BS=1     Time=   42.00ms  Tokens/s=      24  Per-token= 42.00ms  ← memory-bound
# BS=4     Time=   43.50ms  Tokens/s=      92  Per-token= 10.88ms  ← memory-bound (几乎线性扩展)
# BS=16    Time=   48.00ms  Tokens/s=     333  Per-token=  3.00ms  ← 过渡区
# BS=64    Time=   85.00ms  Tokens/s=     753  Per-token=  1.33ms  ← compute-bound (扩展变慢)
# BS=256   Time=  310.00ms  Tokens/s=     826  Per-token=  1.21ms  ← 饱和
```

## 5. Profiling 工具对比

| 工具 | 适用场景 | 粒度 | 开销 | 输出格式 |
|------|---------|------|------|---------|
| nsys | 全局瓶颈分析 | kernel 级 | 低 (~5%) | .nsys-rep (GUI) |
| ncu | 单 kernel 深度分析 | 指令级 | 高 (10-100x) | .ncu-rep (GUI) |
| PyTorch Profiler | Python + CUDA 联合分析 | op 级 | 中 (~10%) | TensorBoard/JSON |
| vLLM 内置 profiling | 推理流程分析 | step 级 | 低 (~5%) | JSON (Chrome trace) |
| DCGM | GPU 硬件监控 | 设备级 | 极低 | Prometheus metrics |
| `torch.cuda.Event` | 精确计时 | 自定义 | 极低 | 数值 |

### 5.1 选择建议

```
问题类型                    → 首选工具
────────────────────────────────────────
"整体哪里慢？"             → nsys
"这个 kernel 为什么慢？"    → ncu
"Python 层面的开销？"       → PyTorch Profiler
"调度和 batching 是否合理？" → vLLM 内置 profiling
"GPU 是否达到硬件极限？"    → ncu Roofline + DCGM
"通信占比多少？"            → nsys (trace NCCL)
```

## 6. 优化验证清单

完成 profiling 后，使用以下清单验证优化效果：

```markdown
## Prefill 优化验证
- [ ] MFU > 50%（H100 上 70B 模型）
- [ ] 启用 FlashAttention v2/v3
- [ ] chunked prefill 配置合理（max_num_batched_tokens）
- [ ] CUDA Graph 已启用（减少 kernel launch overhead）

## Decode 优化验证
- [ ] 带宽利用率 > 70%
- [ ] batch size 在合理范围（不太小导致浪费，不太大导致延迟）
- [ ] KV Cache 使用 FP8 量化（如果支持）
- [ ] 无不必要的 preemption

## 通信优化验证
- [ ] TP GPU 在同一 NVLink 域
- [ ] AllReduce 时间占比 < 15%
- [ ] 通信和计算有重叠（如果框架支持）

## 系统级验证
- [ ] 无 CPU 瓶颈（tokenizer、调度器）
- [ ] NUMA 亲和性正确配置
- [ ] GPU 功率设置为最大性能模式
- [ ] PCIe Gen5 x16（如需 H2D 数据传输）
```

## 7. 总结

LLM 推理性能 profiling 的核心方法论可以归纳为：

1. **先看全局，再看局部**：用 nsys 画出全局画面，识别最大的瓶颈
2. **区分瓶颈类型**：compute-bound 优化算法，memory-bound 优化数据访问，communication-bound 优化拓扑
3. **使用 Roofline Model**：判断当前是否接近理论上限，如果是，则需要换方法（如量化、换硬件）
4. **量化改进效果**：每次优化后重新 profile，确认改善是否符合预期

---

> **延伸阅读：**
> - NVIDIA Nsight Systems 官方文档
> - [FlashAttention Profiling Guide](https://github.com/Dao-AILab/flash-attention)
> - [Roofline Model 原始论文](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworksACM.pdf)
