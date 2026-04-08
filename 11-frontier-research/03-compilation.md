# 推理编译优化

> 从 eager mode 到编译优化——torch.compile、CUDA Graph 和图优化如何加速 LLM 推理

## 1. 为什么推理需要编译优化？

LLM 推理（特别是 decode 阶段）面临一个核心矛盾：**每个 decode step 的计算量很小（单 token），但 kernel launch 和内存访问的开销很大**。

```
一个 decode step 的开销分解 (LLaMA-70B, batch=1):

1. Kernel launch overhead:
   - 每层 ~10 个 CUDA kernel (attention, FFN, norm, ...)
   - 80 层 × 10 kernel = 800 次 kernel launch
   - 每次 launch ~5-10 μs
   - 总 launch overhead: ~4-8 ms

2. 实际计算时间:
   - 受 memory bandwidth 限制 (weight loading)
   - ~10-15 ms (H100, FP16)

3. Python overhead:
   - PyTorch eager mode 的调度开销
   - ~1-5 ms (取决于模型复杂度)

→ overhead 占总时间的 25-45%!
→ 通过编译优化消除这些 overhead 可以显著提速
```

## 2. CUDA Graph：消除 Kernel Launch Overhead

### 2.1 CUDA Graph 基本原理

CUDA Graph 的核心思想：**录制一次 kernel 序列，之后直接重放，跳过 CPU 端的 dispatch 逻辑**。

```
没有 CUDA Graph:
CPU: [launch K1] → [launch K2] → [launch K3] → ...
GPU:    [idle][K1]    [idle][K2]    [idle][K3]
              ↑            ↑            ↑
          launch delay  launch delay  launch delay

使用 CUDA Graph:
CPU: [replay graph] ─────────────────────────────→
GPU: [K1][K2][K3][K4]...  ← 连续执行, 无间隙
```

### 2.2 CUDA Graph 在 vLLM 中的使用

```python
# vLLM 的 CUDA Graph 集成 (简化版)

class CUDAGraphRunner:
    def __init__(self, model):
        self.model = model
        self.graphs = {}  # batch_size → CUDA Graph
    
    def capture(self, batch_size: int):
        """录制特定 batch_size 的 CUDA Graph"""
        # 创建 dummy 输入
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')
        positions = torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')
        
        # Warmup (确保所有 lazy init 完成)
        self.model(input_ids, positions)
        
        # 录制
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(input_ids, positions)
        
        self.graphs[batch_size] = (graph, input_ids, positions, output)
    
    def replay(self, real_input_ids, real_positions):
        """重放 CUDA Graph"""
        batch_size = real_input_ids.shape[0]
        graph, input_buf, pos_buf, output_buf = self.graphs[batch_size]
        
        # 将实际数据复制到录制时的 buffer
        input_buf.copy_(real_input_ids)
        pos_buf.copy_(real_positions)
        
        # 重放
        graph.replay()
        
        return output_buf.clone()
```

### 2.3 动态 Shape 的挑战

CUDA Graph 的最大限制：**录制时的 tensor shape 在重放时不能变**。这与 LLM serving 的动态特性冲突：

```
动态维度:
1. Batch size: 每个 step 可能不同 (有 request 加入/完成)
2. Sequence length: 不同 request 的 seq_len 不同 (影响 KV Cache 访问)
3. Vocabulary size: 通常固定, 不是问题

解决方案:

方案 1: 预录制多个 batch size 的 graph (vLLM 采用)
  batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
  for bs in batch_sizes:
      capture_graph(bs)
  
  # 运行时 padding 到最近的预录制 batch size
  actual_bs = 13
  padded_bs = 16  # 向上取整到最近的预录制大小
  # 浪费: 3/16 = 18.75% 计算
  # 但消除了 launch overhead, 总体仍然更快

方案 2: CUDA Graph + 动态 shape 支持 (实验性)
  # PyTorch 2.4+ 开始支持部分动态 shape
  # 但限制较多, 不适用于所有 kernel

方案 3: FlashInfer 的 Plan-Run 分离
  # Plan 阶段 (CPU, 可以动态): 处理变长 shape
  # Run 阶段 (GPU, 可以 graph): 固定计算模式
  # → 结合 CUDA Graph 和动态 shape 的优点
```

### 2.4 CUDA Graph 的显存开销

```python
# CUDA Graph 需要额外显存:

# 1. 每个 graph 的输入/输出 buffer
#    不能释放, 因为重放时直接使用这些 buffer
#    → 预录制 9 个 batch size 意味着 9 份 buffer

# 2. Workspace 内存
#    某些 kernel 需要临时 workspace
#    在 graph 中, workspace 不能动态分配, 必须预留

# 3. 实际开销估算 (LLaMA-70B, FP16):
#    每个 graph: ~50-200 MB (取决于 batch size)
#    9 个 graph: ~0.5-1.5 GB
#    占总显存 (80GB) 的 1-2%
#    → 相对于性能收益, 这个开销是可以接受的
```

## 3. torch.compile 在推理中的应用

### 3.1 torch.compile 基本原理

PyTorch 2.0 引入的 `torch.compile` 通过以下步骤优化模型：

```
torch.compile 流程:

1. Dynamo (图捕获):
   Python bytecode → FX Graph (中间表示)
   - 追踪 Python 代码执行
   - 处理控制流 (guard + graph break)
   - 输出: 纯 tensor 操作的计算图

2. AOTAutograd (前向/反向分离):
   - 推理时: 只关注前向图
   - 死码消除、常量折叠

3. Inductor (代码生成):
   FX Graph → Triton/CUDA kernels
   - Op fusion: 将多个小 op 合并为一个 kernel
   - 内存优化: 减少中间 tensor 分配
   - 代码生成: 输出 Triton kernel 代码
```

### 3.2 Op Fusion：最大的收益来源

```python
# 未融合的 LayerNorm + Linear:
# Kernel 1: LayerNorm
x_norm = layer_norm(x)  # 读 x, 写 x_norm (显存 IO)
# Kernel 2: Linear  
out = linear(x_norm)    # 读 x_norm, 写 out (显存 IO)
# → x_norm 被写到显存再读回, 浪费带宽

# 融合后:
# Single Kernel: LayerNorm + Linear
out = fused_layernorm_linear(x)  # 读 x, 写 out
# → x_norm 留在寄存器/shared memory, 不写回显存
# → 节省一次显存读写 (对 memory-bound 操作, 这是显著加速)

# 常见的可融合 pattern:
# 1. LayerNorm + QKV projection
# 2. Attention output projection + residual add
# 3. FFN gate + up projection + activation
# 4. RMSNorm + any linear
```

### 3.3 vLLM 中的 torch.compile 集成

```python
# vLLM 对 torch.compile 的使用方式:

# 方式 1: 编译整个模型 (experimental)
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead",  # 最大化减少 overhead
    fullgraph=False,         # 允许 graph break (更兼容)
)

# 方式 2: 编译特定的子模块 (更稳定)
# 只编译最受益的部分, 避免 graph break 问题
model.mlp = torch.compile(model.mlp)
model.self_attn = torch.compile(model.self_attn)

# 方式 3: 自定义 fusion pass
# vLLM 注册了自定义的 fusion pattern
@torch.compile
def fused_moe(hidden_states, gate, up, down, topk_weights, topk_ids):
    """融合的 MoE 前向计算"""
    # gate_up = hidden_states @ concat(gate, up)
    # activated = silu(gate_out) * up_out
    # down_out = activated @ down
    # output = scatter_add(down_out, topk_weights)
    ...
```

### 3.4 Compilation Modes 的选择

```python
# torch.compile 的不同模式:

# 1. mode="default" — 平衡编译时间和性能
#    适合: 开发调试

# 2. mode="reduce-overhead" — 最小化 Python/launch overhead
#    原理: 使用 CUDA Graph + Triton kernels
#    适合: 推理 serving (decode 阶段)
#    缺点: 首次编译慢, 显存占用增加

# 3. mode="max-autotune" — 搜索最优实现
#    原理: 尝试多种 kernel 实现, benchmark 选最优
#    适合: 离线编译, 发布前优化
#    缺点: 编译时间非常长 (几十分钟到几小时)

# 推荐策略:
# 开发时: mode="default"
# 生产部署: mode="max-autotune" 预编译 + mode="reduce-overhead" 运行
```

## 4. TensorRT-LLM vs vLLM 编译策略对比

### 4.1 设计哲学对比

```
TensorRT-LLM (NVIDIA):
  ┌──────────────┐
  │ 模型定义     │  ← 需要用 TRT-LLM 的 API 重写模型
  │ (TRT-LLM API)│
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ TensorRT 优化│  ← 静态图优化 (AOT)
  │ (Layer/Graph │     - kernel selection
  │  Fusion,     │     - precision calibration
  │  Quantization│     - memory planning
  │  Calibration)│
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ TRT Engine   │  ← 二进制引擎文件, 部署到目标 GPU
  │ (binary)     │
  └──────────────┘

vLLM + torch.compile:
  ┌──────────────┐
  │ 标准 PyTorch │  ← 直接使用 HuggingFace 模型
  │ 模型代码     │
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ torch.compile│  ← JIT 编译, 运行时优化
  │ (Dynamo +    │     - 自动图捕获
  │  Inductor)   │     - op fusion
  └──────┬───────┘     - Triton codegen
         ▼
  ┌──────────────┐
  │ 编译缓存     │  ← 首次运行后缓存编译结果
  │ (PYC files)  │
  └──────────────┘
```

### 4.2 详细对比

| 维度 | TensorRT-LLM | vLLM + torch.compile |
|------|-------------|---------------------|
| **模型支持** | 需要手动适配每个模型 | 自动支持 HF 模型 |
| **新模型上线速度** | 慢（需要 NVIDIA 团队适配） | 快（社区 PR 即可） |
| **编译时间** | 长（分钟到小时） | 中等（秒到分钟） |
| **优化深度** | 深（TRT 有十年优化积累） | 中等（但在快速进步） |
| **硬件支持** | NVIDIA only | 多后端（计划中） |
| **FP8/INT8 量化** | 成熟（calibration 工具链完善） | 在跟进 |
| **CUDA Graph** | 内置 | 通过 torch.compile 支持 |
| **灵活性** | 低（修改需要重新编译整个引擎） | 高（可以部分编译） |
| **调试难度** | 高（黑盒引擎） | 中（Triton 代码可读） |

### 4.3 性能对比（实测数据参考）

```
H100 80GB, LLaMA-2 70B, FP16:

                    | TensorRT-LLM | vLLM (no compile) | vLLM (compiled) |
Prefill (2048 tok)  |   ~22 ms     |     ~28 ms        |    ~24 ms       |
Decode (bs=32)      |   ~18 ms     |     ~25 ms        |    ~20 ms       |
Decode (bs=128)     |   ~35 ms     |     ~48 ms        |    ~40 ms       |
Peak throughput     |   ~4200 t/s  |     ~3200 t/s     |    ~3800 t/s    |

注: 以上数据为估算, 具体值取决于配置和版本。
TensorRT-LLM 在 pure throughput 上仍有优势,
但 vLLM + compile 正在缩小差距, 且灵活性远超 TRT-LLM。
```

## 5. 图优化 Pass 详解

### 5.1 常量折叠 (Constant Folding)

```python
# 编译时计算可以提前确定的值

# 优化前:
scale = 1.0 / math.sqrt(head_dim)  # head_dim=128, scale=0.0884
output = attn_weights * scale

# 优化后:
output = attn_weights * 0.0884  # 直接用常量
```

### 5.2 Op Fusion Pass

```python
# torch.compile Inductor 的融合规则:

# Rule 1: Pointwise + Pointwise → 单 kernel
#   add + relu → fused_add_relu
#   mul + add + gelu → fused_mul_add_gelu

# Rule 2: Reduction + Pointwise → 单 kernel
#   layernorm + linear 的 norm 部分可融合
#   softmax + dropout 可融合

# Rule 3: GEMM + Pointwise → 单 kernel (Epilogue fusion)
#   linear + bias_add + activation → 单个 GEMM kernel with epilogue
#   这是 TensorRT 的经典优化, Inductor 也支持

# 实际效果 (LLaMA 单层):
# 优化前: ~12 个 kernel
# 优化后: ~5-6 个 kernel
# → 减少 50% kernel launch
```

### 5.3 Memory Planning (内存规划)

```python
# Inductor 的内存优化:

# 1. Buffer Reuse (缓冲区复用)
#    生命周期不重叠的 tensor 共享同一块内存
#    → 减少显存占用

# 2. In-place Operations (原地操作)
#    当输入 tensor 后续不再使用时, 直接在原地修改
#    → 减少内存分配

# 3. Memory Layout Optimization (内存布局优化)
#    选择对后续 kernel 最优的 tensor layout (contiguous, channels_last, etc.)
#    → 减少 transpose/permute 开销
```

### 5.4 自定义 Fusion Pass (vLLM 示例)

```python
# vLLM 注册了专门的 fusion pattern:

# 1. Fused RMSNorm + QKV Projection
#    RMSNorm(x) → Q, K, V = split(x @ W_qkv)
#    → 单个 kernel: 读 x 一次, 写 Q/K/V 一次

# 2. Fused MoE (Mixture of Experts)
#    gate_logits = x @ W_gate
#    topk_weights, topk_ids = topk(softmax(gate_logits))
#    expert_out = sum(expert_i(x) * weight_i for i in topk)
#    → 高度优化的 fused MoE kernel

# 3. Fused Rotary Embedding
#    标准实现需要 sin/cos 查表 + 复数乘法, 多个小 kernel
#    → 融合为单个 kernel
```

## 6. XLA 和其他编译框架

### 6.1 XLA (Accelerated Linear Algebra)

XLA 是 Google 开发的编译器，主要用于 TPU 和 JAX：

```
XLA 在 LLM 推理中的角色:

1. TPU 上的主要编译器
   - Google 的 PaLM/Gemini serving 使用 XLA
   - TPU 不支持 CUDA, 只能用 XLA

2. JAX 后端
   - JAX 的 jit() 编译通过 XLA
   - 一些 serving 框架基于 JAX (如 JetStream)

3. PyTorch/XLA
   - 通过 torch_xla 在 TPU 上跑 PyTorch 模型
   - vLLM 对 TPU 的实验性支持使用此路径

XLA vs Inductor:
  XLA: 更激进的全局优化, 适合 TPU 的 systolic array
  Inductor: 更灵活, 适合 GPU 的 CUDA/Triton
```

### 6.2 Triton Compiler

Triton 是 OpenAI 开发的 GPU kernel 编程语言和编译器：

```python
# Triton 在 LLM serving 中的角色:

# 1. torch.compile 的后端
#    Inductor 生成 Triton 代码 → Triton 编译为 PTX → GPU 执行

# 2. 自定义 kernel 的首选语言
#    相比 CUDA, Triton 更易编写, 且自动处理:
#    - Thread block 调度
#    - 共享内存管理
#    - 内存合并访问
#    
#    FlashAttention 的 Triton 实现只有 ~200 行 (vs CUDA ~2000 行)

# 3. 示例: Triton 实现的 fused RMSNorm
@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * N + tl.arange(0, BLOCK_SIZE))
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE))
    
    # RMS
    rms = tl.sqrt(tl.sum(x * x) / N + eps)
    
    # Normalize and scale
    out = x / rms * w
    tl.store(out_ptr + row * N + tl.arange(0, BLOCK_SIZE), out)
```

### 6.3 编译框架总结

| 框架 | 主要用途 | 硬件 | 在 serving 中的角色 |
|------|---------|------|-------------------|
| torch.compile (Inductor) | PyTorch JIT 编译 | GPU (CUDA/ROCm) | vLLM 主要编译路径 |
| TensorRT | 静态图推理优化 | NVIDIA GPU | TensorRT-LLM 的核心 |
| XLA | TPU/GPU 编译 | TPU, GPU | Google serving, JAX 框架 |
| Triton | GPU kernel 编写 | NVIDIA/AMD GPU | 自定义 kernel, Inductor 后端 |
| MLIR | 编译器基础设施 | 多种 | IREE, 学术研究 |
| ONNX Runtime | 跨平台推理 | 多种 | 小模型推理, 边缘部署 |

## 7. CUDA Graph 与 torch.compile 的协同

### 7.1 reduce-overhead mode 的原理

```
torch.compile(mode="reduce-overhead") 的工作原理:

1. Dynamo 捕获计算图 → FX Graph
2. Inductor 将 FX Graph 编译为 Triton kernels
3. 将编译后的 kernel 序列录制为 CUDA Graph
4. 运行时直接重放 CUDA Graph

→ 同时获得:
   - Op fusion 的收益 (Inductor)
   - Launch overhead 消除的收益 (CUDA Graph)
   - 两者叠加, 效果 > 单独使用任何一个
```

### 7.2 实际整合

```python
# vLLM 中 torch.compile + CUDA Graph 的工作流程:

# 1. 模型加载时
model = load_model(model_path)
compiled_model = torch.compile(model, mode="reduce-overhead")

# 2. Warmup 阶段 (服务启动时)
for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
    dummy_input = create_dummy_input(batch_size)
    compiled_model(dummy_input)  # 触发编译 + CUDA Graph 录制
    # 编译结果缓存到磁盘, 下次启动直接复用

# 3. 推理阶段
# torch.compile 自动管理 CUDA Graph 的重放
# 如果输入 shape 匹配已有 graph → 重放
# 如果不匹配 → 触发新的编译 (graph break, 性能下降)
# → 所以 warmup 阶段要覆盖所有可能的 batch size
```

## 8. 编译优化的挑战和限制

### 8.1 Graph Break 问题

```python
# torch.compile 的 "graph break":
# 当遇到无法追踪的 Python 操作时, 图被打断

# 常见 graph break 原因:
# 1. 数据依赖的控制流
if x.item() > 0:      # ← graph break! (需要读取 GPU 数据到 CPU)
    return x * 2
else:
    return x * 3

# 2. 动态 shape 操作
y = x[:dynamic_len]    # ← 可能 graph break

# 3. 不支持的操作
np_array = x.numpy()   # ← graph break! (跳出 PyTorch)

# 4. 自定义 CUDA kernel
custom_cuda_op(x)      # ← 可能 graph break (除非正确注册)

# 影响: 每个 graph break 都会引入 Python overhead
# → 在 LLM 模型中, 要确保主要计算路径无 graph break
```

### 8.2 编译时间

```
不同模型的编译时间 (torch.compile, 首次编译):

模型           | default mode | max-autotune |
LLaMA-7B      | ~30s        | ~5min        |
LLaMA-70B     | ~2min       | ~30min       |
Mixtral-8x7B  | ~3min       | ~45min       |

缓解策略:
1. 编译缓存: TORCHINDUCTOR_CACHE_DIR 缓存编译结果
2. 预编译: 部署前在同型号 GPU 上预编译
3. 部分编译: 只编译 hot path (decoder layers), 跳过 embedding 等
```

### 8.3 精度问题

```python
# 编译优化可能影响数值精度:

# 1. Op fusion 改变计算顺序
#    未融合: tmp = a + b; out = tmp * c
#    融合后: out = (a + b) * c  (FMA instruction)
#    → FMA 的精度略有不同

# 2. Triton kernel 的默认精度
#    Triton 默认使用 tf32 (TensorFloat-32) 做矩阵乘
#    → 比 FP32 精度略低, 但通常不影响 LLM 输出

# 3. 验证方法:
compiled_output = compiled_model(input)
eager_output = model(input)
max_diff = (compiled_output - eager_output).abs().max()
# 通常 max_diff < 1e-3 (FP16) 或 < 1e-5 (FP32) 是可接受的
```

## 9. 实践建议

### 9.1 部署清单

```
生产环境使用编译优化的检查清单:

□ 验证模型兼容性
  - 确认无 graph break (TORCH_LOGS=graph_breaks 检查)
  - 确认自定义 op 正确注册

□ 选择编译策略
  - 纯 CUDA Graph (简单, 可靠): 适合追求稳定性
  - torch.compile (more fusion): 适合追求极致性能
  - 两者结合 (reduce-overhead mode): 最大收益

□ Warmup 覆盖
  - 预热所有可能的 batch size
  - 预热 prefill 和 decode 两种模式
  - 将编译缓存持久化到磁盘

□ 精度验证
  - 对比 eager vs compiled 的输出差异
  - 端到端评估模型质量 (perplexity, benchmark)

□ 性能验证
  - Benchmark: 编译前后的 latency / throughput
  - 监控: 运行时有无意外 graph break / recompilation
```

### 9.2 性能提升期望

```
不同优化组合的典型提升 (decode 阶段):

                          | 相对 eager mode |
CUDA Graph only           | 1.2-1.4×        |
torch.compile only        | 1.1-1.3×        |
CUDA Graph + compile      | 1.3-1.6×        |
+ FP8 quantization        | 1.8-2.5×        |
+ FlashAttention-3        | 2.0-3.0×        |

→ 编译优化是 "low-hanging fruit", 与量化和算子优化叠加效果最佳
```

## 10. 小结

推理编译优化是缩小 "理论峰值性能" 与 "实际达到性能" 之间差距的关键手段：

1. **CUDA Graph**：消除 kernel launch overhead，对 decode 阶段尤为有效；但需要处理动态 shape
2. **torch.compile**：通过 op fusion 减少显存带宽消耗，自动生成优化 kernel；但面临 graph break 和编译时间的挑战
3. **TensorRT-LLM**：更深度的静态优化，性能天花板更高；但灵活性和模型支持度不如 vLLM
4. **编译与其他优化叠加**：编译优化 + 量化 + FlashAttention 的组合可以实现 2-3× 的综合加速

对于大多数团队，建议从 **vLLM + CUDA Graph** 开始，逐步引入 `torch.compile`，在追求极致性能时考虑 TensorRT-LLM。
