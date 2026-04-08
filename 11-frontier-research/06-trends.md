# 技术趋势展望

> 从 2026 到 2030——LLM Serving 技术的演进方向与从业者行动指南

## 1. 短期趋势（2026-2027）

短期趋势是指**已有明确技术路线、正在快速落地**的方向。这些技术在 2026 年已有早期实现，预计 2027 年成为标配。

### 1.1 KV Cache 压缩成为标配

**现状（2026 年初）**：
- FP8 KV Cache 已在 vLLM、SGLang 中默认启用
- INT4/FP4 KV Cache 在学术研究中成熟，工程落地中
- DeepSeek-V3 的 MLA 验证了低秩 KV 压缩的实用性

**预期演进**：

```
KV Cache 精度演进:

2023:  FP16 KV Cache (标准)
2024:  FP8 KV Cache (先锋用户采用)
2025:  FP8 标配 + INT4/FP4 实验
2026:  FP8 默认 + INT4 可选 + FP4 实验 (Blackwell)
2027:  INT4/FP4 标配 + 混合精度自适应

显存节省:
  FP16 → FP8:   2× (几乎无损)
  FP16 → INT4:  4× (轻微损失, <1% perplexity)
  FP16 → FP4:   4× (需要更多技术, 但硬件原生支持)

实际影响:
  4× KV Cache 压缩 → 同等显存下 4× 更长上下文
  或 4× 更大 batch size → 4× 更高吞吐
  → 这可能是 2026-2027 年显存效率提升的最大单一因素
```

**技术挑战**：

```python
# 不同层/不同 head 需要不同精度的自适应方案

class AdaptiveKVQuantizer:
    """根据 attention entropy 选择每层的量化精度"""
    
    def quantize_layer(self, layer_idx: int, k: Tensor, v: Tensor):
        entropy = self.compute_attention_entropy(layer_idx)
        
        if entropy < self.threshold_low:
            # 低熵层: attention 集中在少数 token
            # → 可以更激进压缩 (FP4)
            return quantize_fp4(k), quantize_fp4(v)
        elif entropy < self.threshold_high:
            # 中熵层: 使用 INT4
            return quantize_int4(k), quantize_int4(v)
        else:
            # 高熵层: 需要更高精度 (FP8)
            return quantize_fp8(k), quantize_fp8(v)
```

### 1.2 Disaggregated Serving 广泛采用

**现状**：
- Mooncake (Moonshot AI)、DistServe 等系统已在生产环境运行
- vLLM v1 实验性支持 disaggregation
- 大规模 API 提供商（OpenAI、Anthropic、Google）已在内部采用类似架构

**预期演进**：

```
Disaggregated Serving 成熟度路径:

Phase 1 (2024-2025): 研究验证
  - 论文验证可行性 (Splitwise, DistServe)
  - 简单的两层分离 (prefill nodes + decode nodes)
  - KV Cache 通过 RDMA/NVLink 传输

Phase 2 (2025-2026): 早期生产
  - 大厂内部部署
  - vLLM/SGLang 开始支持
  - 简单的静态资源分配

Phase 3 (2026-2027): 广泛采用 ← 我们在这里
  - 开源框架完整支持
  - 动态资源分配 (prefill/decode 节点弹性伸缩)
  - KV Cache Store 成为独立组件
  - 三层分离: Prefill + Decode + KV Store

Phase 4 (2027-2028): 标准架构
  - 所有主流 serving 框架的默认模式
  - 与云基础设施深度集成
  - 专用 KV Cache 传输协议/硬件
```

**关键技术点**：

```
Disaggregation 的核心挑战和解决方向:

1. KV Cache 传输延迟
   当前: RDMA ~100μs for 1MB KV → 长序列需要几十 ms
   方向: CXL 内存池 (μs 级延迟) + 压缩传输 (CacheGen)
   
2. 资源分配比例
   当前: 静态配置 prefill:decode = 1:3
   方向: 动态调整, 基于负载预测 + 排队论模型

3. 故障恢复
   当前: prefill node 故障需要重新计算
   方向: KV Cache 持久化 + 检查点
```

### 1.3 Multi-Token Prediction (MTP) 普及

**现状**：
- DeepSeek-V3 首次在 600B+ 模型中使用 MTP 训练
- MTP 验证了 "训练时多 token 预测 → 推理时内置 draft 能力" 的路线
- 传统投机解码需要额外 draft model，MTP 直接集成到模型中

**预期演进**：

```
MTP 的意义:

传统 Speculative Decoding:
  需要: target model + draft model (额外的模型, 额外的显存)
  优化: 2-3× 加速, 但需要维护两个模型

MTP:
  训练时: 模型学习同时预测 next 2-4 个 token
  推理时: 模型自带 draft 能力, 不需要额外模型
  优化: 2-3× 加速, 零额外开销

→ MTP 可能让投机解码从 "可选优化" 变为 "默认能力"

预期时间线:
2024: DeepSeek-V3 验证 (MTP depth=1)
2025: 更多模型训练时采用 MTP (depth=2-4)
2026: 开源模型标配 MTP
2027: 所有新模型默认包含 MTP 能力
```

### 1.4 超长上下文推理标准化

**现状**：
- Gemini 2.0 支持 2M token 上下文
- Claude 支持 200K token 上下文
- 开源模型（LLaMA-3, Qwen-2.5）支持 128K-1M token

**预期演进**：

```
超长上下文技术栈:

模型层面:
  RoPE 外推 → YaRN, NTK-aware scaling
  长序列训练数据 → 逐步扩展 context window
  架构改进 → Ring Attention, sequence parallelism

Serving 层面:
  KV Cache 管理: 分层存储 (GPU → CPU → SSD)
  Attention 计算: FlashAttention + 分块处理
  调度: 长序列请求的特殊调度策略

当前瓶颈:
  1M token 的 KV Cache ≈ 70B model, FP8:
    per layer: 1M × 8 × 128 × 2 × 1B = 2GB
    80 layers: 2GB × 80 = 160GB → 需要 2+ 个 H100
    → 单个请求就需要整台机器!

解决方向:
  1. KV Cache 压缩: INT4 → 40GB (单 H100 可装)
  2. KV Cache 卸载: hot KV 在 GPU, cold KV 在 CPU
  3. Sparse attention: 不需要 attend 所有 1M token
  4. 序列并行: 将 KV Cache 分布在多 GPU
```

## 2. 中期趋势（2027-2028）

中期趋势是指**技术方向明确、但工程落地尚需突破**的方向。

### 2.1 硬件感知的 KV Cache 管理

```
当前: KV Cache 管理是纯软件逻辑, 不感知底层硬件

未来: 硬件-软件协同设计

1. CXL 内存池 (Compute Express Link)
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  GPU 0  │ │  GPU 1  │ │  GPU 2  │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
   ┌────┴───────────┴───────────┴────┐
   │        CXL Memory Pool          │
   │  (共享 KV Cache 存储, TB 级)    │
   └─────────────────────────────────┘
   
   优势:
   - 延迟: ~200ns (vs CPU DRAM ~100ns, vs GPU HBM ~10ns)
   - 容量: TB 级 (vs GPU HBM 80-192GB)
   - 共享: 多 GPU 可以访问同一块 KV Cache
   → 非常适合 KV Cache 的 "大容量 + 中等带宽" 需求

2. Processing-in-Memory (PIM)
   在 HBM 内部做简单计算 (如 attention score 计算)
   → 减少数据搬运, 加速 memory-bound 的 decode

3. 专用 KV Cache 芯片
   - Groq LPU: 大 SRAM, 适合 KV Cache 存储和计算
   - 类似思路: 定制 KV Cache 存储-计算一体芯片
```

### 2.2 跨模型 KV Cache 共享

```
同族模型之间的 KV Cache 复用:

场景:
  用户先用 small model (7B) 处理长文档
  然后对关键段落切换到 large model (70B) 做深度分析
  
当前: 切换模型 → 完全重新 prefill → 浪费算力
未来: 小模型的 KV Cache 可以被大模型部分复用

技术路线:
1. 蒸馏对齐 KV 空间
   训练小模型时, 让其 KV 表示与大模型对齐
   → 小模型 KV 可以直接给大模型使用 (可能加一个适配层)

2. 适配器网络
   KV_large = Adapter(KV_small)
   适配器很小 (几 MB), 可以快速转换
   → 比完整 prefill 快 10-100×

3. 共享 embedding 空间
   如果模型共享 tokenizer 和 embedding
   → KV Cache 可以更容易复用

限制:
  - 精度损失不可避免, 需要评估可接受范围
  - 只在同族模型间可行 (LLaMA 系列内, DeepSeek 系列内)
  - 不同 attention 配置 (MHA vs GQA vs MLA) 不兼容
```

### 2.3 端云协同推理

```
端云协同的技术架构:

┌──────────────┐          ┌──────────────┐
│   端侧设备   │  ←────→  │   云端集群   │
│ (手机/笔记本) │          │ (GPU 集群)   │
│              │          │              │
│ 小模型 (3-7B)│          │ 大模型 (70B+)│
│ 本地 KV Cache│  传输    │ 共享 KV Cache│
│              │  KV Cache│              │
│ 低延迟推理   │          │ 高质量推理   │
│ 离线可用     │          │ 复杂任务     │
└──────────────┘          └──────────────┘

关键技术挑战:

1. KV Cache 压缩传输
   原始 KV Cache 太大 → 网络传输不现实
   → 需要 10-100× 压缩 + 端侧解压

2. 模型兼容性
   端侧 3B 模型 vs 云端 70B 模型
   → KV 空间维度不同, 需要适配

3. 隐私保护
   用户数据 (KV Cache 包含用户输入信息) 上传到云端
   → 需要 KV Cache 加密/脱敏

4. 网络不稳定
   移动网络可能中断
   → 需要优雅降级 (全部端侧处理)

时间预期:
  2026: 苹果/Google 发布端云协同推理 SDK (概念验证)
  2027: 端云协同在特定场景落地 (写作助手, 代码补全)
  2028: 成为主流推理模式
```

### 2.4 推理专用硬件兴起

```
新型推理硬件:

1. Groq LPU (Language Processing Unit)
   - 特点: 大 SRAM (230MB), 无 HBM, 确定性延迟
   - 优势: decode 极快 (500+ tok/s per user)
   - 劣势: 显存小 (SRAM ≪ HBM), 需要多芯片拼接
   - 适合: latency-sensitive 应用

2. Cerebras WSE (Wafer-Scale Engine)
   - 特点: 整片 wafer 做一个芯片, 40GB SRAM
   - 优势: 超大带宽, 适合大模型
   - 劣势: 极其昂贵, 散热挑战

3. AWS Trainium2 / Inferentia2
   - 特点: 云厂商自研推理芯片
   - 优势: 与云服务深度集成, 性价比高
   - 劣势: 软件生态不如 NVIDIA

4. AMD MI300X / MI350X
   - 特点: 192GB HBM3, 高带宽
   - 优势: 显存大, 适合大模型
   - 劣势: CUDA 生态兼容性挑战 (ROCm)

趋势: 推理市场足够大, 值得专用硬件投资
→ 2027-2028 年推理硬件市场将显著多元化
→ serving 系统需要更好的硬件抽象层
```

## 3. 长期展望（2028-2030+）

### 3.1 新型内存技术

```
CXL (Compute Express Link) 对 Serving 的影响:

CXL 3.0/3.1 (预计 2027-2028 量产):
  - 带宽: ~256 GB/s per link
  - 延迟: ~200-300ns
  - 容量: TB 级内存池
  - 特性: 多设备共享, 动态分配

对 KV Cache 的意义:
  当前: KV Cache 受限于 GPU HBM 容量 (80-192GB)
  CXL: KV Cache 可以扩展到 TB 级
  → 支持 10M+ token 上下文
  → 支持数千个 request 的 KV Cache 同时驻留
  → KV Cache 可以跨 GPU 共享, 无需复制

对 Disaggregated Serving 的意义:
  当前: KV Cache 传输依赖 RDMA (ms 级延迟)
  CXL: KV Cache 存储在共享内存池 (μs 级访问)
  → Prefill node 写入 KV Cache → Decode node 直接读取
  → 无需显式传输!
```

### 3.2 非 Transformer 架构

```
后 Transformer 时代的可能性:

1. SSM 主导
   如果 Mamba-3/4 在质量上追平 Transformer
   → O(1) state 彻底解决长序列问题
   → KV Cache 概念被 "state cache" 取代
   → Serving 系统大幅简化

2. 混合架构标准化
   Transformer 层 (少量, 负责全局注意力) +
   SSM 层 (大量, 负责局部模式) +
   MoE (稀疏激活, 负责容量)
   → 每层不同的缓存和计算策略
   → Hybrid KV Cache Manager 成为标配

3. 全新架构
   - RWKV: RNN 风格, O(1) 推理
   - RetNet: 多尺度保留机制
   - Hyena: 长卷积替代 attention
   → 可能带来 serving 范式的根本改变

预测:
  2026: Transformer 仍然主导 (>90% 市场)
  2028: 混合架构占 20-30%
  2030: Transformer 份额可能降至 50-60%
  → 但 Transformer 不太可能完全消失 (类比 RNN → Transformer 的过程)
```

### 3.3 推理成本的长期趋势

```
推理成本长期预测:

乐观情景 (技术快速进步):
  2026: $1.00/M output tokens (frontier)
  2028: $0.10/M output tokens
  2030: $0.01/M output tokens
  → "推理几乎免费", AI 嵌入所有应用

基准情景 (稳步进步):
  2026: $1.00/M
  2028: $0.30/M
  2030: $0.05/M
  → 推理成本不再是主要考量

悲观情景 (需求增长快于技术进步):
  2026: $1.00/M
  2028: $0.50/M
  2030: $0.20/M
  → 推理成本仍然重要, 但不是瓶颈

影响因素:
  ↓ 降低成本: 硬件进步, 模型效率, 竞争
  ↑ 增加成本: Reasoning 模型 (需要更多计算),
              Agent 普及 (更多推理调用),
              能源价格, 硬件供应链
```

## 4. 对从业者的建议

### 4.1 不同角色的行动指南

#### 对 ML 工程师 / Serving 系统开发者

```
立即行动:
1. 掌握 vLLM / SGLang 的核心架构和源码
   → 这两个框架定义了 serving 的事实标准
   → 理解 PagedAttention, continuous batching, CUDA Graph

2. 学习 FlashInfer 的 API 和设计思想
   → Plan-Run 模式是未来 kernel 接口的方向
   → 理解不同 KV 布局的优劣

3. 关注 Disaggregated Serving 的工程实现
   → 这是 2026-2027 最重要的架构变化
   → 需要理解 KV Cache 传输、资源调度

中期投资:
4. 学习 Triton 编程
   → 自定义 kernel 的能力越来越重要
   → torch.compile 的 Inductor 后端生成 Triton 代码

5. 理解非 NVIDIA 硬件 (AMD ROCm, TPU/XLA)
   → 硬件多元化趋势不可逆
   → 跨硬件的 serving 能力有高价值

6. 关注 SSM / 混合架构的 serving 挑战
   → Jamba, Zamba 等模型的 serving 需要新的抽象
```

#### 对应用开发者 / AI 工程师

```
立即行动:
1. 掌握 Prompt Caching 的使用
   → 最简单的成本优化, 立竿见影
   → Anthropic: cache_control, OpenAI: 自动 caching

2. 使用 Batch API 处理非实时任务
   → 50% 成本折扣, 几乎零工程改动

3. 实现 model routing
   → 简单任务用小模型, 复杂任务用大模型
   → 可以降低 50-70% 总成本

中期关注:
4. 关注端侧推理的进展
   → Apple Intelligence, Google on-device AI
   → 隐私敏感场景的重要方向

5. 评估自部署 vs API 的盈亏平衡
   → 月调用量 >1B tokens 时值得认真评估
   → 考虑总成本 (GPU + 人力 + 运维)

6. 为 Agent 场景优化推理成本
   → Agent 需要大量推理调用
   → 缓存、路由、异步处理都很重要
```

#### 对技术决策者 / 团队负责人

```
战略建议:
1. 不要过早 all-in 自建推理基础设施
   → API 价格持续下降, 自建的成本优势窗口在收窄
   → 除非月调用量 >10B tokens 且有专业团队

2. 关注开源模型的质量曲线
   → LLaMA-3, DeepSeek-V3, Qwen-2.5 质量已经很高
   → 特定场景下开源微调可能优于通用闭源

3. 建设团队的 serving 工程能力
   → 即使用 API, 理解 serving 原理也有助于:
      - 更好地使用 API (prompt 优化, caching)
      - 评估供应商的技术深度
      - 在需要时快速切换到自部署

4. 制定多云 / 多供应商策略
   → 避免单一供应商锁定
   → OpenAI / Anthropic / Google / 开源 灵活切换
```

### 4.2 技能树建议

```
LLM Serving 工程师技能树:

Level 1 (基础):
  □ 理解 Transformer attention 和 KV Cache
  □ 使用 vLLM / SGLang 部署模型
  □ 基本的 GPU 资源规划 (选择 GPU, 估算显存)
  □ 了解量化基础 (FP16/FP8/INT8)

Level 2 (进阶):
  □ 读懂 vLLM 核心源码 (scheduler, block manager)
  □ 理解 FlashAttention 的 IO-aware 优化原理
  □ 配置和调优 continuous batching
  □ 实施 prefix caching 和 prompt 优化

Level 3 (高级):
  □ 实现/修改 attention kernel (Triton/CUDA)
  □ 设计 disaggregated serving 架构
  □ 优化分布式推理 (TP/PP/EP)
  □ 实施端到端的推理性能 profiling

Level 4 (专家):
  □ 贡献 vLLM/SGLang 核心代码
  □ 设计新的调度算法
  □ 跨硬件优化 (NVIDIA + AMD + 加速卡)
  □ 推理系统的架构设计和技术选型
```

## 5. 总结

LLM Serving 领域正处于技术快速迭代的阶段，几个确定性最高的趋势：

1. **KV Cache 压缩标配化**（确定性：极高）——FP8 已是默认，INT4/FP4 即将普及
2. **Disaggregated Serving 广泛采用**（确定性：高）——大规模部署的必然选择
3. **MTP 替代传统投机解码**（确定性：中高）——训练集成 > 后期添加
4. **推理成本持续快速下降**（确定性：高）——多因素驱动，但下降速度可能放缓
5. **硬件多元化**（确定性：中）——AMD、推理专用芯片崛起，但 NVIDIA 仍主导
6. **混合架构模型增多**（确定性：中）——Transformer + SSM 是一个有前途的方向

不确定性最大的方向：
- 非 Transformer 架构能否在质量上追平？
- CXL 内存池何时真正可用于 ML 推理？
- 端侧推理能否达到 "足够好" 的质量？
- Agent 场景的推理需求会如何演变？

**对于从业者最重要的一点：保持学习和实验的节奏。** LLM Serving 的技术半衰期可能只有 12-18 个月——今天的最佳实践可能在一年后被新技术取代。持续跟踪本仓库涵盖的核心主题（KV Cache、调度、分布式、编译优化），在关键技术变化时快速跟进，是保持竞争力的最佳策略。
