# HANDS_ON.md — 13 章实操路径与速通计划

> 把本仓库 7000+ 行 exercises 串成一条**可执行**路径。STUDY_GUIDE 告诉你"读什么"，本文档告诉你"在什么硬件上怎么跑、要多久、怎么自查"。

**如何使用本文档**

1. 先读 [STUDY_GUIDE.md](STUDY_GUIDE.md) 第 1-2 节，判断你的 Level
2. 先过 [版本基线与 Freshness Gate](resources/version-baseline.md)，确认 vLLM/SGLang/API 文档没有过期
3. 读本文档 [Section 0](#0) 确认硬件路径
4. 按 [Section 3](#3-13-checklist) 逐章学，每章用配套自查题验收
5. 时间宽裕？跟 [Section 5](#5-5) 走 5 周高强度路线

---

## 0. 你的硬件画像与预算分配

### 0.1 本文档面向的资源画像（默认假设）

- 本地：1 张 RTX 3070 (8GB)
- 云：Azure $150 预算
- API：愿意花少量钱调 Anthropic / OpenAI / DeepSeek API

如果你的资源不同，把每个 checklist 项的硬件标签作为参考即可。

### 0.2 3070 (8GB) 能做什么 / 不能做什么

**能跑**
- ✅ 所有纯 Python 模拟器、计算器（Ch01/03/04/05/06/07/08/09 的练习 1）
- ✅ 所有 API 实验（Ch02 prompt caching、Ch12 structured output、Ch13 vision）
- ✅ vLLM + 小模型：Qwen2.5-1.5B、SmolLM2-1.7B、Moondream-2B、Qwen2-VL-2B
- ✅ vLLM + AWQ INT4 量化的 7B（紧，可能要调 `max_model_len`）
- ✅ 阅读 vLLM 源码 + 加 print 跟踪（不需要满血推理）

**跑不动**
- ❌ 任何 TP/PP/EP 多卡实验
- ❌ FP16 / BF16 的 7B 满血模型
- ❌ Prefill-Decode 分离架构（至少要 2 卡）
- ❌ 70B 级别任何精度

### 0.3 Azure $150 建议分配

> 假设按 Azure 公开 pay-as-you-go 价格，**Spot 实例可节省 60-90%，强烈推荐**。

| 项目 | 预算 | 用途 |
|------|-----|------|
| API 实验 | $20-30 | Ch02 prompt caching + Ch12 SO 对照 + Ch13 vision |
| L4 / L40S 单卡 | $50-70 | Ch04/07/08/10/12 的 7B 实验 |
| A100 集中日 | $30-50 | Ch05/09 的多卡 / 分离架构实验 |
| 缓冲 | $10-20 | 重跑、debug、quota 等待时挂着不停 |

### 0.4 三档选择

挑一档作为你的执行模式：

- **A 档（全本地）**：不花云钱，只做 🟢 项 + API 实验。覆盖 60-70% 内容
- **B 档（标准）**：按上表分配 $150，覆盖 90%+ 内容 ← 推荐
- **C 档（深度）**：超出 $150 自费，把 Ch05 disagg、Ch09 多维并行全部实测一遍

后文 checklist 标 `[必] / [荐] / [选]`，A 档可只做"必 + 部分荐"，B 档全做"必 + 荐"，C 档全做。

---

## 1. 四个 Track

每个实操项首字符标签如下：

| 标签 | 含义 | 典型场景 |
|------|------|----------|
| 🟢 | 本地 Track | 3070 + Python + API |
| 🟡 | L4 上云 Track | 24GB 单卡跑 7B |
| 🔴 | A100 集中 Track | 多卡 / 大模型，建议攒一天集中做 |
| 📖 | 纯阅读 Track | 论文 / 源码 / 写笔记 |

**🔴 集中策略**：A100 别零碎用。把所有需要 A100 的练习集中到 1 天（Week 4 的 D3），上午起 VM、配环境、跑 baseline，下午跑实验，傍晚画图、写笔记，晚上 deallocate。$30-50 一次性花完。

---

## 2. 上云操作手册

### 2.1 Azure GPU quota 申请

NCas T4 v3 / NC A100 v4 默认 quota 是 0，必须先申请。

1. Azure Portal → Subscriptions → Usage + quotas
2. 搜 "NC" 系列，找 "Standard NCasT4_v3 Family vCPUs"（L4 实际在另一个家族，按 region 查）
3. New Request，写明用途："Learning project for LLM inference benchmarking, single VM, will deallocate after each session, expected usage < 50 hours total"
4. 通常 1-2 个工作日批准

**Region 选择**：East US / West US 2 资源最足，国内访问加 ~150ms。如果你在中国大陆且对延迟敏感，可以选 Japan East 但 quota 更紧张。

### 2.2 推荐 VM 系列

| 系列 | GPU | 单卡显存 | 用途 |
|------|-----|---------|------|
| Standard_NV36ads_A10_v5 | A10 | 24 GB | 替代 L4，价格相近 |
| Standard_NC24ads_A100_v4 | 1× A100 | 80 GB | 单 A100 大模型 |
| Standard_NC48ads_A100_v4 | 2× A100 | 80 GB × 2 | TP=2 实验 |
| Standard_NC96ads_A100_v4 | 4× A100 | 80 GB × 4 | TP=4 / disagg 实验 |

> Azure 上 L4 (NV* L4 v5 系列) 在某些 region 可用，价格更低，但 quota 难申请。**默认用 A10 替代 L4**。

### 2.3 标准 startup script

VM 创建时选 "Ubuntu 22.04 LTS, Gen2"，并附加以下 cloud-init / startup script：

```bash
#!/bin/bash
set -e

# CUDA driver (Azure GPU image 通常已装，确认一下)
nvidia-smi || { echo "CUDA driver missing"; exit 1; }

# Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda init bash

# Python env
conda create -n vllm python=3.11 -y
conda activate vllm

# vLLM 及周边
pip install vllm openai aiohttp matplotlib pandas jupyter
pip install transformers accelerate bitsandbytes

# 预下载常用模型（HF cache 默认在 ~/.cache/huggingface）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ

# Jupyter remote
nohup jupyter notebook --no-browser --port=8888 --ip=0.0.0.0 \
  --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &
```

通过 SSH port forwarding 用本地浏览器访问 Jupyter：

```bash
ssh -L 8888:localhost:8888 azureuser@<vm-ip>
```

### 2.4 成本控制清单（重要）

- 用完一定 **deallocate**，不是 `stop`。`stop` 在 Azure 上仍计 VM 钱，只有 **deallocate (`az vm deallocate`)** 才真正停计费
- 启动前用 `az vm list-sizes --location eastus` 看实际价格（按需 vs spot）
- 装一个本地 alias：
  ```bash
  alias gpu-up='az vm start -g myrg -n mygpuvm && az vm show -d -g myrg -n mygpuvm --query publicIps -o tsv'
  alias gpu-down='az vm deallocate -g myrg -n mygpuvm'
  ```
- **Spot 实例 (`--priority Spot`)** 可便宜 60-90%，缺点是可能被抢占。学习场景非常合适
- 跑完一个实验立刻 `gpu-down`，**不要边吃饭边挂着**——一顿饭就是 $5-10
- 在 Azure Cost Management 里设置 **预算警报**（$150 触发 80% 警告）

### 2.5 一份起停模板

```bash
# 起 VM 跑实验的标准流程
gpu-up              # 启动 VM
ssh azureuser@<ip>  # SSH 进去
tmux new -s exp     # tmux 开 session（防 SSH 断线丢进度）
conda activate vllm
# ... 做实验 ...
# 退出 tmux: Ctrl+B 然后 D
exit
gpu-down            # 立刻 deallocate
```

---

## 3. 13 章实操 Checklist

> 时间估算原则：阅读按 60-80 字/分钟、实操按"装环境 + 写代码 + 调通 + 验证"全程估，可能高估 20%。

---

### Ch01 KV Cache 深度剖析（🟢 · 阅读 4h + 实操 3h）

#### 必做

- [ ] [🟢] 读完 `01-memory-layout.md` → `03-memory-calculation.md`（~3h）
- [ ] [🟢] 完成 `exercises.md` 练习 1（Qwen-72B KV Cache 精确计算，~1h）
  - 先手算，再用 `python scripts/kv_cache_calculator.py ...` 对照参考输出
- [ ] [🟢] 完成 `exercises.md` 练习 2（写 Python 函数封装计算）
- [ ] [🟢] 完成 `exercises.md` 练习 3（prefill vs decode AI 对比）

#### 自查

1. 给定模型层数 L、KV heads、head_dim、dtype、seq_len，5 分钟内手算单 token 与单序列 KV Cache 大小
2. 解释为什么 GQA-8 相比 MHA 节省 8 倍 KV Cache，但模型质量损失很小
3. 解释为什么 prefill 是 compute-bound、decode 是 memory-bound（用 AI 公式推）

---

### Ch02 前缀缓存与 Prompt Caching（🟢 · 阅读 4h + 实操 3h）

#### 必做

- [ ] [🟢] 读完 `01-principles.md` → `04-api-caching.md`
- [ ] [🟢] 完成 exercises 练习 1（Anthropic Prompt Caching 行为观测，~1.5h，需 API key）
  - 重点观察 `cache_creation_input_tokens` 和 `cache_read_input_tokens` 的变化
  - 实测 5min TTL 失效行为
- [ ] [🟢] 完成 exercises 练习 2（OpenAI / DeepSeek 隐式 caching 对照）

#### 推荐

- [ ] [🟢] 读 vLLM `vllm/v1/core/kv_cache_manager.py` 中 APC 命中逻辑
- [ ] [🟢] 用 Qwen2.5-1.5B 在 3070 本地起 vLLM，发重复前缀请求，观察 `prefix_cache_hit_rate` metric

#### 自查

1. 解释 Anthropic 的 prompt cache vs vLLM 内部 APC 的差异（API 层 vs 引擎层）
2. RadixAttention 在什么场景下比简单 hash-based APC 收益更大？
3. 为什么 cache hit 不能 100% 节省 prefill 时间？（提示：cached KV 还要从某处读）

---

### Ch03 KV Cache 压缩（🟢🟡 · 阅读 5h + 实操 3h）

#### 必做

- [ ] [🟢] 读完 4 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（KV Cache 显存计算器，覆盖 MHA/GQA/MQA/MLA）
- [ ] [🟢] 完成 exercises 练习 2（MLA 57x 压缩比推导验证）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 3-4（量化误差分析、选择性 cache）
- [ ] [🟡 A10] 用 vLLM 跑 `--kv-cache-dtype fp8`，对比 BF16 KV 的吞吐量与质量

#### 自查

1. 用一张表对比 MHA / GQA / MQA / MLA 的：KV size、质量损失、实现复杂度
2. 推导 DeepSeek-V2 MLA 的压缩比："57x vs MHA"具体怎么算出来的
3. FP8 KV cache 在什么 workload 下质量损失明显？

---

### Ch04 PagedAttention（🟢🟡 · 阅读 5h + 实操 5h）

#### 必做

- [ ] [🟢] 读完 4 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（Block Pool + Block Table 模拟器，~2h）
- [ ] [📖] 走读 vLLM `vllm/v1/core/block_pool.py` 与 `kv_cache_manager.py`，画出 block 分配/释放流程图

#### 推荐

- [ ] [🟢] 完成 exercises 练习 2-3（preemption 模拟）
- [ ] [🟢] 本地用 Qwen2.5-1.5B + vLLM，加 print 在 `block_pool.py` 关键函数，跑一次推理观察 block 流动
- [ ] [🟡 A10] 用 7B 模型 + 高并发触发 preemption（构造多个长 prompt + 短 context window），观察 `num_preempted_requests` 指标

#### 选做

- [ ] [🟢] 完成 exercises 练习 4-5（fragmentation 分析）

#### 自查

1. 描述 num_blocks=1000、block_size=16 时，一个 prompt=2000 tokens 的请求如何分配 block
2. preemption 的 swap vs recompute 策略，各自适用什么场景
3. 为什么 PagedAttention 相比 contiguous KV cache 不降低性能（gather 操作 vs 连续读）

---

### Ch05 Prefill-Decode 分离架构（🟢🔴 · 阅读 4h + 实操 4h）

#### 必做

- [ ] [🟢] 读完 5 篇子文档（重点：when-to-use）
- [ ] [🟢] 完成 exercises 练习 1（KV Cache 传输开销计算器）
  - 用 `scripts/kv_transfer_calculator.py --kv-shard-factor ...` 区分 full KV 与单 shard 传输
- [ ] [🟢] 完成 exercises 练习 2-3（分离 break-even point 计算）

#### 推荐

- [ ] [📖] 走读 vLLM disagg 实现（`vllm/distributed/kv_transfer/`），理解 KV transfer 的实际带宽
- [ ] [🔴 A100×2] **集中日**：用 2 张 A100 起 disagg setup，跑同一个 workload 对比聚合 vs 分离的 TTFT/TBT/throughput

#### 自查

1. 给定网络带宽、模型大小、平均 prompt 长度，5 分钟算出分离架构的 break-even 并发数
2. 为什么 GQA 模型比 MHA 模型更适合做分离？
3. 列出 3 个不适合用分离的场景

---

### Ch06 KV Cache Offloading（🟢🟡 · 阅读 4h + 实操 4h）

#### 必做

- [ ] [🟢] 读完 5 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（SimpleOffloadManager，pinned memory + async copy）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 2-3（offload vs recompute 决策、LMCache 配置阅读）
- [ ] [🟡 A10] 用 vLLM `--swap-space 16` 启用 CPU offloading，构造超过显存的 cache 压力，观察 `cpu_cache_usage` 指标

#### 自查

1. 估算 GPU→CPU pinned memory 的传输带宽（PCIe 4.0 ~32 GB/s 上限）
2. offload 一个 token 的 KV 与重算它的 prefill，哪个更便宜？给出决策公式
3. SSD 作为 tier 3 在什么场景才有意义？

---

### Ch07 投机解码（🟢🟡 · 阅读 6h + 实操 4h）

#### 必做

- [ ] [🟢] 读完 6 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（Rejection Sampling 模拟器，~2h）—— 验证 $E[\text{accepted}] = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$
  - 用 `scripts/speculative_decoding_simulator.py` 对照理论值和模拟值
- [ ] [🟢] 完成 exercises 练习 2（EAGLE / Medusa / MTP 架构对比表）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 3-4（draft length γ 调参曲线）
- [ ] [🟡 A10] 用 vLLM `--speculative-model` 跑 Qwen2.5-7B + draft model，测同一 prompt 的吞吐变化，观察 `accept_rate` metric

#### 自查

1. 不看公式，推导投机解码无损性证明的核心直觉（accept-or-resample）
2. EAGLE-2 相比 EAGLE-1 改进了哪一步？为什么效果好
3. 给定 α=0.7、γ=5，期望接受 token 数是多少？为什么 γ 不是越大越好

---

### Ch08 调度与批处理（🟢🟡 · 阅读 5h + 实操 5h）

#### 必做

- [ ] [🟢] 读完 5 篇子文档
- [ ] [📖] 走读 vLLM `vllm/v1/core/sched/scheduler.py` 的 `schedule()` 方法
- [ ] [🟢] 完成 exercises 练习 1（chunked prefill 数学分析）
  - 用 `scripts/batching_throughput_estimator.py` 先估算 `L_max / L_mean`

#### 推荐

- [ ] [🟡 A10] 完成 exercises 练习 2-3：在真实 vLLM 上调 `--max-num-batched-tokens` 和 `--max-num-seqs`，画 TTFT vs TBT vs throughput 三角图
- [ ] [🟡 A10] 完成 exercises 练习 4（优先级 / 公平性策略对比）

#### 自查

1. 描述 vLLM scheduler 的 `schedule()` 做了哪 5-7 步
2. chunked prefill `chunk_size` 增大 → TTFT/TBT/throughput 各往哪边走？
3. continuous batching 相比 static batching 提升 2-8x 吞吐的根本原因？

---

### Ch09 分布式推理（🟢🔴 · 阅读 6h + 实操 5h）

#### 必做

- [ ] [🟢] 读完 6 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（TP 通信开销分析与建模）
  - 用 `scripts/tp_comm_estimator.py` 对比 H100/NVLink 与 PCIe-Gen4 的通信占比
- [ ] [🟢] 完成 exercises 练习 2（PP bubble 计算）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 3-4（EP / DP 策略选择）
- [ ] [🔴 A100×4] **集中日**：起 4× A100 VM，分别跑 Qwen2.5-72B 的 TP=4 / TP=2+PP=2，对比延迟和吞吐

#### 自查

1. 推导 TP=N 下单层 AllReduce 通信量公式（用 token 数、hidden_dim、N 表达）
2. 推理时为什么用 1F1B PP 收益小？（hint: 推理无反向）
3. DeepSeek-V3 用 EP=256 的关键挑战和 All-to-All 优化思路

---

### Ch10 生产环境实践（🟢🟡 · 阅读 5h + 实操 6h）

#### 必做

- [ ] [🟢] 读完 6 篇子文档
- [ ] [🟢🟡] 完成 exercises 练习 1（搭 Prometheus + Grafana + vLLM 监控）—— 本地 Qwen2.5-1.5B 即可

#### 推荐

- [ ] [🟢] 完成 exercises 练习 2-3（routing 策略 + cost 计算）
- [ ] [🟡 A10] 完成 exercises 练习 4（profiling：用 nsys / py-spy 抓一次 vLLM 推理）

#### 自查

1. 列出 vLLM 你会监控的 8 个关键 metric 及其告警阈值
2. cache-aware routing 与 round-robin 在 RAG 场景下的吞吐差异？
3. canary release 滚动到 100% 流量的 4 个判停指标

---

### Ch11 前沿研究（📖 · 阅读 8h + 实操 4h）

#### 必做

- [ ] [📖] 读完 6 篇子文档
- [ ] [📖] 完成 exercises 练习 6（论文精读报告，2 页 / ~1500 字，~3h）
- [ ] [📖] 完成 exercises 练习 5（vLLM hybrid KV cache 源码走读）

#### 推荐

- [ ] [📖] 完成 exercises 练习 1-3（多选 1 篇额外论文 + 趋势分析 + Hybrid KV design）

#### 自查

1. 解释为什么 Mamba / Hybrid 架构是 KV Cache 的"终极优化"
2. 列出 2026-2027 你认为最有前景的 3 个推理优化方向
3. 选 1 篇你精读过的论文，5 分钟讲清楚动机/方法/局限

---

### Ch12 结构化输出（🟢🟡 · 阅读 3h + 实操 4h）

#### 必做

- [ ] [🟢] 读完 4 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（DFA 手画 + 合法 next token 列举）
- [ ] [🟢] 完成 exercises 练习 5（OpenAI SO vs Outlines 对照实验，~2h，需 API）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 2-4（speculative 兼容性 + 生产部署 + jump-forward）
- [ ] [🟡 A10] 完成 exercises 练习 6（vLLM structured outputs backend 实测，多个 backend 对比）
- [ ] [📖] 阅读 Ch12.4，给自己的业务写一份 schema cache + fallback 上线 checklist

#### 自查

1. 解释 logits mask 如何保证 100% schema 合法性
2. jump-forward 优化的两个前提条件
3. constrained decoding 何时会与 speculative decoding 互相伤害

---

### Ch13 多模态推理（🟢🟡 · 阅读 3h + 实操 4h）

#### 必做

- [ ] [🟢] 读完 4 篇子文档
- [ ] [🟢] 完成 exercises 练习 1（VLM KV Cache 预算计算）
  - 用 `scripts/kv_cache_calculator.py --mix ...` 校验低/中/高分辨率流量分布
- [ ] [🟢] 完成 exercises 练习 4（Vision API 计费规律观察，~1.5h，需 API）

#### 推荐

- [ ] [🟢] 完成 exercises 练习 2-3（prefill 延迟分析 + 优化收益估算）
- [ ] [📖] 阅读 Ch13.4，设计一套 max images / max tiles / downgrade policy

#### 选做

- [ ] [🟢] 完成 exercises 练习 5（3070 本地跑 VLM 可行性测试，~3h）
- [ ] [🟡 A10] 用 vLLM 跑 Qwen2-VL-7B，对比单图 vs 多图请求的 TBT

#### 自查

1. 解释 VLM 比纯文本模型对 prefill 延迟更敏感的根本原因
2. ViT compute cache 在什么业务下收益 > 30%
3. FastV 50% token pruning 的质量风险在哪些任务最高

---

## 4. 通关里程碑

按 STUDY_GUIDE 的 Checkpoint 体系扩展到 5 个，加入实操验收。

### Checkpoint 1: KV Cache 全栈基础（Ch01-04 完成后）
- [ ] 任意模型 3 分钟内手算 KV Cache 大小
- [ ] 能画 PagedAttention block 分配流程图
- [ ] 完成了 Ch01-04 的所有 [必] 项

### Checkpoint 2: 内存管理与解码优化（Ch05-07 完成后）
- [ ] 能推导分离架构 break-even 并发数
- [ ] 不看公式能讲投机解码数学直觉
- [ ] 完成了 Ch05-07 的所有 [必] 项

### Checkpoint 3: 调度与分布式（Ch08-09 完成后）
- [ ] 能讲清 vLLM scheduler 的 5-7 步
- [ ] 给定模型 + GPU + workload，能选出 TP/PP/EP 组合并说理由
- [ ] 完成了 Ch08-09 的所有 [必] 项

### Checkpoint 4: 生产就绪（Ch10 + Ch12 + Ch13 完成后）
- [ ] 能搭起 Prometheus + Grafana 监控 vLLM
- [ ] 能给一个 RAG/IE/VLM 业务写技术选型文档
- [ ] 完成了 Ch10/12/13 的所有 [必] 项

### Checkpoint 5: 前沿与综合（Ch11 完成 + 速通收尾）
- [ ] 完成至少 1 篇论文精读报告（练习 6）
- [ ] 能讲 3 个 2026-2027 推理优化趋势
- [ ] 回头看 STUDY_GUIDE 自测里程碑 1-4 全部通过

---

## 5. 速通计划（5 周高强度）

> 适合"全力以赴、每天 6-8 小时"的状态。可压缩到 4 周（每天 8-10h）或拉长到 8 周（每天 4h）。

### 总体节奏

```
Week 1 (~40h)：KV Cache 全栈基础         — Ch01-04 理论
Week 2 (~40h)：内存管理 + 解码优化       — Ch04 源码 + Ch05-07
Week 3 (~40h)：调度 + 分布式 + 多模态/SO — Ch08-09 + Ch12-13
Week 4 (~40h)：上云实操周（关键周）       — 集中跑所有 🟡🔴 项
Week 5 (~25h)：生产 + 前沿 + 总结         — Ch10 + Ch11 + 复盘
```

### Week 1: KV Cache 全栈基础（~40h）

| 天 | 上午 (3h) | 下午 (3h) | 晚上 (1-2h) | 验收 |
|----|-----------|-----------|------------|------|
| D1 | Ch01.1-1.2 KV Cache 布局 + Prefill-Decode | exercises 练习 1（72B 计算） | 笔记：prefill vs decode 一句话总结 | 5 min 算出任意模型 KV size |
| D2 | Ch01.3 + exercises 练习 2-3 | Ch02.1-2.2 prefix caching 原理 | 笔记：caching 何时有效 | 能讲 KV cache 复用机制 |
| D3 | Ch02.3 RadixAttention + 2.4 API caching | exercises 练习 1（Anthropic API） | 实测 5min TTL，记数据 | 看懂 cache_*_tokens 字段 |
| D4 | Ch03.1-3.2 量化 + MLA | exercises 练习 1（计算器） | 推导 MLA 57x | 手写 MLA 压缩比推导 |
| D5 | Ch03.3-3.4 + Ch04.1 PagedAttention | exercises 练习 2 + Ch04 准备 | 复习本周 | Checkpoint 1 前半 |
| D6 | Ch04.2-4.3 vLLM memory + preemption | exercises 练习 1（Block Pool 模拟器） | 调通模拟器 | 能描述 block 分配流程 |
| D7 | Ch04.4 fragmentation + 整周复盘 | 走读 vLLM `block_pool.py`，加 print | 周笔记整理 | **过 Checkpoint 1** |

### Week 2: 内存管理与解码优化（~40h）

| 天 | 上午 | 下午 | 晚上 | 验收 |
|----|------|------|------|------|
| D8 | Ch05.1-5.2 分离动机 + 架构 | exercises 练习 1 传输开销 | 笔记：分离 vs 聚合 | 算出 break-even 并发 |
| D9 | Ch05.3-5.5 KV transfer + 何时用 | exercises 练习 2-3 | 走读 vLLM disagg 代码 | 能列 3 个不适合分离的场景 |
| D10 | Ch06 全章节（5 篇） | exercises 练习 1 OffloadManager | 调通 pinned memory copy | 估算 PCIe 4.0 上限 |
| D11 | Ch06 exercises 练习 2-3 | Ch07.1 数学基础 | 推导 E[accepted] | 不看书写出公式 |
| D12 | Ch07.2-7.4 EAGLE / Medusa / MTP | exercises 练习 1 模拟器 | 验证公式 | accept rate 与理论吻合 |
| D13 | Ch07.5-7.6 + exercises 练习 2 | exercises 练习 3-4 调参 | 整理 Ch05-07 笔记 | Checkpoint 2 前半 |
| D14 | 整周复盘 + 自查题挑战 | 准备 Week 3 | 休息 / 缓冲 | **过 Checkpoint 2** |

### Week 3: 调度 + 分布式 + 多模态/SO（~40h）

| 天 | 上午 | 下午 | 晚上 | 验收 |
|----|------|------|------|------|
| D15 | Ch08.1-8.2 continuous batching + scheduler | 走读 `scheduler.py` 的 `schedule()` | 画 schedule 流程图 | 能讲 5-7 步 |
| D16 | Ch08.3-8.5 chunked prefill + 优先级 + DBO | exercises 练习 1 数学分析 | 笔记：调参直觉 | 三角图能画 |
| D17 | Ch09.1-9.2 TP + PP | exercises 练习 1 TP 通信建模 | 推导 AllReduce 公式 | 手写通信量公式 |
| D18 | Ch09.3-9.5 EP + DP + CP | exercises 练习 2 PP bubble | 笔记：DeepSeek-V3 EP=256 | 能讲 All-to-All 挑战 |
| D19 | Ch09.6 混合并行 + exercises 练习 3-4 | Ch12.1-12.2 SO + serving | exercises 练习 1 DFA | 能列合法 next tokens |
| D20 | Ch12.3 生产模式 + exercises 练习 5（API） | Ch13.1-13.2 VLM 挑战 + 调度 | exercises 练习 1 计算 | VLM KV 预算清楚 |
| D21 | Ch13.3 优化 + exercises 练习 2-4 | 整周复盘 | 准备 Week 4 上云 | Checkpoint 3 + 4 部分 |

### Week 4: 上云实操周（~40h，预算 $80-120）

**关键周**。前面三周积累的理论，本周一次性落地。提前一天申请 quota。

| 天 | 安排 | 预算 | 验收 |
|----|------|-----|------|
| D22（API 日）| 上午：Ch02 exercises 练习 1 完整跑 + Ch12 练习 5 OpenAI SO 对照 + Ch13 练习 4 Vision API。下午：Ch11 选 1 篇 paper 精读。晚上：写 Ch11 练习 6 报告草稿 | API ~$10-15 | 三组对照实验数据齐 |
| D23（L4/A10 日 - 单卡）| **09:00 gpu-up A10 VM，准备环境**。10:00-13:00：Ch04 exercises 练习 +preemption 观察。14:00-17:00：Ch07 投机解码 + Ch08 调参三角图。18:00-19:00：Ch10 监控 dashboard 搭建。**19:00 gpu-down** | ~$25-35 | Ch04/07/08/10 [荐] 项全完成 |
| D24（L4/A10 日 - 续）| **09:00 gpu-up**。上午：Ch06 swap-space 实验 + Ch12 exercises 练习 6 guided decoding 三 backend 对比。下午：Ch13 VLM 实测（Qwen2-VL-7B）。**18:00 gpu-down** | ~$20-30 | Ch06/12/13 [荐] 项完成 |
| D25（A100 集中日）| **09:00 gpu-up A100×4 VM**。10:00-13:00：Ch05 disagg setup（2 卡）+ baseline。14:00-18:00：Ch09 多维并行 — 跑 Qwen2.5-72B TP=4、TP=2+PP=2 对比。**19:00 gpu-down** | ~$30-45 | Ch05/09 [荐] 项完成 |
| D26 | 缓冲：D22-25 没跑完的补；数据画图；写实验报告 | ~$5-15 | 所有 [荐] 项收尾 |
| D27 | 本地：Ch11 练习 6 报告定稿 + Ch10 cost 计算 + Ch13 练习 5（3070 跑 VLM 可行性） | $0 | 报告 2 页交付 |
| D28 | 整周复盘 + 数据回看 + 准备 Week 5 | $0 | **过 Checkpoint 4** |

> Week 4 预算实际花费很可能 $80-100。如果 quota 卡住或资源紧张，D23/D24 用 A10 替代 L4 完全没问题，价格相近。

### Week 5: 生产 + 前沿 + 总结（~25h）

| 天 | 安排 | 验收 |
|----|------|------|
| D29 | Ch10 全章节阅读 + exercises 练习 2-3 | 列出 8 个监控 metric |
| D30 | Ch11 全章节阅读 + exercises 练习 1-3 | 趋势分析报告草稿 |
| D31 | Ch11 exercises 练习 5（hybrid KV 源码走读） | 画时序图 |
| D32 | Ch11 exercises 练习 6 论文精读报告（如 W4 D27 没完成则补）+ 选 1 篇额外 | 2 篇报告完成 |
| D33 | 整体复盘：把 5 周笔记整合成一份 ~3-5 页技术总结 | **过 Checkpoint 5** |

### 速通生存建议

- **每天起床先看当天的"验收"是什么**，目标导向
- 困了就读论文 / 笔记整理，别硬撑写代码（错的代码 debug 一小时不如休息半小时）
- 周末必留半天放空（D14、D21、D28）—— 学习曲线衰减很真实
- **数据落到 Excel / Notion**：每个实验的输入参数、结果、画的图都存档。后面写简历或面试讲项目，靠的就是这些数字
- **不要追求 100% 完成度**：[选] 项可以跳；卡住 30 分钟就跳过 → 周末补
- **每周末抽 30 分钟回看上一周笔记**，强化巩固比赶进度更重要

---

## 附录 A: 文件 / 命令速查

```bash
# 进度跟踪：把本文档的 checklist 转成 todo
grep -E "^\- \[ \]" HANDS_ON.md | wc -l    # 还剩多少项
grep -E "^\- \[x\]" HANDS_ON.md | wc -l    # 已完成

# 起停 Azure VM（假设 alias 已配）
gpu-up && ssh azureuser@$(az vm show -d -g myrg -n mygpuvm --query publicIps -o tsv)
gpu-down  # 每天结束必须执行

# 监控本地 GPU
watch -n 1 nvidia-smi
```

## 附录 B: 当目标偏离本路径时

- **只想做面试准备**：Week 1-3 + Checkpoint 5 + 论文精读，跳过 Week 4 上云
- **只想上线一个 vLLM 服务**：Ch04 + Ch08 + Ch10 + Week 4 D23-24，跳过其他
- **想转岗到推理引擎团队**：全部 [必][荐]，外加多读 vLLM PR / commit history
