# VLM 优化技术

> 从模型层到系统层，多维度削减视觉 token 的计算和存储开销

## 1. 视觉 Token 压缩

### 1.1 问题：视觉 token 存在大量冗余

图像不同于文本——文本中每个 token 通常都承载关键语义，但图像中**大量 patch 是背景或重复纹理**，对应的视觉 token 信息量极低。研究表明，一张典型图像中，仅 20–50% 的视觉 token 对最终生成结果有显著贡献。

这意味着我们有机会在**不显著损失质量**的前提下，大幅减少视觉 token 数量，从而：

- 降低 prefill 计算量
- 减少 KV Cache 显存占用
- 提高并发容量

### 1.2 Token 剪枝：FastV

**FastV** (An Image is Worth 1/2 Tokens After Layer 2) 的核心发现是：LLM 在处理视觉 token 时，**前几层就能通过 Attention 分数区分"重要"和"不重要"的视觉 token**。

**工作原理：**

```
Layer 0:  [vis_1, vis_2, ..., vis_576, text_1, ..., text_30]  ← 全部 606 tokens
Layer 1:  [vis_1, vis_2, ..., vis_576, text_1, ..., text_30]  ← 全部 606 tokens
Layer 2:  计算 attention scores → 识别低注意力的视觉 token
          剪掉 50% 低注意力视觉 tokens
Layer 3+: [vis_top_288, text_1, ..., text_30]                 ← 318 tokens
...
Layer 32: [vis_top_288, text_1, ..., text_30]                 ← 318 tokens (生成)
```

**注意力分数计算：**

对于每个视觉 token $v_i$，在第 $k$ 层计算其被文本 token 关注的平均程度：

$$\text{importance}(v_i) = \frac{1}{|T|} \sum_{t \in T} \text{Attn}(t, v_i)^{(k)}$$

其中 $T$ 是所有文本 token 的集合，$\text{Attn}(t, v_i)^{(k)}$ 是第 $k$ 层中文本 token $t$ 对视觉 token $v_i$ 的注意力权重。

**性能和质量权衡：**

| 剪枝比例 | 剩余视觉 Token | Prefill 加速 | KV Cache 节省 | 质量下降 |
|---------|--------------|-------------|--------------|---------|
| 0% (无剪枝) | 576 | 1× | 0% | 0% |
| 25% | 432 | ~1.3× | ~14% | <1% |
| 50% | 288 | ~1.7× | ~28% | 1–2% |
| 75% | 144 | ~2.5× | ~42% | 3–5% |
| 90% | 58 | ~3.5× | ~50% | 8–15% |

!!! info "Serving 中的实现"
    FastV 的剪枝发生在 LLM 的前向计算中（Layer 2 之后），因此：
    
    - **Prefill 阶段**：前 2 层仍处理全部视觉 token，第 3 层起减半。Prefill 加速不到 2×（因为前 2 层是全量计算）
    - **KV Cache**：被剪掉的视觉 token 在第 3 层之后不再生成 KV，但前 2 层的 KV 已经生成（可以释放或标记为不再需要）
    - **Decode 阶段**：每步只需 attend 剩余的视觉 token KV，TBT 改善明显

### 1.3 Token 合并：LLaVA-PruMerge

**LLaVA-PruMerge** 采用了一种不同的策略：不是简单丢弃低重要性的视觉 token，而是将**语义相似的视觉 token 合并**为一个代表性 token。

**工作原理：**

```
Step 1: 计算视觉 token 间的相似度矩阵
        sim(v_i, v_j) = cosine(v_i, v_j)

Step 2: 根据重要性分数选择 "anchor tokens" (保留集)
        anchors = top_k(importance_scores)

Step 3: 将非 anchor token 合并到最近的 anchor
        for non_anchor in remaining:
            nearest_anchor = argmax(sim(non_anchor, anchors))
            anchor.embedding += weight × non_anchor.embedding
```

**与 FastV 的对比：**

| 特性 | FastV (剪枝) | LLaVA-PruMerge (合并) |
|------|-------------|---------------------|
| 信息保留 | 直接丢弃低重要性 token | 合并保留部分信息 |
| 质量 | 高压缩比下质量下降较快 | 同压缩比下质量更好 |
| 计算开销 | 低（只需排序 + 掩码） | 中（需要相似度计算 + 加权平均） |
| 实现复杂度 | 低 | 中 |
| 适合场景 | 需要最大速度提升 | 需要在高压缩比下保持质量 |

### 1.4 自适应分辨率

从源头减少视觉 token 的另一种方式是 **降低输入图像的分辨率**——更少的 tile 就意味着更少的视觉 token。

**自适应策略：**

```python
# 伪代码：根据图像内容自适应选择分辨率
def adaptive_resolution(image, complexity_threshold=0.5):
    """简单图像用低分辨率，复杂图像用高分辨率。"""
    # 计算图像复杂度 (边缘密度、纹理丰富度等)
    complexity = estimate_complexity(image)

    if complexity < complexity_threshold:
        # 简单图像 (纯色背景、简单图表等)
        max_tiles = 1   # → 1024 tokens
    elif complexity < 0.8:
        # 中等复杂度
        max_tiles = 4   # → 4096 tokens
    else:
        # 高复杂度 (密集文本、复杂场景)
        max_tiles = 12  # → 12288 tokens

    return resize_and_tile(image, max_tiles)
```

**权衡：**

- 降低分辨率可以直接减少 ViT encoding 时间和视觉 token 数量
- 但过度降低会丢失细节信息（特别是文档 OCR、小物体检测等场景）
- 需要根据任务类型动态调整：聊天场景可以低分辨率，文档理解需要高分辨率

---

## 2. ViT 计算缓存

### 2.1 动机：同一图像的重复编码

在实际应用中，同一张图像可能被多次使用：

- **同图多问** — 用户对同一张图片问多个问题（"这张图是什么？" → "图中有几个人？" → "他们在做什么？"）
- **Few-shot 示例** — 图像作为 few-shot 示例出现在多个请求中
- **重试/重新生成** — 用户对同一输入请求重新生成

在这些场景中，ViT encoding 的结果完全相同——但如果每次都重新编码，就浪费了 GPU 计算。

### 2.2 缓存方案

```
ViT Compute Cache:

                 ┌─────────────────────────┐
                 │   Image Hash → Cache     │
                 │                         │
                 │  hash_1 → embeddings_1  │
                 │  hash_2 → embeddings_2  │
                 │  hash_3 → embeddings_3  │
                 │  ...                    │
                 └─────────────────────────┘

请求到达:
  1. 计算图像 hash (perceptual hash 或 content hash)
  2. 查询 cache
     → 命中: 直接使用缓存的 embeddings, 跳过 ViT encoding
     → 未命中: ViT encoding → 存入 cache
```

**缓存的内容是什么？**

缓存的是 **Projector 输出后的 visual embeddings**，即已经映射到 LLM embedding 空间的向量：

```
缓存对象: post-projector embeddings
形状: [num_visual_tokens, hidden_dim]
大小: 576 × 4096 × 2 bytes (BF16) ≈ 4.5 MB / 图 (LLaVA-7B)
      1024 × 4096 × 2 bytes (BF16) ≈ 8 MB / 图 (Qwen-VL-7B)
```

!!! tip "为什么缓存 post-projector 而非 pre-projector embeddings？"
    Projector 通常是简单的 MLP（2 层），计算量很小。但如果缓存 pre-projector 的 ViT 输出，则每次使用缓存时仍需执行 Projector forward——虽然计算量不大，但增加了复杂度。直接缓存 post-projector embeddings 更简洁。

### 2.3 缓存管理

```python
# 伪代码：ViT Compute Cache 实现
class ViTComputeCache:
    def __init__(self, max_entries=1000, max_memory_gb=8):
        self.cache = OrderedDict()  # LRU
        self.max_entries = max_entries
        self.max_memory_gb = max_memory_gb
        self.current_memory = 0

    def get_or_compute(self, image, vit_model, projector):
        image_hash = compute_hash(image)

        if image_hash in self.cache:
            # Cache hit
            self.cache.move_to_end(image_hash)  # LRU update
            return self.cache[image_hash]

        # Cache miss: encode and store
        with torch.no_grad():
            vit_output = vit_model(image)
            embeddings = projector(vit_output)

        entry_size = embeddings.nbytes
        self._evict_if_needed(entry_size)
        self.cache[image_hash] = embeddings
        self.current_memory += entry_size

        return embeddings

    def _evict_if_needed(self, new_entry_size):
        while (self.current_memory + new_entry_size > self.max_memory_gb * 1e9
               or len(self.cache) >= self.max_entries):
            _, evicted = self.cache.popitem(last=False)  # LRU eviction
            self.current_memory -= evicted.nbytes
```

**缓存命中率的影响：**

| 场景 | 预期命中率 | ViT 计算节省 |
|------|----------|-------------|
| 同图多问（多轮对话） | 80–95% | 显著 |
| Few-shot 固定图像 | ~100% | 几乎消除 ViT 开销 |
| 随机用户图片上传 | <5% | 几乎无益 |

!!! warning "注意"
    ViT Compute Cache 只节省 ViT encoding 的计算，不节省 LLM 的 KV Cache。即使 ViT 输出被缓存，视觉 token 仍然需要经过 LLM 的完整 prefill 并生成 KV Cache。要节省 KV Cache，需要使用前缀缓存（Ch02），但如 13.1 节所述，图像的 KV Cache 很难跨请求共享。

---

## 3. 异步 ViT 处理

### 3.1 动机：消除 ViT 的串行开销

默认的 VLM 推理流程是串行的：

```
串行流程:
  ViT encoding (5-20ms) → LLM prefill (50-400ms) → LLM decode (每步 5-15ms)
  ────────────────────────────────────────────────→ 时间

总延迟 = T_vit + T_prefill + T_decode
```

ViT encoding 和 LLM 是两个独立的模型，可以利用 GPU 的并行能力进行**流水线化**。

### 3.2 流水线设计

```
异步 ViT 流水线:

Batch N 的处理:
  Time:     0    5ms   55ms              350ms
  ViT:      [==N==]
  LLM:             [===== N prefill =====][N decode][N decode]...

Batch N+1 的处理 (流水线化):
  Time:     0    5ms   55ms   60ms        360ms
  ViT:      [==N==][====N+1====]
  LLM:             [===== N prefill =====][== N+1 prefill ==]

流水线后:
  ViT 处理 N+1 与 LLM 处理 N 同时进行
  T_vit(N+1) 被 T_prefill(N) 隐藏
```

**实现方式：**

```python
# 伪代码：异步 ViT 流水线
class AsyncViTPipeline:
    def __init__(self, vit_model, llm_model):
        self.vit_model = vit_model
        self.llm_model = llm_model
        self.vit_stream = torch.cuda.Stream()  # 独立 CUDA stream
        self.pending_embeddings = {}

    async def process_batch(self, current_batch, next_batch_images):
        # 1. 在独立 stream 上异步启动下一批的 ViT encoding
        if next_batch_images:
            with torch.cuda.stream(self.vit_stream):
                next_embeddings = self.vit_model(next_batch_images)
                self.pending_embeddings = next_embeddings

        # 2. 在默认 stream 上执行当前批的 LLM forward
        #    (使用之前预计算好的 visual embeddings)
        llm_output = self.llm_model.forward(
            current_batch.token_ids,
            visual_embeddings=current_batch.cached_visual_emb,
        )

        # 3. 同步 ViT stream，确保下一批的 embeddings 就绪
        self.vit_stream.synchronize()

        return llm_output
```

### 3.3 多 GPU 场景

在多 GPU 部署中，可以将 ViT 和 LLM 放在不同的 GPU 上，实现**物理级别的并行**：

```
GPU 0: ViT Encoder (专用)
  [encode img_1][encode img_2][encode img_3]...
        │              │             │
        ▼              ▼             ▼
   (通过 PCIe/NVLink 传输 embeddings)
        │              │             │
        ▼              ▼             ▼
GPU 1-3: LLM (TP=3)
  [prefill req_1][prefill req_2][prefill req_3]...

优点: ViT 和 LLM 完全并行，无资源竞争
缺点: 需要额外的 GPU，传输 embeddings 有通信开销
适合: ViT 较大 (如 InternViT-6B) 的场景
```

---

## 4. ViT 量化

### 4.1 ViT 的计算特征

ViT 在 VLM 中的计算特征与 LLM 不同：

| 特征 | ViT Encoding | LLM Prefill | LLM Decode |
|------|-------------|-------------|------------|
| 计算类型 | Compute-bound | Compute-bound | Memory-bound |
| Batch size | 通常 1（单图） | 可变 | 可变 |
| 量化敏感性 | 较低 | 中等 | 中等 |
| 参数量 | 0.3–6B | 7–70B+ | 同 prefill |

ViT 的量化敏感性较低，主要因为：

1. ViT 的任务是**特征提取**而非精确的 token 生成——少量精度损失在后续 LLM 处理中被稀释
2. ViT 的参数量相对小，量化节省的绝对显存有限，但**计算加速**效果显著

### 4.2 量化策略

**INT8 量化 ViT：**

```
ViT-L/14 (304M params):
  FP16: 608 MB 显存, encoding 336×336 → ~5 ms (A100)
  INT8: 304 MB 显存, encoding 336×336 → ~3 ms (A100)  ← 加速约 1.7x

InternViT-6B:
  FP16: 12 GB 显存, encoding 448×448 → ~25 ms (A100)
  INT8:  6 GB 显存, encoding 448×448 → ~15 ms (A100)  ← 加速约 1.7x
```

**FP8 量化 ViT (H100/Ada)：**

FP8 在 Hopper/Ada 架构上有原生硬件支持，且对 ViT 的精度影响极小：

```
ViT-L/14 FP8:
  显存: 304 MB (同 INT8)
  速度: ~2.5 ms (A100 不支持, H100 上有原生加速)
  质量: 与 FP16 几乎无差异 (<0.1% accuracy drop on ImageNet)
```

### 4.3 ViT 和 LLM 独立量化

VLM 的两个组件可以使用**不同的量化策略**，各取最优：

```
推荐组合:

配置 1 (质量优先):
  ViT:  FP16 (最高质量特征提取)
  LLM:  BF16 (标准推理精度)

配置 2 (平衡):
  ViT:  INT8 (轻微加速，质量基本无损)
  LLM:  BF16 + FP8 KV Cache

配置 3 (性能优先):
  ViT:  FP8 / INT8
  LLM:  INT8 / FP8 (W8A8 或 FP8)
  KV:   FP8

配置 4 (极限压缩):
  ViT:  INT8
  LLM:  INT4 (AWQ/GPTQ)
  KV:   FP8
```

!!! warning "注意"
    不建议对 ViT 使用 INT4 量化——ViT 的 Attention 和 LayerNorm 对低精度更敏感，INT4 通常会导致特征质量明显下降。INT8 是 ViT 量化的安全下限。

---

## 5. 视频与多图优化

### 5.1 视频：帧数爆炸

视频理解是 VLM serving 的终极挑战。一个短视频就能产生海量视觉 token：

```
假设: 30 fps 视频, 每帧 576 visual tokens

10 秒视频: 300 帧 × 576 tokens = 172,800 visual tokens
1 分钟视频: 1800 帧 × 576 tokens = 1,036,800 visual tokens (!!)

即使 LLaMA-3-8B (GQA-8, KV per token = 128 KB):
  10 秒视频的 KV Cache: 172,800 × 128 KB ≈ 21 GB  (单请求!!)
```

这显然不可行——需要**帧采样**来大幅减少帧数。

### 5.2 帧采样策略

**均匀采样 (Uniform Sampling)：**

```
策略: 从视频中均匀选取 N 帧 (N 通常为 8-32)

10 秒视频, N=16:
  采样帧: 每 0.625 秒取一帧
  Visual tokens: 16 × 576 = 9,216 tokens
  KV Cache: 9,216 × 128 KB ≈ 1.15 GB

优点: 实现简单, 覆盖全时段
缺点: 可能错过关键动作, 对静态场景浪费帧
```

**关键帧采样 (Keyframe-based)：**

```
策略: 提取场景变化的关键帧

方法:
  1. 计算相邻帧的差异 (帧间差分, 或 CLIP 特征距离)
  2. 差异超过阈值时标记为关键帧
  3. 保留关键帧 + 时间均匀补充

优点: 关注场景变化, 减少冗余
缺点: 需要额外的帧差异计算, 关键帧数量不固定
```

**自适应采样 (Adaptive)：**

```
策略: 根据视频内容动态调整采样密度

高动态段 (运动, 场景切换): 高密度采样
低动态段 (静态场景, 对话): 低密度采样

实现: 
  1. 快速扫描视频的运动强度
  2. 为每段分配不同的采样率
  3. 总帧数控制在预算 N 内
```

### 5.3 多图场景优化

多图输入（如文档页面、商品多角度图）的优化策略：

**共享 ViT 编码器的批处理：**

```
多张图像可以在 ViT 中批量编码:

逐个编码:
  [ViT(img_1)] [ViT(img_2)] [ViT(img_3)] → 15 ms × 3 = 45 ms

批量编码:
  [ViT(img_1, img_2, img_3)] → ~20 ms (GPU 利用率更高)
```

**跨图 token 压缩：**

如果多张图像之间有重叠内容（如同一物体的不同角度），可以跨图合并相似的视觉 token：

```
img_1: 576 tokens
img_2: 576 tokens (与 img_1 有 40% 相似)
img_3: 576 tokens (与 img_1 有 50% 相似)

独立处理: 1728 tokens
跨图合并后: ~1100 tokens (节省 ~36%)
```

### 5.4 与长上下文优化的交叉

视频和多图场景本质上是**超长上下文**问题。Ch03 中的 KV Cache 压缩技术同样适用：

| 技术 | 与 VLM 的交叉应用 |
|------|------------------|
| KV Cache 量化 (Ch03.1) | 将视觉 token 的 KV Cache 从 BF16 量化到 FP8，节省 50% 显存 |
| 选择性缓存 (Ch03.3) | 在 decode 阶段选择性 evict 不重要的视觉 token KV |
| GQA/MQA (Ch03.4) | 使用 GQA 架构的 LLM 天然减少视觉 token 的 KV 开销 |
| KV Cache 卸载 (Ch06) | 将早期帧的视觉 KV Cache 卸载到 CPU/SSD |

!!! tip "实践建议"
    对于视频理解任务，建议组合使用：
    
    1. **帧采样** — 从源头控制视觉 token 总量（目标: <10K tokens）
    2. **FastV 剪枝** — 进入 LLM 后进一步削减 50%
    3. **FP8 KV Cache** — 再节省 50% 显存
    4. **KV Cache 卸载** — 将早期帧的 KV 移到 CPU 内存
    
    这四层优化可以将视频理解的显存需求降低到原始的 ~12.5%。

---

## 6. 小结

| 优化技术 | 层级 | 节省维度 | 典型收益 | 实现复杂度 |
|---------|------|---------|---------|-----------|
| FastV (token 剪枝) | 模型内 | Prefill 计算 + KV Cache | 1.5–2.5× | 低 |
| LLaVA-PruMerge (token 合并) | 模型内 | 同上，质量更好 | 1.5–3× | 中 |
| 自适应分辨率 | 预处理 | ViT 计算 + 全链路 | 1–10× | 低 |
| ViT Compute Cache | 系统 | ViT 计算 | 取决于命中率 | 低 |
| 异步 ViT 流水线 | 系统 | 端到端延迟 | 隐藏 ViT 延迟 | 中 |
| ViT INT8/FP8 量化 | 模型 | ViT 计算 + 显存 | 1.5–2× | 低 |
| 帧采样 (视频) | 预处理 | 全链路 | 10–100× | 低–中 |
| 跨图 token 合并 | 模型内 | Prefill + KV | 1.2–1.5× | 中 |
