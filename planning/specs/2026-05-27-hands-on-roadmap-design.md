# Spec: HANDS_ON.md 实操路径 + Ch11/12/13 exercises 补充 + 速通计划

**日期**：2026-05-27
**作者**：与用户协作设计
**状态**：待 review

---

## 1. 背景与目标

用户希望"全面学习这个 repo + 实操关键路径"。Repo 现状：

- 13 章正文充实，理论密度高
- Ch01-Ch10 每章 `exercises.md` 已有 500-950 行的完整动手练习（含 Python 骨架、vLLM 命令、benchmark 脚本）
- Ch11/12/13 的 `exercises.md` 只有 34-98 行，明显稀薄
- `STUDY_GUIDE.md` 提供了 20 周/12 周路径规划，但**没有按硬件约束分级**，也没把现有 exercises 串成可执行列表

用户资源约束：
- 本地：1 张 RTX 3070（8GB）
- 云：Azure $150 预算
- API：可承担适量 Anthropic / OpenAI / DeepSeek API 费用

目标：让用户拿到 repo 后能**立即知道每章做什么、用什么硬件、要多久、做完如何自查**，并提供一份"全力以赴"的速通计划。

## 2. 范围

### In scope
- 新建 `HANDS_ON.md`（根目录），作为实操总入口
- 补充 Ch11/Ch12/Ch13 的 `exercises.md`，使其与 Ch01-10 体量相当
- 提供 4-5 周速通时间表（用户每天 6-8h，"4 周 / 5 周均可"）

### Out of scope
- 不改写已有 Ch01-Ch10 的 exercises（它们已经足够好）
- 不新增章节内容（13 章正文不动）
- 不自动化部署脚本到生产级（云上 startup script 仅做示例，不做 IaC）
- 不预订 Azure 资源、不替用户操作云

## 3. 交付物

### 3.1 `HANDS_ON.md`（根目录，预计 800-1200 行）

#### Section 0 — 你的硬件画像与预算分配
- 3070（8GB）能做什么 / 不能做什么的明确清单
- Azure $150 的建议分配方案（API 实验 $20-30 / L4 单卡 $50-70 / A100 集中日 $30-50 / 缓冲 $10-20）
- 三条选择路径：**全本地档**（不花云钱）、**标准档**（按建议分配）、**深度档**（自费补充）

#### Section 1 — 4 个 Track 说明
- 🟢 本地 Track（3070 + Python + API）：能做的事清单
- 🟡 L4 上云 Track（24GB 单卡跑 7B）：什么场景才需要
- 🔴 A100 集中 Track（多卡 / 大模型）：建议攒到一天一次性做完
- 📖 纯阅读 Track：Ch11 等理论章节的阅读方法（读什么、写什么笔记）

#### Section 2 — 上云操作手册
- Azure GPU quota 申请要点（NCas T4 v3、NCv4 等系列名 + 申请文案模板）
- L4 / L40S VM 标准 startup script（apt install、CUDA、conda、vllm pip install、jupyter）
- 成本控制清单：
  - VM 用完 `stop`（不是 deallocate？要明确 deallocate 才停计费）
  - Spot 实例适用场景
  - 实验前先估算时长，限定预算
- 一份"一键起 + 一键停"的本地 bash 别名建议

#### Section 3 — 13 章实操 Checklist（核心）
每章一个小节，模板：

```
## Ch04 PagedAttention（🟢🟡 混合 · 阅读 4h + 实操 4h）

### 必做
- [ ] [🟢] 读完 `01-paged-attention.md` 到 `04-fragmentation.md`
- [ ] [🟢] 完成 `exercises.md` 练习 1（Block 分配模拟器，~2h）
- [ ] [🟢] 在 vLLM 源码 `vllm/v1/core/block_pool.py` 加 print，跑一次推理观察 block 分配

### 推荐
- [ ] [🟡 L4] 用 Qwen2.5-7B 触发 preemption（手动构造长序列 + 高并发），观察 metrics

### 选做
- [ ] [🟢] 完成 exercises.md 练习 2-5

### 自查（不查资料 5 分钟内能答出）
1. 给定 num_blocks=1000、block_size=16，描述 token 数 7000 的请求如何分配
2. preemption 的两种策略（swap vs recompute）各自的优劣
3. 为什么 PagedAttention 相比 contiguous KV cache 不会显著降低性能
```

13 章覆盖完整，每章 3-7 个必做项 + 推荐 + 自查。

#### Section 4 — 通关里程碑（5 个 Checkpoint）
对齐 STUDY_GUIDE 的 4 个 checkpoint，新增 1 个"实操总验"。

每个 checkpoint：
- 必须能答出的 5-8 道题
- 必须完成的实操项核对
- 如果某项不会，回头补哪一章

#### Section 5 — 速通计划

##### 整体节奏

```
Week 1 (35-40h)：KV Cache 全栈基础 — Ch01-04
Week 2 (35-40h)：内存管理 + 解码优化 — Ch04 源码 + Ch05-07
Week 3 (35-40h)：调度 + 分布式 + 多模态/结构化 — Ch08-09 + Ch12-13
Week 4 (40h)：上云实操周 — 把 🟡🔴 项目集中做掉
Week 5 (20-30h)：Ch10 生产 + Ch11 前沿 + 总结
```

##### 每天表格（部分示例）

```
Day 1 — Ch01.1-1.2 KV Cache 内存布局 + Prefill-Decode
- 上午（3h）: 读 01-memory-layout.md + 02-prefill-decode.md
- 下午（3h）: 做 exercises 练习 1（Qwen-72B KV Cache 计算）
- 晚上（1h）: 写当天笔记 — 一句话概括 prefill 与 decode 的区别
- 验收: 能 5 分钟算出任意模型给定 seq_len 下的 KV Cache 大小
```

5 周 × 5-7 天每天都给出这种结构。Week 4（上云日）会精确到小时（"上午 09:00 起 L4 VM、10:00 跑 baseline、12:00 停"）。

### 3.2 Ch11/12/13 exercises.md 补充

#### Ch11 补 2 题（在现有 3 题后追加）
- 练习 4: vLLM Hybrid KV Cache 实现走读 — 给定 commit hash + 关键文件路径
- 练习 5: 论文精读报告 — 从 `05-paper-list.md` 选 1 篇 ⭐，2 页报告

#### Ch12 补 2 题（在现有 2 题后追加）
- 练习 3: OpenAI structured output API + Outlines 对照实验（纯 API + Python 本地）
- 练习 4: vLLM structured outputs backend 实测（🟡 L4 上做）

#### Ch13 补 1-2 题（在现有 1 题后追加）
- 练习 2: Anthropic / OpenAI vision API 计费规律观察（单图 / 多图 / 高低分辨率）
- 练习 3（选做）: 3070 上跑 LLaVA-1.5-7B AWQ 量化版可行性测试

每题保持与 Ch01-10 同等深度：明确目标、给 Python 骨架或具体步骤、列出验收标准。

## 4. 写作原则

1. **硬件标签优先于章节顺序** — 每个 checklist 项首字符标 🟢🟡🔴📖，扫一眼就知道能不能现在做
2. **时间估算必给** — 每章给"阅读 Xh + 实操 Yh"，每个练习给"~Nh"
3. **自查问题贴近面试** — 5 个自查问题要能模拟系统面试场景，逼用户从描述跳到推导
4. **不重复 STUDY_GUIDE** — STUDY_GUIDE 是宏观路径（按 level/兴趣分流），HANDS_ON 是微观执行（按硬件分流 + 每日表格）。两者在 HANDS_ON 顶部明确说"先读 STUDY_GUIDE 选档位，再用本文档执行"
5. **数字保守** — 预算和时间都按"完不成会沮丧"的反向给：宁可估高
6. **中文为主，技术术语保留英文** — 与 repo 现有风格一致

## 5. 文件结构变更

```
新增：
  HANDS_ON.md
  planning/specs/2026-05-27-hands-on-roadmap-design.md（本文件）

修改：
  11-frontier-research/exercises.md（追加练习 4-5）
  12-structured-output/exercises.md（追加练习 3-4）
  13-multimodal-serving/exercises.md（追加练习 2-3）
  README.md（在"前置知识"附近加一行指向 HANDS_ON.md）
  STUDY_GUIDE.md（顶部加跨链：实操路径见 HANDS_ON.md）
```

## 6. Commit 计划

1. `chore: add spec for hands-on roadmap` — 本 spec 文档
2. `docs: expand Ch11/12/13 exercises` — 三章 exercises 补充
3. `docs: add HANDS_ON.md - chapter-by-chapter hands-on roadmap` — 主文档
4. `docs: cross-link HANDS_ON from README and STUDY_GUIDE` — 加导航

每个 commit 独立成原子操作，方便日后翻阅。

## 7. 自我 review（spec 写完后）

- [x] 无 TBD / 占位符
- [x] 无矛盾（HANDS_ON 与 STUDY_GUIDE 定位清楚分工）
- [x] 单一交付物范围合适，无须拆分子 spec
- [x] 关键决策已明确：硬件标签体系、时间估算粒度、5 周节奏

## 8. 不打算做的事（明确）

- 不做"自动选路"小工具（如填问卷生成个人路径）—— 用户能力足够手选
- 不做交互式 jupyter 教程 —— 增加维护负担，与 repo 风格不符
- 不录视频 / 不画大量插图 —— ASCII / Mermaid 已足够
- 不替用户判断"3070 能不能跑 X 模型" 的边界情况 —— 给出明确清单后边界由用户测
