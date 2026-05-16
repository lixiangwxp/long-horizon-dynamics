1.tmux在远程服务器跑
2.调用一个agent在后台检测nividia-smi的进度+统计目前的实验结果
3.给我linux操作指令的时候，要同时帮我讲解。执行每条 Linux/SSH 指令都会先说“我要执行什么”和“为什么执行”，以及具体的linux语法，让我也可以顺着学这一套排查流程。
4.失败实验确认失败后可清理，但只能清理对应失败实验的进程、tmux会话和实验产物；禁止删除无关文件、源码、数据集、基线、成功实验和汇总报告。
5.假如代码有轻微不涉及逻辑的bug导致实验无法运行，可自行修改和push以及同步。
6.每跑完一次训练或 locked audit，都要把实验 id、配置、关键结果、结论、是否读取 horizon/test、是否保留/清理 artifacts 写进 Prompt.md。Prompt.md 是重要长期记忆，不写完整巡检流水账，只写后续决策真正需要的摘要。
7.写代码一定要注重代码的简洁和可读性，不要有防御性编程。
8.模型开发时不局限于当前 grutcn/tcnlstm 的原始结构叙事；允许上网查找文献、benchmark、模型 trick、复杂模块和训练技巧，并允许根据 validation 结果、训练曲线、horizon/test 聚合指标和误差模式对实验模型做较大结构改造，只要改造方向能带来潜在指标提升。但只能把 scripts/dynamics_learning/models/grutcn.py 和 scripts/dynamics_learning/models/tcnlstm.py 的模型叙事改得更复杂、更有表达力、更有说服力，不能简化或削弱；最终目标是在各项指标上实现全面提升，而不是只优化单一验证数值。
9.实验监控不要等完整训练结束；活跃实验中自行决定检查时间，包括 tmux/进程/GPU、日志、CSVLogger、checkpoint、train_summary 和 horizon/test 结果。检查频率根据训练速度和决策紧迫度自行选择。训练和开发阶段允许读取 horizon/test metric 内容，并允许用它来挑选候选、调整结构和改进模型；每次读取都必须记录实验 id、checkpoint、关键 horizon/test 指标、是否影响后续调参/结构决策，以及结论。
10.代码修改权限限定在 `scripts/dynamics_learning/models/tcnlstm.py`、`scripts/dynamics_learning/models/grutcn.py`，以及让这两个模型运行、训练、测试、评估所必需的对应 train/test/eval wiring。每次修改后必须记录实验步骤、架构思路、文件改动、检查/同步命令和结果到 `Prompt.md`。
11.模型开发可使用 `/Users/lixiang/Documents/Obsidian Vault/trick/trick.md` 里的 trick，也可上网查找其他 trick、模块或训练技巧。使用任何 trick 或外部研究导致模型、训练、测试或评估行为变化时，必须记录来源、理由、改动文件/协议和观察结果，并明确告诉用户。不得泄漏数据：禁止把私有数据集、原始轨迹、标签、checkpoint、完整日志或可复现样本上传到外部网站/工具；禁止把 validation/test/horizon 的样本、标签或轨迹硬编码进模型、训练流程或提示词；允许使用聚合指标、误差趋势和已记录的实验摘要做结构决策。
12.最终目标是让 `scripts/dynamics_learning/models/tcnlstm.py` 和 `scripts/dynamics_learning/models/grutcn.py` 这两个新模型中有一个在 locked long-horizon rollout 指标上全面超过原模型/基线；新模型不再限制可见历史长度 `history_length`，可以使用更长历史、多尺度历史或自适应历史选择机制，只要不泄漏数据，最终以是否超过原始 baseline 各指标最佳包络为准；历史长度不是越长越好，候选结构应允许模型通过 gate/attention/pooling/reliability 权重选择有效历史并抑制过长历史噪声；至少先拿到一个冻结候选的 locked audit 胜利，随后继续推动两个新模型的广泛指标提升。
13.默认采用“主会话总控 + 分支 agent 执行/分析”的工作模式，以减少主会话上下文污染并提升决策效率。主会话只做总指挥、架构判断、边界控制、最终 review 和停止/继续/同步/审计/记录决策；不要把大量训练日志、diff 细节和普通巡检流水塞进主会话。结果分析 agent 负责读取 loss/validation/horizon/test/checkpoint/结果表、tmux/进程/GPU、日志、CSVLogger、train_summary 和必要的实验产物，给出是否继续、停止、清理、保留、改结构或进入 audit 的建议；允许基于 horizon/test metric 做候选筛选和模型改进，但必须在 `Prompt.md` 记录。代码修改 worker 只在明确边界内执行小范围实现、评审或验证，默认限制在 `scripts/dynamics_learning/models/grutcn.py`、`scripts/dynamics_learning/models/tcnlstm.py` 及必要 train/test/eval wiring，完成后只回报改动摘要、文件列表、验证命令和结果、风险与下一步建议。`Prompt.md` 作为长期记忆，只写后续决策真正需要的压缩摘要；`MODEL_DEV_HANDOFF.md` 作为可交接详细状态，记录远程路径、tmux、日志、checkpoint、门槛、近期结论和下一步建议；tmux 只作为远程训练实际运行处。遇到迷茫或重大策略节点时，不要单 agent 闷头推进；可以并行调用多个 agent 讨论和分工，并由主会话统一 review 后再决定是否采纳。
14.以后写项目文档、交接记录、实验记录和说明文档时，默认使用中文撰写；代码标识、命令、路径、实验 id、指标名和必要英文术语保持原样。
15.Prompt.md 写入规则：只保留长期有用信息，包括当前活跃实验状态、关键门槛、最终实验摘要、代码/协议/trick 改动、locked audit pass/fail、失败清理范围和重要异常；不要记录普通巡检细节。文件过长时，把旧完整流水账归档到 `docs/archive/`，主 Prompt.md 保持压缩可读。
16.MODEL_DEV_HANDOFF.md 写入规则：作为新聊天、新 agent 或人工接手的详细中文交接文档，要比 Prompt.md 更自包含，记录当前活跃实验、远程路径/tmux/log/checkpoint、监控门槛、近期关键实验结论、当前代码/协议状态和下一步执行建议；不要写逐分钟巡检流水。若文件过长，归档旧版到 `docs/archive/`，主文件保留足够新 agent 继承的详细摘要。
17.已确认的模型 idea（后续优先）：自适应历史选择（anchor 只看短窗，长历史仅进 adaptive branch）+ raw-history token Transformer side branch + latent SE residual。history-expanded 候选必须用 gate/attention/multi-scale/null context/reliability 抑制长历史噪声。所有新分支默认 checkpoint-safe/zero-init，先 smoke 确认 checkpoint 加载与 trainable 参数过滤，再正式训练。
18.`trick.md` 使用顺序：先诊断（train/valid、one-batch overfit、bad case、数据切分泄漏）再增强（EMA/SWA、R-Drop/MC Dropout、一致性/不确定性、物理合理扰动）。允许 physics-informed loss，但必须带 slack/reliability gate 且默认 train-only（参考：Physics-Inspired Temporal Learning of Quadrotor Dynamics for Accurate Model Predictive Trajectory Tracking；Physics-Informed Neural Network for Multirotor Slung Load Systems Modeling 的 slack variables 思路）。任何 trick/协议变更都要在 `Prompt.md` 记录来源、权重/配置、改动文件与关键聚合结果；禁止伪标签或测试信息泄漏进训练。
19.指标协议（压缩版）：除 `valid_*`/`h50_E_*`/`mean_E_*` 外，允许引入 open-loop 多步汇总指标（例如 `MSE_{1:H} = (1/(NH)) * Σ_t Σ_{h=1..H} ||x_{t+h}-xhat_{t+h}||_2^2` 或等价的 `MSE_1_to_F`），用于更稳定地衡量长时域 rollouts；任何新增指标/聚合方式一旦用于决策，必须在 `Prompt.md` 记录“公式/实现位置/本次读到的值/如何影响后续决策”。
20.上下文节流规则：同一聊天窗口或 heartbeat 自动化中，优先读取 `MODEL_DEV_CURRENT.md` 获取当前 active/gate/路径/下一步；只有新聊天、上下文压缩后状态不明、当前状态冲突、重大复盘或人工接手时，才完整读取 `Prompt.md` 和 `MODEL_DEV_HANDOFF.md`。`Prompt.md` 只做长期压缩 ledger，`MODEL_DEV_HANDOFF.md` 做详细交接，二者都不要被普通巡检反复整篇读入上下文。
21.`MODEL_DEV_CURRENT.md` 是当前状态单页总控，必须及时更新。以下事件发生后必须同步更新它，并同时按需更新 automation：active training/evaluation 启动或停止；gate 触发；出现新 best；训练自然结束；eval/audit 启动或完成；读取 horizon/test metric；发生 OOM/NaN/Traceback/远程中断；改变模型结构、训练协议、评估协议或关键阈值。普通无变化巡检不要写入；若当前状态已过期，优先修正 `MODEL_DEV_CURRENT.md` 再继续下一步，避免后续 agent 盯错 tmux、路径或 gate。
22.实验推进要像实验负责人而不是单纯候选执行器：每个候选启动前必须有明确假设、预期能改善的指标、失败判据和下一步分叉；训练后按“明确假设 -> 小验证/smoke -> validation gate -> horizon/test 聚合判断 -> 失败就换结构”的循环推进。连续同类微调如果 1-2 个候选没有带来 horizon/test 聚合改善，应主动收缩该方向并复盘失败模式，不要继续开相似小 adapter 或只调学习率；优先把资源投到结构假设更清楚、能解释误差模式、且有可能全面改善 h50/mean/MSE_1_to_F 的方案。
23.GPT Pro 2026-05-14 建议中的 eval guard 与 `state_update_mode` 已实现并保留：`eval.py` 必须检查 `unroll_length >= max(eval_horizons)`，summary 记录 `actual_unroll_length`、`requested_eval_horizons`、`computed_eval_horizons`、`skipped_eval_horizons`，若 h50 被跳过则不能写 locked audit 结论；`state_update_mode` 支持 `residual_full_state`、`hard_vomega_kinematic`、`soft_vomega_kinematic`。但 hard/soft vomega 在当前实现上已失败，结论只能说明“在已有 full-state residual head 上硬接/软接 v/omega kinematic update”失败，不能否定 `Delta v / Delta omega` 父级思路。
24.GPT Pro 2026-05-15 策略更新：`true seq2seq Delta v / Delta omega` 不再作为当前第一优先 pivot，而是最后阶段 fallback。论文主叙事优先保持为动力系统 / 系统辨识风格：one-step transition、recursive open-loop rollout、latent context、几何一致性、hidden actuator dynamics、自适应历史选择。Seq2seq 可作为 late-stage future-control decoder extension，但不是第一创新点。
25.当前三个优先创新点依次为：SO(3)-aware raw-token geometric history representation（quaternion delta 使用 `Log(q_prev^{-1} ⊗ q_next)` 而不是欧氏差分）；history-only hidden dynamics context（首轮只用历史 `dmot/vbat`，禁止 future context，首轮不使用 `a/alpha`）；adaptive history selector（短窗 anchor 保持 H20，长历史只进入 gated side branch，必须有 gate/attention/null context/reliability 抑制长历史噪声）。
26.当前实验优先级：先完成 active `raw_token_geometric_delta`；若有接近正信号，最多做一次 conservative scope 或短 continuation；之后进入 history-only `dmot/vbat` context branch；再进入 adaptive history selector；再把成熟 trick 迁移到 `tcnlstm.py`；只有这些方向都没有 horizon/test 聚合收益时，才启动完整 true seq2seq `Delta v[1:F] / Delta omega[1:F]` predictor。
27.Gate 更新：`raw_token_geometric_delta` 属于 representation fix，除非 e0/e1 灾难性恶化，否则至少观察 e1 或 natural finish；`dmot/vbat` context branch 重点看 `valid_v/valid_omega`、h50 `E_v/E_omega` 和 `MSE_1_to_F`，若没有正信号先检查字段进入 forward、归一化和 leakage；adaptive history selector 除 loss 外必须记录 gate mean/std/saturation，gate 学会关闭长历史且指标不坏也可作为有效诊断；true seq2seq 候选不能用 e0 直接杀死，至少给 1-2 个完整 validation 周期、独立 LR/warmup 和结构匹配 trainable scope。
28.禁止连续小修规则：若连续 1-2 个同类 tiny adapter / scale / LR continuation 没有带来 `MSE_1_to_F` 或 h50 `E_v/E_omega/E_q` 聚合收益，不得继续开相似候选；必须转向下一层结构假设。
29.在 snapshot/review 等待态下，Codex 只能做 read-only 巡检和网络/SSH 诊断；不得启动训练、eval、horizon 或 locked audit，除非用户明确恢复实验推进。网络/SSH 诊断只能做低风险只读检查；除非发现会改变 active 状态的事实，否则不要写项目文档。

## Git / Commit / Experiment Version Rules

### 核心原则

本项目不强制每个调参实验都开分支，允许在 `main` 上快速开发和调参；但正式实验必须可追溯到明确的 commit SHA。

一句话规则：

> 代码结构变化先 commit；调参失败可以不 commit；调参成功要 commit 记录；任何正式 validation、locked horizon/test、new best 或 GPT Pro review 都必须能追到 commit SHA + experiment id + 完整配置。

### 1. 两种工作模式

项目采用两种模式：Architecture Mode 和 Tuning Mode。

#### Architecture Mode

只要改动以下内容，就属于 Architecture Mode：

* `scripts/dynamics_learning/models/grutcn.py`
* `scripts/dynamics_learning/models/tcnlstm.py`
* `scripts/dynamics_learning/lighting.py`
* `scripts/dynamics_learning/data.py`
* `scripts/dynamics_learning/registry.py`
* `scripts/config.py`
* `scripts/train.py`
* `scripts/eval.py`
* 任何会改变模型结构、rollout 逻辑、loss、metric、训练协议、数据协议、eval 协议、checkpoint loading 行为的代码

Architecture Mode 的代码必须 commit 后才能启动正式训练。允许先 dirty smoke/debug，但不能把 dirty run 作为正式结论。

Architecture Mode 推荐流程：

```bash
git status --short
python -m py_compile <changed_python_files>
git diff --check -- <changed_files>
git add <changed_files>
git commit -m "arch: <short description>"
git rev-parse HEAD
git push
```

#### Tuning Mode

如果只是在已有代码基础上改命令行参数或实验配置，则属于 Tuning Mode，例如：

* learning rate
* batch size
* `accumulate_grad_batches`
* epochs / patience / `min_delta`
* loss weights
* `trainable_parameter_patterns`
* `history_length` / `unroll_length`
* `scale_init`
* selector prior
* gate threshold
* 是否继续 continuation

Tuning Mode 可以不为每个失败调参 commit。失败调参只记录实验摘要，不污染 commit history。

但是每个 tuning run 必须记录：

* base commit SHA
* experiment id
* 完整启动命令或关键 CLI 参数
* 初始化 checkpoint
* 关键配置
* validation gate
* 是否读取 horizon/test
* 结论：continue / stop / audit / cleanup / next variant

如果 tuning run 最终产生新的 best、冻结候选、或影响后续结构决策，则必须更新 `Prompt.md` / `MODEL_DEV_CURRENT.md` / `MODEL_DEV_HANDOFF.md`，并 commit 这些记录文件。

### 2. Dirty working tree 规则

允许在 dirty working tree 上做 smoke/debug。

禁止在 dirty working tree 上声明正式结果：

* official validation
* locked horizon/test
* new best
* baseline win
* paper result
* GPT Pro review 依据

如果 `git status --short` 不干净，正式训练启动前必须：

* 要么 commit 相关代码；
* 要么说明这只是 smoke/debug；
* 要么把调参改成命令行参数而不是改源码。

任何 `git_dirty=true` 的运行，只能写成 debug/smoke/exploratory，不能作为正式结论，除非用户明确批准。

### 3. 必须 commit 的情况

以下情况必须 commit：

* 模型架构变化；
* 训练/评估协议变化；
* 数据处理变化；
* 新增 CLI 参数；
* 修改 loss / metric / rollout；
* 修复会影响结论的 bug；
* 产生新的 best 或冻结候选后，更新实验记录；
* 准备让 GPT Pro review 代码；
* 准备做 locked horizon/test 结论。

### 4. 可以暂不 commit 的情况

以下情况可以暂不 commit：

* 单纯 learning rate / loss weight / batch / patience 调参；
* 失败的 continuation；
* smoke/debug；
* 只为了观察 e0/e1 而启动的临时 tuning run；
* 没有改变代码，只改变命令行参数的试验。

但暂不 commit 的 tuning run 仍然必须写清 base commit 和完整配置。

### 5. 禁止无脑 add

禁止 `git add .`，除非用户明确要求提交所有当前文件。

默认只 add 本次相关文件，例如：

```bash
git add AGENTS.md
git add scripts/dynamics_learning/models/tcnlstm.py scripts/config.py
```

不要把无关 dirty files 混进一个 commit。

### 6. 正式训练 summary 版本记录

正式训练或 eval 的 summary 中应尽量写入：

* `git_branch`
* `git_commit`
* `git_dirty`
* `experiment_id`
* `model_type`
* `history_length`
* `unroll_length`
* `state_update_mode`
* `trainable_parameter_patterns`
* `init_checkpoint`
* key loss weights

如果暂时无法实现自动写入，也必须在 `Prompt.md` 和 `MODEL_DEV_CURRENT.md` 里手动记录 base commit。

### 7. GPT Pro review 请求格式

不要让 GPT Pro review “GitHub 最新代码”，必须提供：

* repo URL
* branch name，通常是 `main`
* head commit SHA
* experiment id，如果已有
* changed files summary
* smoke / syntax check result
* validation / horizon 摘要，如果已有
* 当前需要 GPT Pro 判断的问题

如果代码没有 push，或者和 review 相关的文件仍是 dirty 状态，不得要求 GPT Pro review GitHub 代码。此时只能：

* 先 commit + push；
* 或把完整 diff patch 贴给 GPT Pro；
* 或明确说明 GitHub 不是最新版，不能作为 review 依据。

### 8. 调参成功后的冻结动作

如果某个 tuning run 比 reference 更好，必须立刻执行：

```bash
git status --short
git rev-parse HEAD
```

然后：

* 如果代码没有变化，只更新实验记录并 commit 文档；
* 如果代码有变化，先 commit 代码，再 commit/更新实验记录；
* 写入 `Prompt.md`：experiment id、base commit、配置、best validation、horizon/test、结论；
* 写入 `MODEL_DEV_CURRENT.md`：当前 best reference 和下一步 gate。

### 9. 大结构例外：什么时候建议开分支

日常快速调参可以继续在 `main` 上做。

以下情况建议开 `exp/*` 分支，但不是强制：

* true seq2seq `delta_v/delta_omega`；
* dataset adapter；
* 大改 `data.py` / `lighting.py` / `eval.py`；
* 多个候选并行开发；
* 需要 GPT Pro review 清晰 diff；
* `main` 上已有大量混杂 dirty changes。

如果开分支，命名格式：

```text
exp/<model>-<idea>-<date>
```

例如：

```text
exp/tcnlstm-actuatorctx-h10-20260516
```

日常快速调参可以继续在 `main` 上做，只要正式实验有 commit SHA 和完整记录。

### 10. 给用户的回报格式

每次 Architecture Mode 改完代码后，回报：

```text
模式：Architecture Mode
branch:
commit:
changed_files:
checks:
- py_compile:
- git diff --check:
push:
smoke:
是否可启动正式训练：
```

每次 Tuning Mode 启动或结束后，回报：

```text
模式：Tuning Mode
base_commit:
git_dirty:
experiment_id:
command/config:
init_checkpoint:
validation:
horizon/test:
结论:
是否需要 commit:
下一步:
```
