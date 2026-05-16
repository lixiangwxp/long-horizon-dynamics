# 模型开发详细交接

最后更新：2026-05-16 12:55 CST。

本文件用途：给新的 Codex 聊天、子 agent 或人工接手者快速继承当前模型开发 sprint。它应该比 `Prompt.md` 更像“可执行交接说明”，但不保存逐分钟巡检流水。完整历史归档见：

- `Prompt.md`：压缩长期记忆和当前状态。
- `docs/archive/Prompt_full_before_compaction_2026-05-10.md`：旧完整流水账。
- `docs/archive/MODEL_DEV_HANDOFF_full_before_zh_2026-05-10.md`：本文件英文旧版归档。

接手优先级：

1. 读 `AGENTS.md`，确认用户硬规则。
2. 读 `MODEL_DEV_CURRENT.md`，确认当前 active/gate/路径/下一步。
3. 只有新聊天、上下文压缩后状态不明、当前状态冲突或重大复盘时，才读 `Prompt.md` 和本文件全文。
4. 同一窗口的 heartbeat 普通巡检不要反复整篇读取本文件，优先用 `rg`/`sed` 定向读取必要片段。

## 当前接手状态（2026-05-16 12:55 CST）

- 当前 active：none。不要启动训练、eval、horizon 或 locked audit；本轮目标是整理当前 dirty main，形成可追溯 architecture/protocol snapshot 并 push，供 GPT Pro review。
- base SHA：`27be3448d7bbf3bda7f48522fa03c747477f8d1b`。当前 dirty files 属于同一个 coherent snapshot：文档工作流规则 + 已实现的模型/训练/eval/data 协议改动。
- 本次 snapshot 文件：`AGENTS.md`、`MODEL_DEV_CURRENT.md`、`MODEL_DEV_HANDOFF.md`、`Prompt.md`、`scripts/config.py`、`scripts/eval.py`、`scripts/train.py`、`scripts/dynamics_learning/data.py`、`scripts/dynamics_learning/lighting.py`、`scripts/dynamics_learning/registry.py`、`scripts/dynamics_learning/models/grutcn.py`、`scripts/dynamics_learning/models/tcnlstm.py`。
- 当前代码/协议摘要：包含 eval no-silent-skip guard、`state_update_mode` wiring、history-only `dmot/vbat` context、adaptive history stats、GRUTCN adaptive history side branch、TCNLSTM side-history/null-trust switches，以及训练/eval summary 字段扩展。
- 最近实验状态：无 active。`modeldev_20260516_tcnlstm_geoactctx_H10_nulltrust_from_attitude_e3_p1` 已在 e0 前替换；`smoke_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1` 启动日志显示 one-batch finite，但远程 shell 不稳定，summary 未确认；不写正式结论。
- 远程状态：Tailscale/WSL/SSH 曾出现 session 创建超时，网络/SSH 诊断 subagent 已介入；snapshot 完成前不要依赖远程启动训练。
- 本次 snapshot commit：`<pending>` `arch: snapshot model sprint workflow and protocol changes`。commit/push 后回填最终 SHA，交给 GPT Pro review。
- 下一步路线：等待 GPT Pro review 当前 snapshot 后，再决定是否恢复并启动 `modeldev_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1`，或按 TCNLSTM anchor-first / actuator context / FiLM-gate / H20 conservative 决策树调整。

## 最新代码/协议变更（2026-05-15 23:40 CST）

- 改动文件：`scripts/dynamics_learning/models/tcnlstm.py`、`scripts/dynamics_learning/registry.py`。
- `TCNLSTM.forward()` 现在接受 `context_hist=None`，避免 `DynamicsLearning.forward(..., context_hist=...)` 在 TCNLSTM 上 TypeError。
- 新增默认关闭的 `adaptive_history_context` 支持：`tcnlstm_side_history_*` 分支把 H50 raw state/control、SO(3) geometric motion delta 和 past-only `dmot/vbat` 编码为 side context，通过 `null/short/mid/full` selector 与 reliability gate 产生 zero-init residual 注入 `head_input`。旧 checkpoint 初始为 no-op；旧路径不启用时行为不变。
- `registry.py` 将 `adaptive_history_context`、窗口参数和 `history_context_dim` 传给 TCNLSTM。检查：本地/远程 `py_compile` 和 `git diff --check` 通过；local/remote SHA256 一致；远程 dummy forward 与 one-batch smoke 通过。
- 2026-05-16 补充默认不改变旧行为的 null-trust 开关：`--tcnlstm_side_history_scale_init`（默认 `0.05`）和 `--tcnlstm_side_history_selector_prior uniform|null_short`（默认 `uniform`）。本地/远程 `py_compile`、`git diff --check` 和 H10 null-trust smoke 均通过。

## 刚停止候选（2026-05-16 10:18 CST）

- `modeldev_20260515_tcnlstm_geoactctx_H50_from_attitude_e3_p1`：early_stopped=true，best=`0.4656468630`（epoch=2），e2 `valid_q_loss_epoch=0.0416224226`、`valid_v_loss_epoch=0.1515640318`、`valid_omega_loss_epoch=0.2400307506`，对比 TCNLSTM H10 attitude baseline 整体回退，按 gate 不跑 horizon/test；训练 tmux 已退出，GPU watch 已清理，artifacts 保留。

## 刚停止候选（2026-05-15 23:32 CST）

- `modeldev_20260515_grutcn_adaptivehist_H50_from_vomega_e3_p1` 已按 e1 validation gate 停止，不跑 horizon/test。e0->e1 `valid_loss_epoch=0.6879014373 -> 0.6870185733`，e1 `valid_v=0.1602650136`、`valid_q=0.0390152447`、`valid_omega=0.2308619916`、`valid_state_mse=0.2525024712`，仍明显差于 current GRUTCN best；adaptive gate e1 `null=0.0084`、`short=0.7560`、`full=0.1937`、`gate_saturation=0.3051`，长历史 side branch 没显示泛化收益。训练/watch tmux 已停止，artifacts 保留。

## 最新代码/协议变更（2026-05-15 16:45 CST）

- 改动文件：`scripts/dynamics_learning/models/grutcn.py`、`scripts/config.py`、`scripts/dynamics_learning/registry.py`、`scripts/dynamics_learning/lighting.py`、`scripts/train.py`。
- 新增 `--adaptive_history_context`、`--adaptive_history_short_window`、`--adaptive_history_mid_window`，默认关闭保持旧行为。GRUTCN 新增 checkpoint-safe `adaptive_history_*`：H20 anchor 主路径不变，长历史只走 raw-token Transformer side branch、`null/short/mid/full` selector、reliability gate、zero-init residual。
- `DynamicsLearning` 新增 `log_adaptive_history_stats()`，训练/验证 CSVLogger 记录 gate mean/std/saturation 与 reliability mean/std/saturation；`train_summary.json` 写入 adaptive history 配置。
- 检查/同步：本地 `py_compile` 通过；远程 `python3 -m py_compile` 通过；远程 conda env dummy forward finite；远程脚本级 smoke 通过。同步中曾误建远程 `scripts/scripts/` 副本，已只删除该误同步副本并重新同步正确路径。

## 代码/协议变更（2026-05-15 14:10 CST）

- 改动文件：`scripts/dynamics_learning/models/grutcn.py`、`scripts/config.py`、`scripts/dynamics_learning/data.py`、`scripts/dynamics_learning/registry.py`、`scripts/dynamics_learning/lighting.py`、`scripts/train.py`、`scripts/eval.py`。
- 新增 `--history_context_mode none|dmot_vbat`，默认 `none` 保持旧行为。`dmot_vbat` 模式下 dataloader 只选历史 `dmot/vbat` 作为 `context_hist`；`DynamicsLearning.full_state_rollout()` 将 `context_hist` 传给模型；GRUTCN 新增 gated/zero-init `history_context_*` 分支修正 `history_context`；train/eval summary 记录 `history_context_mode` 与 `history_context_dim`。
- 检查/同步：本地和远程 `py_compile` 均通过；local/remote SHA256 一致；远程 smoke 通过，确认 `history_context_dim=5`、trainable count `19`、one-batch train/valid finite。

## 刚停止候选（2026-05-15 16:05 CST）

- `modeldev_20260515_grutcn_dmotvbatctx_H20_from_vomega_e3_p1` 已按 validation gate 停止，不跑 horizon/test。e0/e1 `valid_loss_epoch=0.6827901006 -> 0.6828653216`，e1 `valid_v=0.1596235037`、`valid_omega=0.2291074544`、`valid_q=0.0388034955`、`valid_state_mse=0.2502800524`，相对 current best 的 v/omega/q/state_mse 均无正信号。训练 tmux/GPU watch 与匹配实验路径的训练进程已停止，artifacts 保留。
- `modeldev_20260515_grutcn_rawtokgeo_H20_from_vomega_e3_p1` 已按 validation gate/early stop 判失败，不跑 horizon/test。训练 early-stopped，best 停在 e0：e0/e1/e2 `valid_loss_epoch=0.6829264164 -> 0.6830326319 -> 0.6831049919`，e2 `valid_v=0.1597713530`、`valid_q=0.0388241075`、`valid_omega=0.2290891409`、`valid_state_mse=0.2502748668`；相对 current best 的 `valid_loss/v/q/state_mse` 均无正信号。训练 tmux 已退出，GPU watch 已停止，artifacts 保留；horizon/test 未读取。结论：SO(3) raw-token representation fix 单独未带来 validation 正信号，不做 rawtokgeo continuation，转 `dmot/vbat` context。
- `modeldev_20260515_grutcn_softvomega_H20_from_hard_e1_p1` 已按 e0 validation gate 停止，不跑 horizon/test。e0 `valid_v=0.1629530936`、`valid_omega=0.2299810797`、`valid_q=0.0454349183`、`valid_state_mse=0.2749448121`，相对 hard e1 和当前 GRUTCN best 均明显恶化。只停止对应训练/GPU watch tmux，artifacts 保留。
- `modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1` 已按 validation gate 停止，不跑 horizon/test。e0/e1 相对当前 GRUTCN best 参考 `valid_v=0.1595940590`、`valid_omega=0.2290888280`、`valid_q=0.0388018563` 均未反转；e1 `valid_loss_epoch=0.4708360136`、`valid_v=0.1596593559`、`valid_q=0.0388144702`、`valid_omega=0.2290989012`。只停止对应训练/GPU watch tmux，artifacts 保留。

## 当前代码/协议状态（2026-05-14 17:55 CST）

- `eval no-silent-skip guard` 已实现并同步远程：`scripts/eval.py` 在 eval 开始前检查 `unroll_length >= max(eval_horizons)`；`horizon_summary.json` 记录 `actual_unroll_length`、`requested_eval_horizons`、`computed_eval_horizons`、`skipped_eval_horizons` 和 `state_update_mode`。远程 smoke 确认 `unroll_length=2 + eval_horizons=1,10,25,50` 会直接失败，不再 silent skip h50。
- `state_update_mode` 已实现并同步远程：`scripts/config.py` 新增 `--state_update_mode residual_full_state|hard_vomega_kinematic|soft_vomega_kinematic` 与 `--state_update_soft_residual_scale`；`scripts/dynamics_learning/lighting.py` 中默认 `residual_full_state` 完全保留旧行为，`hard_vomega_kinematic` 只用 `delta_v/delta_omega` 更新 v/omega 并用梯形积分恢复 p/q，`soft_vomega_kinematic` 在 hard 结果上加小 p/theta residual；`scripts/train.py` 将两个字段写入 `train_summary.json`。
- 验证结果：本地与远程 `py_compile` 均通过；`rsync -avR` 已同步 touched files 和状态文档到 `/home/ubuntu/Developer/long-horizon-dynamics`；本地/远程 SHA256 一致；远程 smoke 确认 50 步 summary 字段正常，hard/soft kinematic update 输出 finite，quaternion norm 为 1。
- 接手建议：hard/soft vomega 已完成并按 validation gate 失败，不继续同类训练。当前先完成 rawtokgeo；若没有 horizon/test 聚合收益但 validation 接近 current best，最多允许一次 conservative trainable scope 或短 continuation；随后转 history-only `dmot/vbat` context branch。

## GPT Pro 策略更新（2026-05-15，覆盖优先级）

用户提供了 `/Users/lixiang/Desktop/codex_priority_reorder_keep_seq2seq_last.md`。接手 agent 应把它作为当前路线优先级，而不是继续按 2026-05-14 的“Delta v / Delta omega 主预测优先”推进。

核心判断：

- hard/soft vomega 失败只能说明当前“在已有 full-state residual head 上硬接/软接 v/omega kinematic update”的实现失败，不能否定 `Delta v / Delta omega` 父级思想。
- `true seq2seq Delta v / Delta omega` 更像 finite-horizon future-control decoder，不应作为当前第一创新点；它放到最后 fallback / late-stage extension。
- 论文主叙事先走 neural system identification / learned dynamics 更自然的三点：SO(3)-aware geometric history representation、history-only hidden actuator context、自适应历史选择。

执行优先级：

1. 完成当前 `raw_token_geometric_delta`。
2. 如果 rawtokgeo 有接近正信号，最多做一次 conservative scope 或短 continuation。
3. 做 history-only `dmot/vbat` context branch：只用历史 `dmot,vbat`，禁止 future context，首轮不使用 `a/alpha`，zero-init/gated side branch。
4. 做 adaptive history selector / multi-scale reliability gate：短窗 anchor 保持 H20，长历史只进 gated side branch，记录 gate mean/std/saturation。
5. 将成熟的 geometry/context trick 迁移到 `tcnlstm.py`，暂不迁移 hard/soft vomega。
6. 只有上述方向均没有 horizon/test 聚合收益时，才启动完整 true seq2seq `Delta v[1:F] / Delta omega[1:F]` predictor；该候选至少给 1-2 个完整 validation 周期，不能用 e0 直接杀死。

## GPT Pro 建议并入当前路线（2026-05-14）

用户提供了 Pro 的外部建议文档 `/Users/lixiang/Desktop/codex_delta_vomega_kinematic_rollout_plan.md`。其中 eval guard、`state_update_mode` 和 raw-token 几何修正仍保留；但其“优先验证 `Delta v / Delta omega` 主预测”的排序已被 2026-05-15 策略更新覆盖。

核心判断：

- 现在的 GRUTCN 开发期 best 约为 `MSE_1_to_F=0.3993478401`、h50 `E_q=0.0852212122`、`E_v=0.3800343019`、`E_omega=0.3008244657`，距离 strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）仍很远。
- 继续做低 LR continuation、v/omega loss reweight、latent SE tiny residual、weak physics loss continuation，大概率只能带来 `1e-5 ~ 1e-4` 量级收益。
- 真正要验证的结构假设是：主模型学习 `Delta v / Delta omega` dynamics，`p/q` 主要由运动学恢复，而不是继续让网络自由预测完整 `[p,v,q,omega]` residual。

历史执行状态：

- 已完成协议 guard：`eval.py` 禁止 long-horizon eval silent skip；若请求 h50，`unroll_length` 必须覆盖 50；summary 会记录 actual/requested/computed/skipped horizons。
- 已完成 `state_update_mode`：`residual_full_state` 保持旧行为，`hard_vomega_kinematic` / `soft_vomega_kinematic` 可作为 ablation，但当前 hard/soft 候选均已按 validation gate 失败，不继续作为优先路线。
- 已完成 raw-token 几何 delta 实现并启动当前 active rawtokgeo。后续排序以 2026-05-15 策略更新为准：rawtokgeo -> 最多一次保守 continuation -> history-only `dmot/vbat` context -> adaptive history selector -> TCNLSTM 迁移 -> true seq2seq 最后 fallback。

## 1. 当前 sprint 目标

用户在 2026-05-10 设定 12 小时 sprint。目标不是 validation loss 好看，而是让至少一个新 latent-context dynamics 模型在 locked long-horizon rollout 指标上击败旧模型/基线。

2026-05-13 目标更新：两个新模型 `grutcn` / `tcnlstm` 不再限制可见历史长度 `history_length`。后续实验可以使用更长历史、多尺度历史、自适应历史选择或更强历史记忆结构，只要不泄漏数据；最终目标是超过原始 baseline 的最佳指标包络，而不是必须在同一 `H10` / `H20` 条件下公平配对比较。若候选使用更长历史，实验记录和结论里要明确标注为 history-expanded。历史长度不能简单当作越长越好；优先考虑 gate、attention over history、multi-scale pooling、reliability/slack 权重等机制，让模型自动选择有效历史并降低过长历史噪声。
2026-05-14 当前无 active training/evaluation。`modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1` 的 e2 horizon/test MSE-schema eval 已完成，但因为 `unroll_length=2`，`h10/h25/h50` 被脚本跳过，只产出 `h=1` / `mean_1_to_F`，因此不能直接作为 long-horizon locked audit 胜利。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1`；evaluated checkpoint：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1/checkpoints/model-epoch-02-best_valid_loss=0.17.pth`；eval tmux `eval_grutcn_multistepdelta_H20_e2_mse_p1` 和 GPU watch `gpu_watch_eval_grutcn_multistepdelta_H20` 已退出。训练结果：`train_summary.json` 显示 `best_valid_loss=0.1660599709`、`valid_v_loss_epoch=0.0156964753`、`valid_omega_loss_epoch=0.0856593177`、`valid_q_loss_epoch=0.0021088682`、`valid_state_mse_epoch=0.0295935068`，trainable 参数仅 `model.multi_step_*`；smoke 通过 checkpoint-safe load 和 one-batch finite，缺失仅新增 `multi_step_*` key，无旧主路径 shape mismatch。已读取 horizon/test：`average rollout loss=0.17696784436702728`，`MSE_1_to_F=0.029530191901839857`，但只覆盖 2 步展开。结论：该 multi-step 结构在训练/验证上健康，但当前 eval 不能证明 long-horizon 胜利；它更适合作为可保留的结构 ablation，若后续继续该方向，应先把 unroll/eval 改到能产出长 rollout 指标，再做 locked audit 决策。
2026-05-14 当前下一候选已从同类 reweight/latentse 微调切到真正的 `multi-step delta_v/delta_omega predictor`。已完成代码准备和 smoke：`smoke_20260514_grutcn_multistepdelta_H20_from_vomega_e3_p1` 从 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_cont2_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth` 成功加载，缺失项仅为新分支 `multi_step_*` key，无旧主路径 shape mismatch；trainable 参数仅 `model.multi_step_*`，一批 `train_loss_step=0.0523`、`valid_loss_epoch=0.0194`、`best_valid_loss=0.0194001738`。未读取 horizon/test metric，artifacts 保留。代码/协议改动集中在 `scripts/dynamics_learning/models/grutcn.py`、`scripts/config.py`、`scripts/dynamics_learning/registry.py`、`scripts/eval.py`、`scripts/train.py`；`scripts/dynamics_learning/lighting.py` 维持 full_state rollout contract 暂不改。下一步是在远端 tmux 启动正式训练，优先观察 e0/e1 的 unweighted `valid_v/valid_omega/valid_q` 是否继续走强，再决定是否跑 horizon/test locked audit。

主要目标文件：

- `scripts/dynamics_learning/models/grutcn.py`
- `scripts/dynamics_learning/models/tcnlstm.py`

优先策略：

- 优先只改这两个 model 文件。
- 如果纯 model 文件不够，可以改让这两个模型正确运行、训练、测试、评估所必需的 train/test/eval wiring。
- 可以使用 `/Users/lixiang/Documents/Obsidian Vault/trick/trick.md`，也可以 web research 文献、benchmark、复杂结构、模块或训练技巧。
- 2026-05-12 起，允许根据 validation 结果、训练曲线、horizon/test 聚合指标和误差模式对 `grutcn` / `tcnlstm` 做较大模型改造，只要改造方向更复杂、更有表达力、更有论文故事且可能提升指标；不要只局限在小 adapter。
- 不得泄漏数据：不能上传私有数据集、原始轨迹、标签、checkpoint、完整日志或可复现样本；不能把 validation/test/horizon 的样本、标签或轨迹硬编码进模型、训练流程或提示词；只允许用聚合指标和误差趋势做结构决策。
- 任何 trick 或外部研究只要改变模型/训练/测试/评估行为，就必须记录来源、理由、改动文件/协议和观察结果，并明确告诉用户。

成功定义：

- 允许用 validation 与 horizon/test 指标共同挑选候选 checkpoint 和结构方向。
- 不限制新模型可见历史长度；允许通过扩大 `history_length`、引入多尺度历史结构或自适应历史选择提高最终指标，但要监控过长历史噪声。
- 至少击败一个重要 long-horizon target，最好多个指标全面提升。
- validation-only improvement 不是成功；horizon/test 指标改善才是模型冲刺的核心证据。
- 若需要声明“完全未见测试集”的最终结果，必须另设新的 holdout 或明确标注为开发期 horizon/test-guided 结果；当前 sprint 优先推进模型效果。
- failed audit 或 failed horizon/test-guided screening 后任务仍未完成，不能停止整体推进。

## 2. 不可违背规则

- 远程训练必须跑在 `gpu4060` 的 tmux 里。
- 活跃实验不等完整训练结束才分析；当前用户已要求自动巡检按总控 automation 的 30 分钟节奏检查 tmux/进程/GPU、日志、CSVLogger、checkpoint、summary 和 horizon/test 结果；若出现临界 gate、OOM、NaN、进程异常或 audit，可临时加密人工检查。
- 自 2026-05-11 起，训练/开发阶段允许读取 horizon/test metric 内容，并允许用它来挑选候选、调整结构和改进模型。
- 每次读取 horizon/test metric 都必须记录实验 id、checkpoint、关键指标、是否影响后续调参/结构决策，以及结论。
- 失败实验确认后才可清理，而且只能清理该实验的进程、tmux 会话和产物；不能删除无关文件、源码、数据集、基线、成功实验或汇总报告。
- `Prompt.md` 只写长期有用摘要，不写普通巡检流水账；完整历史过长时归档到 `docs/archive/`。
- 所有项目文档、交接记录、实验记录默认中文；代码标识、命令、路径、实验 id、指标名和必要英文术语保持原样。
- 写代码要简洁、可读，避免防御性过度工程。
- 重大策略节点不要单 agent 闷头推进；使用结果分析 agent、架构/代码 agent、实验调度 agent 交叉检查。

## 3. 项目路径和运行环境

本地：

- Mac repo：`/Users/lixiang/Developer/long-horizon-dynamics`
- Mac repo 是代码源头。

远程：

- SSH alias：`gpu4060`，当前默认走 Tailscale IP `100.106.154.6`
- 备用 alias：`gpu4060-ts` 同样走 Tailscale；`gpu4060-lan` 保留旧局域网 IP `192.168.1.108`
- repo：`/home/ubuntu/Developer/long-horizon-dynamics`
- conda env：`dynamics_learning`
- GPU：RTX 4060，约 8GB VRAM

同步原则：

- 代码从 Mac 同步到 remote。
- 不要用同步覆盖远程 `resources/experiments/`。
- 同步后验证 touched file 的 local/remote hash。
- 远程实验结果需要单独 pull 回本地，不要反向手改远程代码后忘记同步。

## 4. 给用户 Linux/SSH 指令的格式

用户要求学习排查流程。给 Linux/SSH 指令时必须说明：

- 我要执行什么。
- 为什么执行。
- 这条命令的关键语法是什么意思。

例子：检查远程 tmux 时，应说明 `tmux ls` 是列出当前 tmux session；如果看到实验 session，说明后台终端还活着。

## 5. 模型故事必须保留

新模型叙事必须围绕 latent context：

- 从历史状态/控制 `[x,u]` 学习 latent context。
- 用 latent context 恢复当前状态 `x_t` 单独无法描述的未观测动态。
- 物理动机包括 actuator lag、aerodynamic state、hidden disturbances、filter state。
- 文献叙事可连接 Mohajerin 2018 风格 RNN hidden-state initialization 和 Serrano 2024 风格 encoder context extraction。

不要把 `grutcn` 或 `tcnlstm` 简化成平凡 baseline。结构变化只能更强、更有表达力、更有论文故事，不能削弱。

## 6. 固定基线目标

来源：远程 `horizon_results.csv`，最后完整基线读取为 2026-05-07。

| config | best_valid_loss | h50_E_q | h50_E_v | h50_E_omega | mean_E_q | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `gru_H10_F50_seed10` | `0.580389` | `0.0800042` | `0.356014` | `0.283807` | `0.0426877` | 当前最佳 h50 quaternion。 |
| `gru_H20_F50_seed10` | `0.573358` | `0.0802197` | `0.353015` | `0.260392` | `0.0420377` | 当前最佳 h50 velocity、omega、mean quaternion、sum quaternion。 |
| `tcn_H10_F50_seed10` | `0.533292` | `0.0891246` | `0.347036` | `0.327506` | `0.0452568` | validation 强，但 h50 quaternion 不如 GRU。 |
| `mlp_H1_F50_seed10` | `0.704774` | `0.174414` | `0.588885` | `0.579955` | `0.0847114` | h1 强，long horizon 弱。 |

当前最重要的 locked target：

- h50 quaternion：低于 `0.0800042`。
- h50 velocity：低于 `0.353015`。
- h50 omega：低于 `0.260392`。
- mean quaternion：低于 `0.0420377`。

## 7. 当前状态

- 最新接手状态（2026-05-14 11:35 CST）：当前无 active training/evaluation。`modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1` e2 horizon/test MSE-schema eval 已完成：远程路径 `/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1`；evaluated checkpoint：`checkpoints/model-epoch-02-best_valid_loss=0.17.pth`；eval tmux `eval_grutcn_multistepdelta_H20_e2_mse_p1` 已退出，GPU watch `gpu_watch_eval_grutcn_multistepdelta_H20` 已停止；eval log `logs/eval_horizontest_20260514_grutcn_multistepdelta_H20_e2_b32_mse_p1.log`，GPU watch log `logs/gpu_watch_eval.log`。训练已自然结束，`train_summary.json` 显示 `best_valid_loss=0.1660599709`、`valid_v_loss_epoch=0.0156964753`、`valid_omega_loss_epoch=0.0856593177`、`valid_q_loss_epoch=0.0021088682`、`valid_state_mse_epoch=0.0295935068`，trainable 参数仅 `model.multi_step_*`；smoke 通过 checkpoint-safe load 和 one-batch finite，缺失仅新增 `multi_step_*` key，无旧主路径 shape mismatch。已读取 horizon/test，但脚本由于 `unroll_length=2` 只展开到 2 步：`average rollout loss=0.17696784436702728`，`h=1` 的 `E_p/E_v/E_q/E_omega/MSE_x` 已记录，`MSE_1_to_F=0.029530191901839857`；`h=10/25/50` 被跳过，因此不能作为 long-horizon locked audit 胜利或失败定论。下一步若继续 multi-step 路线，先把 `unroll_length` 修正到能覆盖长 horizon，再做新的验证。
- 最新接手状态（2026-05-14 09:18 CST）：当前无 active training/evaluation。刚完成 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1` e3 horizon/test MSE-schema eval p1 并读取 metric。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`，training 加权 `best_valid_loss=0.6827782393`，eval log `logs/eval_horizontest_20260514_grutcn_vomegaweight_cont_H20_e3_b32_mse_p1.log`，average rollout loss `0.8086838126`（加权 loss，不与未改权重实验直接比较）；h50 `E_q=0.0852212122`、`E_v=0.3800343019`、`E_omega=0.3008244657`、`MSE_x=0.7797312295`；mean `E_q=0.0436578958`、`E_v=0.2156254028`、`E_omega=0.2425080560`；`MSE_1_to_F=0.3993478401`。相对 vomegaweight e3，`MSE_1_to_F 0.3993592611 -> 0.3993478401`，h50 `E_v 0.3800689345 -> 0.3800343019`，h50 `E_omega 0.3008467932 -> 0.3008244657`，mean q/v/omega 与 h50 `MSE_x` 继续小幅改善；h50 `E_q` 第 8 位略回退，基本持平。结论：这是新的 GRUTCN 开发期 best，但提升已是 `1e-5` 量级且仍远落后 strongest GRU targets；该 horizon/test 读取影响后续结构决策：停止同类 reweight/latentse 微调，下一步转真正 `multi-step delta_v/delta_omega predictor`。eval/watch 已停止，artifacts 保留，无清理。下一步执行：先在 `grutcn.py` 设计 checkpoint-safe `multi_step_delta_v/delta_omega` 分支，保留当前 raw-token/latent_se anchor；smoke 确认从当前 e3 checkpoint 加载时只 missing 新分支 key、one-batch train/valid finite、trainable patterns 不白训；再决定是否启动正式训练。
- 最新接手状态（2026-05-14 08:58 CST）：当前 active evaluation 是 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1` e3 horizon/test MSE-schema eval p1。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；eval tmux：`eval_grutcn_vomegaweight_cont_H20_e3_mse_p1`；eval GPU watch：`gpu_watch_eval_grutcn_vomegaweight_cont_H20`；eval log：`logs/eval_horizontest_20260514_grutcn_vomegaweight_cont_H20_e3_b32_mse_p1.log`；eval GPU watch log：`logs/gpu_watch_eval.log`；evaluated checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`（来自 `train_summary.json`）。训练已自然结束，训练 tmux 和训练 GPU watch 均退出，GPU 回落到约 `506/8188 MiB`；无 OOM/NaN/Traceback。e0-e3 加权 validation 连续刷新：`0.6827921271 -> 0.6827870011 -> 0.6827825308 -> 0.6827781796`，`best_valid_loss=0.6827782393`，`early_stopped=false`，`max_epochs=4`；e3 unweighted `valid_p_loss_epoch=0.0432733931`、`valid_v_loss_epoch=0.1595940590`、`valid_q_loss_epoch=0.0388018563`、`valid_omega_loss_epoch=0.2290888280`、`valid_state_mse_epoch=0.2502373457`。因为本实验改了 loss 权重，`valid_loss` 不能与未改权重实验直接比较；gate 只看 unweighted v/omega/q。相对 vomegaweight e3 参考 `valid_v=0.1596023738`、`valid_omega=0.2290926725`、`valid_q=0.0388069339`，e3 的 v/omega/q 均继续小幅改善且 q 未越线；训练阶段 horizon/test metric 未读取，artifacts 保留。eval 启动检查：tmux/watch alive，GPU 约 `772/8188 MiB`、util `39%`，horizon/test metric 尚未读取。下一步：eval 完成后读取 average rollout loss、h50/mean `E_q/E_v/E_omega`、h50 `MSE_x`、`MSE_1_to_F`，比较 vomegaweight e3 `MSE_1_to_F=0.3993592611`、h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`；若没有聚合收益，转真正 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 08:18 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；训练 tmux：`modeldev_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；GPU watch：`modeldev_gpu_watch_grutcn_vomegaweight_cont_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth`。结构仍是 GRUTCN H20/F50 raw-token main + `latent_se`，不改源码，physics OFF；loss weights 仍为 `lambda_p=0.6`、`lambda_v=1.6`、`lambda_q=0.9`、`lambda_omega=1.6`；训练配置为 epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-8`，`cosine_lr=3e-8`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。巡检结果：train/watch alive，GPU 约 `6586/8188 MiB`、util `21%`，无 OOM/NaN/Traceback。e0/e1/e2 validation 已生成并通过 gate：加权 `valid_loss_epoch=0.6827921271 -> 0.6827870011 -> 0.6827825308`，`best_valid_loss=0.6827824116`；unweighted e2 `valid_p_loss_epoch=0.0432737619`、`valid_v_loss_epoch=0.1595954895`、`valid_q_loss_epoch=0.0388026014`、`valid_omega_loss_epoch=0.2290894836`、`valid_state_mse_epoch=0.2502363622`。因为本实验改了 loss 权重，`valid_loss` 不能与未改权重实验直接比较；gate 只看 unweighted v/omega/q。相对 vomegaweight e3 参考 `valid_v=0.1596023738`、`valid_omega=0.2290926725`、`valid_q=0.0388069339`，e2 的 v/omega/q 均继续小幅改善且 q 未越线；state_mse 小幅上升但不是当前 gate 主指标；checkpoint `model-epoch=00/01/02-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；horizon/test metric 未读取。下一步：继续观察 natural finish；若后续 v/omega 信号消失或 `valid_q >0.03882`，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`；若 natural finish 后仍保持正信号，跑/read MSE-schema horizon/test，并比较 vomegaweight e3 `MSE_1_to_F=0.3993592611`、h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`。
- 最新接手状态（2026-05-14 07:38 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；训练 tmux：`modeldev_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；GPU watch：`modeldev_gpu_watch_grutcn_vomegaweight_cont_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth`。结构仍是 GRUTCN H20/F50 raw-token main + `latent_se`，不改源码，physics OFF；loss weights 仍为 `lambda_p=0.6`、`lambda_v=1.6`、`lambda_q=0.9`、`lambda_omega=1.6`；训练配置为 epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-8`，`cosine_lr=3e-8`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。巡检结果：train/watch alive，GPU 约 `6580/8188 MiB`、util `28%`，无 OOM/NaN/Traceback。e0/e1 validation 已生成并通过 gate：加权 `valid_loss_epoch=0.6827921271 -> 0.6827870011`，`best_valid_loss=0.6827870011`；unweighted e1 `valid_p_loss_epoch=0.0432742499`、`valid_v_loss_epoch=0.1595965624`、`valid_q_loss_epoch=0.0388036631`、`valid_omega_loss_epoch=0.2290903032`、`valid_state_mse_epoch=0.2502349019`。因为本实验改了 loss 权重，`valid_loss` 不能与未改权重实验直接比较；gate 只看 unweighted v/omega/q。相对 vomegaweight e3 参考 `valid_v=0.1596023738`、`valid_omega=0.2290926725`、`valid_q=0.0388069339`，e1 的 v/omega/q 均继续小幅改善且 q 未越线；checkpoint `model-epoch=00/01-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；horizon/test metric 未读取。下一步：继续观察 e2/natural finish；若后续 v/omega 信号消失或 `valid_q >0.03882`，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`；若 natural finish 后仍保持正信号，跑/read MSE-schema horizon/test，并比较 vomegaweight e3 `MSE_1_to_F=0.3993592611`、h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`。
- 最新接手状态（2026-05-14 06:58 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；训练 tmux：`modeldev_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；GPU watch：`modeldev_gpu_watch_grutcn_vomegaweight_cont_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth`。结构仍是 GRUTCN H20/F50 raw-token main + `latent_se`，不改源码，physics OFF；loss weights 仍为 `lambda_p=0.6`、`lambda_v=1.6`、`lambda_q=0.9`、`lambda_omega=1.6`；训练配置为 epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-8`，`cosine_lr=3e-8`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。巡检结果：train/watch alive，GPU 约 `6580/8188 MiB`、util `39%`，无 OOM/NaN/Traceback。e0 validation 已生成并通过 gate：加权 `valid_loss_epoch=0.6827921271`、`best_valid_loss=0.6827921867`；unweighted e0 `valid_p_loss_epoch=0.0432747900`、`valid_v_loss_epoch=0.1595973969`、`valid_q_loss_epoch=0.0388051271`、`valid_omega_loss_epoch=0.2290917784`、`valid_state_mse_epoch=0.2502326965`。因为本实验改了 loss 权重，`valid_loss` 不能与未改权重实验直接比较；gate 只看 unweighted v/omega/q。相对 vomegaweight e3 参考 `valid_v=0.1596023738`、`valid_omega=0.2290926725`、`valid_q=0.0388069339`，e0 的 v/omega/q 均继续小幅改善且 q 未越线；checkpoint `model-epoch=00-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；horizon/test metric 未读取。下一步：继续观察 e1/e2/natural finish；若后续 v/omega 信号消失或 `valid_q >0.03882`，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`；若 natural finish 后仍保持正信号，跑/read MSE-schema horizon/test，并比较 vomegaweight e3 `MSE_1_to_F=0.3993592611`、h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`。
- 最新接手状态（2026-05-14 05:58 CST）：当前 active training 为 `modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；训练 tmux：`modeldev_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`；GPU watch：`modeldev_gpu_watch_grutcn_vomegaweight_cont_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth`。结构仍是 GRUTCN H20/F50 raw-token main + `latent_se`，不改源码，physics OFF；保持 loss weights `lambda_p=0.6`、`lambda_v=1.6`、`lambda_q=0.9`、`lambda_omega=1.6`。配置：epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-8`，`cosine_lr=3e-8`，`warmup_steps=50`，`cosine_steps=800`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。启动检查：train/watch alive，日志显示成功加载 vomegaweight e3 checkpoint，trainable patterns 正确且不含 `raw_token_adaptive_*`，GPU 约 `6576/8188 MiB`、util `32%`，无 OOM/NaN/Traceback；尚无 validation row，horizon/test metric 未读取。Gate：当前 valid_loss 是加权 loss，不与未改权重 valid_loss 直接比较；参考 vomegaweight e3 unweighted `valid_v=0.1596023738`、`valid_omega=0.2290926725`、`valid_q=0.0388069339`。若 continuation e0 `valid_v >=0.1596024` 且 `valid_omega >=0.2290927`，或 `valid_q >0.03882`，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`；若 v/omega 继续下降且 q 可控，natural finish 后跑/read MSE-schema horizon/test，并比较 vomegaweight e3 `MSE_1_to_F=0.3993592611`、h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`。
- 刚完成 eval（2026-05-14 05:51 CST）：`modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1` e3 horizon/test MSE-schema eval p1 已完成并读取 metric。checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`，training 加权 `best_valid_loss=0.6828041077`，eval log `logs/eval_horizontest_20260514_grutcn_vomegaweight_H20_e3_b32_mse_p1.log`，average rollout loss `0.8087291121`（加权 loss，不与未改权重实验直接比较）；h50 `E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`、`MSE_x=0.7797522271`；mean `E_q=0.0436584481`、`E_v=0.2156444532`、`E_omega=0.2425160258`；`MSE_1_to_F=0.3993592611`。相对 cont2 e3，`MSE_1_to_F 0.3993805727 -> 0.3993592611`，h50 q/v/omega 和 mean q/v/omega 全部小幅改善，是新的 GRUTCN 开发期 best。该 horizon/test 读取影响后续决策：允许一次更低 LR 同权重 continuation 验证收益是否延续；若 continuation 无聚合收益，停止同类 reweight/latentse 微调并转真正 `multi-step delta_v/delta_omega predictor`。eval/watch 已退出，artifacts 保留，无清理。
- 最新接手状态（2026-05-14 05:18 CST）：当前 active evaluation 为 `modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1` e3 horizon/test MSE-schema eval p1。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`；eval tmux：`eval_grutcn_vomegaweight_H20_e3_mse_p1`；eval GPU watch：`gpu_watch_eval_grutcn_vomegaweight_H20`；eval log：`logs/eval_horizontest_20260514_grutcn_vomegaweight_H20_e3_b32_mse_p1.log`；eval GPU watch log：`logs/gpu_watch_eval.log`；evaluated checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`（来自 `train_summary.json`）。训练已自然完成，训练 tmux 和训练 GPU watch 均已退出，GPU 回落到约 `501/8188 MiB`；无 OOM/NaN/Traceback。e0-e3 加权 validation 连续刷新：`0.6828369498 -> 0.6828231215 -> 0.6828121543 -> 0.6828041673`，`best_valid_loss=0.6828041077`，`early_stopped=false`，`max_epochs=4`；e3 unweighted 分项 `valid_p_loss_epoch=0.0432761610`、`valid_v_loss_epoch=0.1596023738`、`valid_q_loss_epoch=0.0388069339`、`valid_omega_loss_epoch=0.2290926725`、`valid_state_mse_epoch=0.2502301335`。因为本实验改了 loss 权重，`valid_loss` 不能与 cont2 `0.470826` 直接比较；gate 只看 unweighted v/omega/q。相对 cont2 e3 参考 `valid_v=0.1596247554`、`valid_omega=0.2291012257`、`valid_q=0.0388166271`，e3 的 v/omega/q 均继续改善且 `valid_q <0.03884`，满足预声明 eval 条件；训练阶段 horizon/test metric 未读取。eval 启动检查：tmux/watch alive，eval log 到 `Seed set to 10`，GPU 约 `771/8188 MiB`、util `38%`，horizon files 尚未生成，horizon/test metric 尚未读取。下一步：eval 完成后读取 average rollout loss、h50/mean `E_q/E_v/E_omega`、h50 `MSE_x`、`MSE_1_to_F`，并比较 cont2 e3 `MSE_1_to_F=0.3993805727`、h50 `E_q=0.0852223037`、`E_v=0.3801424071`、`E_omega=0.3008873714`；若无聚合收益，转真正 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 04:58 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`。训练 tmux `modeldev_grutcn_vomegaweight_H20_from_cont2_e3_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_vomegaweight_H20` 均 alive；GPU 约 `6580/8188 MiB`、util `40%`；无 OOM/NaN/Traceback。e0/e1/e2 validation 连续刷新：加权 `valid_loss_epoch=0.6828369498 -> 0.6828231215 -> 0.6828121543`，`best_valid_loss=0.6828124523`；unweighted e2 分项 `valid_p_loss_epoch=0.0432770699`、`valid_v_loss_epoch=0.1596052945`、`valid_q_loss_epoch=0.0388082638`、`valid_omega_loss_epoch=0.2290939987`、`valid_state_mse_epoch=0.2502283454`。因为本实验改了 loss 权重，`valid_loss` 不能与 cont2 `0.470826` 直接比较；gate 只看 unweighted v/omega/q。相对 cont2 e3 参考 `valid_v=0.1596247554`、`valid_omega=0.2291012257`、`valid_q=0.0388166271`，e2 的 v/omega/q 均继续改善且 `valid_q <0.03884`，判定通过 gate；state_mse 小幅上浮但未触发停止条件。checkpoint `model-epoch=00/01/02-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，horizon/test metric 未读取。下一步：继续观察 natural finish；若 natural finish 后仍保持 v/omega 正信号并 q 可控，跑/read horizon/test MSE-schema；若后续正信号消失或 q 明显回退，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 04:18 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`。训练 tmux `modeldev_grutcn_vomegaweight_H20_from_cont2_e3_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_vomegaweight_H20` 均 alive；GPU 约 `6580/8188 MiB`、util `30%`；无 OOM/NaN/Traceback。e0/e1 validation 已出并刷新：加权 `valid_loss_epoch=0.6828369498 -> 0.6828231215`，`best_valid_loss=0.6828230023`；unweighted e1 分项 `valid_p_loss_epoch=0.0432783440`、`valid_v_loss_epoch=0.1596084684`、`valid_q_loss_epoch=0.0388101935`、`valid_omega_loss_epoch=0.2290958017`、`valid_state_mse_epoch=0.2502252459`。因为本实验改了 loss 权重，`valid_loss` 不能与 cont2 `0.470826` 直接比较；gate 只看 unweighted v/omega/q。相对 cont2 e3 参考 `valid_v=0.1596247554`、`valid_omega=0.2291012257`、`valid_q=0.0388166271`，e1 的 v/omega/q 均继续改善且 `valid_q <0.03884`，判定通过 gate；state_mse 微升但未触发停止条件。checkpoint `model-epoch=00/01-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，horizon/test metric 未读取。下一步：继续观察 e2/natural finish；若 natural finish 后仍保持 v/omega 正信号并 q 可控，跑/read horizon/test MSE-schema；若后续正信号消失或 q 明显回退，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 03:38 CST）：当前 active training 仍为 `modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`。训练 tmux `modeldev_grutcn_vomegaweight_H20_from_cont2_e3_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_vomegaweight_H20` 均 alive；GPU 约 `6580/8188 MiB`、util `30%`；无 OOM/NaN/Traceback。e0 validation 已出：加权 `valid_loss_epoch=0.6828369498`、`best_valid_loss=0.6828368902`；unweighted 分项 `valid_p_loss_epoch=0.0432799421`、`valid_v_loss_epoch=0.1596120149`、`valid_q_loss_epoch=0.0388131440`、`valid_omega_loss_epoch=0.2290987074`、`valid_state_mse_epoch=0.2502215207`。因为本实验改了 loss 权重，`valid_loss` 不能与 cont2 `0.470826` 直接比较；gate 只看 unweighted v/omega/q。相对 cont2 e3 参考 `valid_v=0.1596247554`、`valid_omega=0.2291012257`、`valid_q=0.0388166271`，e0 的 v/omega/q 均小幅改善且 `valid_q <0.03884`，判定通过 gate。checkpoint `model-epoch=00-best_valid_loss=0.68.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，horizon/test metric 未读取。下一步：继续观察 e1/e2/natural finish；若 natural finish 后仍保持 v/omega 正信号并 q 可控，跑/read horizon/test MSE-schema；若后续正信号消失或 q 明显回退，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 02:45 CST）：当前 active training 是前置验证 `modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`；训练 tmux：`modeldev_grutcn_vomegaweight_H20_from_cont2_e3_p1`；GPU watch：`modeldev_gpu_watch_grutcn_vomegaweight_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1/checkpoints/model-epoch=03-best_valid_loss=0.47.pth`。结构仍为 GRUTCN H20/F50 raw-token main + `latent_se`，不改源码，physics OFF；loss 权重改为 `lambda_p=0.6`、`lambda_v=1.6`、`lambda_q=0.9`、`lambda_omega=1.6`。目标是作为 `multi-step delta_v/delta_omega predictor` 的低风险前置验证：若单纯 v/omega 提权都不能在 validation submetrics 与 horizon/test 上带来更大收益，就不值得继续做同类 output-head 微调，直接转结构。配置：epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=1.5e-7`，`cosine_lr=5e-8`，`warmup_steps=50`，`cosine_steps=800`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。启动检查：tmux/watch alive，GPU 约 `6576/8188 MiB`、util `40%`；日志显示成功加载 cont2 e3 checkpoint，trainable patterns 正确且不含 `raw_token_adaptive_*`；无 OOM/NaN/Traceback，尚无 validation row，horizon/test metric 未读取。Gate：该实验的 `valid_loss` 是加权 loss，不可直接与 cont2 `0.470826` 比较；优先看 unweighted `valid_v_loss_epoch` / `valid_omega_loss_epoch` 是否低于 cont2 e3 的 `0.1596247554` / `0.2291012257`，且 `valid_q_loss_epoch` 不超过约 `0.03884`。若 e0 无 v/omega 正信号或 q 明显回退，停止不 eval 并转真正 `multi-step delta_v/delta_omega predictor`；若 v/omega 有正信号且 q 可控，natural finish 后跑/read horizon/test MSE-schema。
- 最新接手状态（2026-05-14 02:38 CST）：当前无 active training/evaluation。刚完成 `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1` e3 horizon/test MSE-schema eval p1 并读取 metric。checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4708260596`，eval log `logs/eval_horizontest_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_e3_b32_mse_p1.log`，average rollout loss `0.5625137091`；h50 `E_q=0.0852223037`、`E_v=0.3801424071`、`E_omega=0.3008873714`、`MSE_x=0.7797898716`；mean `E_q=0.0436600007`、`E_v=0.2156833140`、`E_omega=0.2425306675`；`MSE_1_to_F=0.3993805727`。相对 cont e5，`MSE_1_to_F 0.3993963243 -> 0.3993805727`，h50 `E_q 0.0852242063 -> 0.0852223037`，`E_v 0.3801910379 -> 0.3801424071`，`E_omega 0.3009120027 -> 0.3008873714`，average rollout loss `0.5625592470 -> 0.5625137091`。结论：cont2 是新的 GRUTCN 开发期 best，但改善幅度只有 `1e-5` 到 `5e-5` 量级，仍明显落后 strongest GRU targets；该读取已影响后续结构决策，停止继续同类 latentse/raw-token 低 LR 微调，转 `multi-step delta_v/delta_omega predictor` 或其低风险前置验证。eval/watch 已自然退出，artifacts 保留，无清理。
- 最新接手状态（2026-05-14 02:18 CST）：当前 active evaluation 是 `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1` e3 horizon/test MSE-schema eval p1。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`；eval tmux：`eval_grutcn_rawtoktf_latentsefix_cont2_H20_e3_mse_p1`；eval GPU watch：`gpu_watch_eval_grutcn_rawtoktf_latentsefix_cont2_H20`；eval log：`logs/eval_horizontest_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_e3_b32_mse_p1.log`；eval GPU watch log：`logs/gpu_watch_eval.log`。训练已自然结束，训练 tmux 和训练 GPU watch 均已退出，GPU 回落到约 `501/8188 MiB`；无 OOM/NaN/Traceback。e0-e3 validation 连续刷新：`0.4708608985 -> 0.4708451629 -> 0.4708339274 -> 0.4708261788`，`train_summary.json` 指向 e3 checkpoint `checkpoints/model-epoch=03-best_valid_loss=0.47.pth`，`best_valid_loss=0.4708260596`；e3 分项为 `valid_p_loss_epoch=0.0432835445`、`valid_v_loss_epoch=0.1596247554`、`valid_q_loss_epoch=0.0388166271`、`valid_omega_loss_epoch=0.2291012257`、`valid_state_mse_epoch=0.2502173185`。训练阶段未读取本实验 horizon/test metric，artifacts 保留。eval 启动检查：tmux/watch alive，log 到 `Seed set to 10`，GPU 约 `766/8188 MiB`、util `40%`；`horizon_summary.json` / `horizon_metrics.csv` 尚未生成，horizon/test metric 尚未读取。下一步：eval 完成后读取 average rollout loss、h50/mean `E_q/E_v/E_omega`、h50 `MSE_x`、`MSE_1_to_F`，比较 cont e5；若没有聚合改善，停止同类 latentse/raw-token 微调并转 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-14 01:38 CST）：当前 active training 仍是 `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont2_H20` 均 alive；GPU 约 `6574/8188 MiB`、util `30%`；无 OOM/NaN/Traceback。e0/e1/e2 validation 连续刷新：`0.4708608985 -> 0.4708451629 -> 0.4708339274`，当前 `best_valid_loss=0.4708339274`；e2 分项 `valid_p_loss_epoch=0.0432847776`、`valid_v_loss_epoch=0.1596295685`、`valid_q_loss_epoch=0.0388175435`、`valid_omega_loss_epoch=0.2291020453`、`valid_state_mse_epoch=0.2502169013`。checkpoint `model-epoch=00/01/02-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，本实验 horizon/test metric 未读取。判定：e0-e2 连续健康刷新，继续观察 natural finish；最终仍以 horizon/test `MSE_1_to_F` 和 h50 `E_q/E_v/E_omega` 是否超过 cont e5 为准。
- 最新接手状态（2026-05-14 00:58 CST）：当前 active training 仍是 `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont2_H20` 均 alive；GPU 约 `6570/8188 MiB`、util `32%`；无 OOM/NaN/Traceback。e0/e1 validation 连续刷新：`0.4708608985 -> 0.4708451629`，当前 `best_valid_loss=0.4708452523`；e1 分项 `valid_p_loss_epoch=0.0432867706`、`valid_v_loss_epoch=0.1596363783`、`valid_q_loss_epoch=0.0388187915`、`valid_omega_loss_epoch=0.2291033268`、`valid_state_mse_epoch=0.2502155006`。checkpoint `model-epoch=00/01-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，本实验 horizon/test metric 未读取。判定：e0/e1 连续健康刷新，继续观察 e2/natural finish；最终仍以 horizon/test `MSE_1_to_F` 和 h50 `E_q/E_v/E_omega` 是否超过 cont e5 为准。
- 最新接手状态（2026-05-14 00:18 CST）：当前 active training 仍是 `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont2_H20` 均 alive；GPU 约 `6570/8188 MiB`、util `30%`；无 OOM/NaN/Traceback。e0 validation 已出并刷新：`valid_loss_epoch=0.4708608985`、`best_valid_loss=0.4708608091`，分项 `valid_p_loss_epoch=0.0432894938`、`valid_v_loss_epoch=0.1596454829`、`valid_q_loss_epoch=0.0388207026`、`valid_omega_loss_epoch=0.2291049510`、`valid_state_mse_epoch=0.2502139807`。checkpoint `model-epoch=00-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成；`train_summary.json` 尚未生成，本实验 horizon/test metric 未读取。判定：e0 低于健康线 `0.47088`，继续观察 e1/e2/natural finish；最终仍以 horizon/test `MSE_1_to_F` 和 h50 `E_q/E_v/E_omega` 是否超过 cont e5 为准。
- 最新接手状态（2026-05-13 23:26 CST）：当前 active training 是最后一次同类 continuation `modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`；GPU watch：`modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont2_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1/checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。结构：GRUTCN H20/F50 raw-token main + `latent_se`，physics OFF；trainable patterns 为 `latent_se,raw_token_pos,raw_token_input_norm,raw_token_proj,raw_token_encoder,raw_token_query,raw_token_score,raw_token_context_norm,raw_token_head,raw_token_velocity,raw_token_attitude`，不训练 `raw_token_adaptive_*`。配置：epochs `4`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=1e-7`，`cosine_lr=3e-8`，`warmup_steps=50`，`cosine_steps=800`，early stopping patience `2`，min_delta `5e-6`，WANDB disabled。启动检查：第一次启动因非交互 shell `conda` 不在 PATH 立即退出，已只清理该 cont2 空运行目录/对应 tmux，用显式 `/home/ubuntu/miniconda3/bin/conda` 与 `/usr/lib/wsl/lib/nvidia-smi` 重启成功；当前训练 tmux/watch alive，日志已加载 e5 checkpoint，trainable list 正确，GPU 约 `6568/8188 MiB`，无 OOM/NaN/Traceback，尚无 validation row、checkpoint 或本实验 horizon/test metric。Gate：参考 cont e5 `best_valid_loss=0.4708843529` 和 horizon/test `MSE_1_to_F=0.3993963243`、h50 `E_q=0.0852242063`、`E_v=0.3801910379`、`E_omega=0.3009120027`；e0 `>0.47091` 或 p/v/q/omega/state_mse 同步明显回退则停止该实验、不 eval、转 `multi-step delta_v/delta_omega predictor`；e0 `<=0.47088` 或后续刷新则健康。natural finish/validation best 后跑/read horizon/test；若 `MSE_1_to_F` 或 h50 q/v/omega 未继续改善，停止同类 latentse/raw-token 微调并转结构。
- 刚完成 eval（2026-05-13 23:26 CST）：`modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` e5 horizon/test MSE-schema eval p1 已完成并读取 metric，checkpoint `checkpoints/model-epoch=05-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4708843529`。结果：average rollout loss `0.5625592470`；h50 `E_q=0.0852242063`、`E_v=0.3801910379`、`E_omega=0.3009120027`、`MSE_x=0.7798174486`；mean `E_q=0.0436615263`、`E_v=0.2157105523`、`E_omega=0.2425391548`；`MSE_1_to_F=0.3993963243`。相对上一轮 latentsefix e4，`MSE_1_to_F`、h50 q/v/omega、mean q 和 average rollout loss 全部小幅改善，是新的 GRUTCN 开发期 best；仍明显落后 strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。该 horizon/test 读取影响后续决策：只允许最后一次严格低 LR continuation，若 cont2 不继续提升聚合指标就转 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-13 23:02 CST）：当前 active evaluation 是 `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` e5 horizon/test MSE-schema eval p1。训练已自然跑满 `max_epochs=6`，`early_stopped=false`，`train_summary.json` 指向 e5 best checkpoint `checkpoints/model-epoch=05-best_valid_loss=0.47.pth`，`best_valid_loss=0.4708843529`。e0-e5 validation 连续刷新：`0.4710687995 -> 0.4710116088 -> 0.4709645212 -> 0.4709274471 -> 0.4709027410 -> 0.4708842933`；e5 分项为 `valid_p_loss_epoch=0.0432935506`、`valid_v_loss_epoch=0.1596616060`、`valid_q_loss_epoch=0.0388227813`、`valid_omega_loss_epoch=0.2291064113`、`valid_state_mse_epoch=0.2502119839`。训练 tmux 已退出，训练 GPU watch 已停止，artifacts 保留；训练阶段未读取本轮 horizon/test metric。已启动 eval tmux `eval_grutcn_rawtoktf_latentsefix_cont_H20_e5_mse_p1`，GPU watch `gpu_watch_eval_grutcn_rawtoktf_latentsefix_cont_H20`，eval log `logs/eval_horizontest_20260513_grutcn_rawtoktf_latentsefix_cont_H20_e5_b32_mse_p1.log`，eval batch `32`；启动检查：tmux/watch alive，log 到 `Seed set to 10`，GPU watch 约 `756/8188 MiB`、util `36%`。下一步读取 eval 完成后的 average rollout loss、h50/mean `E_q/E_v/E_omega`、h50 `MSE_x`、`MSE_1_to_F`，并判断是否继续 latentse/raw-token 或 pivot 到 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-13 22:20 CST）：当前 active training 仍是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。e4 继续刷新 validation best：e0 `valid_loss_epoch=0.4710687995` -> e1 `0.4710116088` -> e2 `0.4709645212` -> e3 `0.4709274471` -> e4 `0.4709027410`，`best_valid_loss=0.4709026217`；e4 分项为 `valid_p_loss_epoch=0.0432964787`、`valid_v_loss_epoch=0.1596727371`、`valid_q_loss_epoch=0.0388250090`、`valid_omega_loss_epoch=0.2291083932`、`valid_state_mse_epoch=0.2502122521`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20` 均 alive；GPU watch 约 `6569/8188 MiB`、util `31-33%`；无 Traceback/OOM/NaN；checkpoint `model-epoch=02/03/04-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成，`train_summary` 尚未生成，horizon/test metric 尚未读取。判定：validation 仍在缓慢下降，继续等待 natural finish；训练自然结束或 validation best 后按协议跑/read horizon/test MSE-schema。
- 最新接手状态（2026-05-13 21:40 CST）：当前 active training 仍是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。e3 继续刷新 validation best：e0 `valid_loss_epoch=0.4710687995` -> e1 `0.4710116088` -> e2 `0.4709645212` -> e3 `0.4709274471`，`best_valid_loss=0.4709275663`；e3 分项为 `valid_p_loss_epoch=0.0433006957`、`valid_v_loss_epoch=0.1596887559`、`valid_q_loss_epoch=0.0388276987`、`valid_omega_loss_epoch=0.2291103750`、`valid_state_mse_epoch=0.2502110302`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20` 均 alive；GPU watch 约 `6569/8188 MiB`、util `31-39%`；无 Traceback/OOM/NaN；checkpoint `model-epoch=01/02/03-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成，`train_summary` 尚未生成，horizon/test metric 尚未读取。判定：e0-e3 validation 连续健康刷新，继续等待 natural finish；训练自然结束或 validation best 后按协议跑/read horizon/test MSE-schema，验证 validation gain 是否转化为 `MSE_1_to_F` 或 h50 `E_q/E_v/E_omega` 改善。
- 最新接手状态（2026-05-13 21:00 CST）：当前 active training 仍是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。e2 继续刷新 validation best，并首次低于 `0.471`：e0 `valid_loss_epoch=0.4710687995` -> e1 `0.4710116088` -> e2 `0.4709645212`，`best_valid_loss=0.4709644318`；e2 分项为 `valid_p_loss_epoch=0.0433063544`、`valid_v_loss_epoch=0.1597131640`、`valid_q_loss_epoch=0.0388310663`、`valid_omega_loss_epoch=0.2291137427`、`valid_state_mse_epoch=0.2502114773`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20` 均 alive；GPU watch 约 `6570/8188 MiB`、util `29-40%`；无 Traceback/OOM/NaN；checkpoint `model-epoch=00/01/02-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成，`train_summary` 尚未生成，horizon/test metric 尚未读取。判定：validation 正信号明确，但最终仍以 horizon/test MSE-schema 是否改善 `MSE_1_to_F` 与 h50 `E_q/E_v/E_omega` 为准；继续等待 natural finish 后评估。
- 最新接手状态（2026-05-13 20:20 CST）：当前 active training 仍是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。e1 继续刷新 validation best：e0 `valid_loss_epoch=0.4710687995` -> e1 `0.4710116088`，`best_valid_loss=0.4710114896`；e1 分项为 `valid_p_loss_epoch=0.0433137342`、`valid_v_loss_epoch=0.1597451419`、`valid_q_loss_epoch=0.0388350338`、`valid_omega_loss_epoch=0.2291177213`、`valid_state_mse_epoch=0.2502109706`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20` 均 alive；GPU watch 约 `6576/8188 MiB`、util `30-38%`；无 Traceback/OOM/NaN；checkpoint `model-epoch=00/01-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成，`train_summary` 尚未生成，horizon/test metric 尚未读取。判定：e0/e1 连续健康刷新，继续观察 e2/natural finish；训练自然结束或 validation best 后按协议跑/read horizon/test MSE-schema，验证 validation gain 是否转化为 `MSE_1_to_F` 或 h50 `E_q/E_v/E_omega` 改善。
- 最新接手状态（2026-05-13 19:40 CST）：当前 active training 仍是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。e0 validation 已出并刷新当前 GRUTCN validation best：`valid_loss_epoch=0.4710687995`、`best_valid_loss=0.4710687697`、`valid_p_loss_epoch=0.0433222577`、`valid_v_loss_epoch=0.1597835869`、`valid_q_loss_epoch=0.0388404727`、`valid_omega_loss_epoch=0.2291224897`、`valid_state_mse_epoch=0.2502124310`。训练 tmux `modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1` 和 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20` 均 alive；GPU watch 约 `6568/8188 MiB`、util `32-39%`；无 Traceback/OOM/NaN；checkpoint `model-epoch=00-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成，`train_summary` 尚未生成，horizon/test metric 尚未读取。判定：e0 低于健康线 `0.47112`，并且 p/v/q/omega/state_mse 较上一轮 e4 同步小幅改善；继续观察 e1/e2/natural finish。训练自然结束或 validation best 后按协议跑/read horizon/test MSE-schema；若 continuation 不继续改善 `MSE_1_to_F` 或 h50 `E_q/E_v/E_omega`，停止同类 latentse/raw-token 微调并转 `multi-step delta_v/delta_omega predictor`。
- 最新接手状态（2026-05-13 18:42 CST）：当前 active training 是 continuation `modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_cont_H20`；训练日志：`logs/train_phase1.log`；GPU watch log：`logs/gpu_watch.log`；init checkpoint：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。配置：GRUTCN H20/F50 raw-token main + `latent_se`，physics OFF，epochs `6`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=2e-7`，`cosine_lr=7e-8`，`warmup_steps=50`，`cosine_steps=1200`，early stopping patience `2`，min_delta `1e-5`。trainable patterns 同上一轮，不训练 `raw_token_adaptive_*`。启动检查：训练 tmux/GPU watch alive，日志已加载 e4 checkpoint，trainable params 约 `3.4M`，GPU 约 `6543/8188 MiB`，无 OOM/NaN/Traceback。Gate：e0 `>0.47118` 或 q/v/omega/state_mse 同步回退则停止不 eval；e0 `0.47112-0.47118` 观察 e1/e2；e0 `<=0.47112` 健康。
- 刚完成 eval：`modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1` e4 horizon/test MSE-schema eval p1 已完成并读取 metric。checkpoint：`checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4711425900`，average rollout loss `0.5627378821`；h50 `E_q=0.0852424028`、`E_v=0.3803976436`、`E_omega=0.3009922140`、`MSE_x=0.7799340487`；mean `E_q=0.0436740862`、`E_v=0.2158192034`、`E_omega=0.2425650592`；`MSE_1_to_F=0.3994537865`。结论：相对 physkin e5 / rawtoktf cont e7 小幅全面改善，是当前 GRUTCN 开发期 best；仍明显落后 strongest GRU targets。该读取已用于决策：允许一次低 LR continuation；若 continuation horizon/test 不能继续改善，停止同类 latentse/raw-token 微调并转 `multi-step delta_v/delta_omega predictor`。

- 最新巡检异常（2026-05-13 14:48 CST）：heartbeat 尝试巡检当前 active training `modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1` 时，`gpu4060` 与 `gpu4060-ts` 到 Tailscale IP `100.106.154.6` 均 SSH timeout，`gpu4060-lan` 到 `192.168.1.108` 连接后关闭。本轮无法确认 tmux/进程/GPU/CSVLogger/checkpoint/train_summary；未停止任何实验、未清理 artifacts、未读取 horizon/test metric。处理建议：下一轮 heartbeat 继续先读 `MODEL_DEV_CURRENT.md`，再重试远程压缩巡检；若 SSH 恢复，优先检查是否已完成 e0/e1 和是否触发 gate。不要把本次 SSH 不可达直接判定为训练失败。

- 最新接手状态（2026-05-13 14:25 CST）：当前 active training 是 `modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_shufflefix_H20`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。
- 关键复盘：旧 `modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1` 的 e0 `valid_loss_epoch=0.5998380780` 不是模型 no-op 崩坏，而是 validation shuffle 协议 bug。`scripts/dynamics_learning/data.py` 原先对 training/validation/test 都用 `args.shuffle`；失败实验传了 `--shuffle true`，在 `limit_val_batches=0.5` 下实际评估随机半个验证集，不能和 physkin e5 的顺序半验证集 `0.4715962` 比。no-op 诊断：当前代码加载 physkin e5 checkpoint 不训练时，沿用 `args.shuffle=True` 得到 `valid_loss_epoch=0.6004404`；把 validation 固定顺序后，同一 checkpoint 精确回到 `0.4715962`。修复：`load_dataset()` 改为 `shuffle=args.shuffle and mode == "training"`，已同步远程；修复后即使训练命令继续 `--shuffle true`，validation loader 也显示 `SequentialSampler`。
- 当前候选结构/配置：GRUTCN H20/F50，raw-token main branch + `latent_se` channel recalibration，从 physkin e5 继续，physics loss OFF；trainable patterns 为 `latent_se,raw_token_pos,raw_token_input_norm,raw_token_proj,raw_token_encoder,raw_token_query,raw_token_score,raw_token_context_norm,raw_token_head,raw_token_velocity,raw_token_attitude`，明确不训练 `raw_token_adaptive_*`。epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=3e-7`，`cosine_lr=1e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `1e-5`，WANDB disabled。
- 启动检查：训练 tmux 和 GPU watch 均已启动；日志已加载 physkin e5 checkpoint，missing keys 仅未训练的 `raw_token_adaptive_*`；GPU watch 首条约 `1090/8187 MiB`，未见 Traceback/OOM/NaN。旧 `latentsefix_H20` p1 artifacts 保留，horizon/test metric 未读取；其结构失败结论作废为“validation shuffle 协议 bug”。
- Gate：参考 physkin e5 `best_valid_loss=0.4715962` 与 horizon/test `h50 E_q=0.0853132015, E_v=0.3808130120, E_omega=0.3010985338, MSE_1_to_F=0.3995425400`。e0 若 `valid_loss_epoch >0.47170` 或 q/v/omega/state_mse 同步明显回退则停止不 eval；e0 `0.47159-0.47170` 观察 e1/e2；e0 `<=0.47159` 健康。natural finish/validation best 后按协议跑/read horizon/test MSE-schema，并比较 physkin e5、rawtoktf cont e7 和 strongest GRU targets。

- 最新接手状态（2026-05-13 13:50 CST）：当前无 active training/evaluation。刚停止的实验是 `modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`；已停止训练 tmux：`modeldev_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`；已停止 GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_H20`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。
- 失败结果：GRUTCN H20/F50，raw-token main branch + `latent_se` channel recalibration，从 physkin e5 继续，physics loss OFF，trainable patterns 显式避开 `raw_token_adaptive_*`。CSVLogger e0：`valid_loss_epoch=0.5998380780`、`best_valid_loss=0.5998381972`，checkpoint `checkpoints/model-epoch=00-best_valid_loss=0.60.pth`，远高于 gate stop line `0.47170`。无 OOM/NaN/Traceback；停止前 GPU 约 `6553/8188 MiB`，停止后约 `474/8188 MiB`；artifacts 保留，未读取 horizon/test metric。
- 结论/下一步：H20 latentsefix 也崩到约 `0.600`，说明近期失败不只是 H30 长历史噪声；优先怀疑当前 `grutcn.py` 相对 physkin e5 checkpoint 的 no-op 兼容被破坏，或训练/加载协议与原 physkin 结果不一致。不要继续开相似 raw-token/latent_se/adapter 微调。下一步先做 checkpoint/code no-op 诊断：加载 physkin e5 checkpoint 不训练时 validation 是否仍接近 `0.4715962`；若 no-op validation 不接近，先修复兼容再谈新结构；若 no-op 正常，再复盘为什么极低 LR 一轮训练会把 validation 打到 0.60。
- 最新接手状态（2026-05-13 13:03 CST）：当前 active training 是 `modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_latentsefix_H20`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。
- 当前候选结构/配置：GRUTCN H20/F50，从 physkin e5 继续，physics loss OFF；本轮不是继续只微调 physics，而是在 H20 校准内训练 raw-token 主分支 + `latent_se` channel recalibration。epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=3e-7`，`cosine_lr=1e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `1e-5`，WANDB disabled。
- 重要启动纠偏：最初短暂启动过 `modeldev_20260513_grutcn_rawtoktf_latentse_H20_from_physkin_e5_p1`，使用 `--trainable_parameter_patterns raw_token,latent_se`，但这会把随机新增的 `raw_token_adaptive_*` 也纳入训练，和刚失败的 H30 adaptive 风险同源；该 p1 已立即停止，仅清理运行态、artifacts 保留、未读 horizon/test。当前 fix 版显式 trainable patterns 为 `latent_se,raw_token_pos,raw_token_input_norm,raw_token_proj,raw_token_encoder,raw_token_query,raw_token_score,raw_token_context_norm,raw_token_head,raw_token_velocity,raw_token_attitude`，启动日志确认 trainable list 不含 `raw_token_adaptive_*`；checkpoint missing keys 仍只有未训练的 `raw_token_adaptive_*`，这是预期兼容项。
- 当前候选 gate：参考 physkin e5 `best_valid_loss=0.4715962` 与 horizon/test `h50 E_q=0.0853132015, E_v=0.3808130120, E_omega=0.3010985338, MSE_1_to_F=0.3995425400`。e0 若 `valid_loss_epoch >0.47170` 或 q/v/omega/state_mse 同步明显回退则停止不 eval；e0 `0.47159-0.47170` 观察 e1/e2；e0 `<=0.47159` 健康。natural finish/validation best 后按协议跑/read horizon/test MSE-schema，并比较 physkin e5、rawtoktf cont e7 和 strongest GRU targets。
- 最新接手状态（2026-05-13 12:55 CST）：当前无 active training/evaluation。刚完成停止的实验是 `modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_adapthistfix_H30`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260512_grutcn_rawtoktf_cont_H20_from_rawtok_e4_p1/checkpoints/model-epoch=07-best_valid_loss=0.47.pth`。
- p2 失败结果：该实验为 H30 history-expanded adaptive raw-token fix，`anchor_history_len=20`，旧 anchor/raw-history/raw-token anchor 均看 recent H20，full H30 只进入 `raw_token_adaptive_*`，并用 `--trainable_parameter_patterns raw_token_adaptive` 只训练新长历史 adaptive 分支；physics loss OFF。checkpoint 加载日志显示 missing keys 仅 `raw_token_adaptive_*`，无 unexpected/shape-mismatch/OOM/NaN/Traceback；但 e0 `train_loss_epoch=0.359`、`valid_loss_epoch=0.604`、`best_valid_loss=0.604`，checkpoint `checkpoints/model-epoch=00-best_valid_loss=0.60.pth`，远高于 gate stop line `0.47190`。已只停止该实验训练 tmux、匹配该实验路径的训练进程和 GPU watch；停止前 GPU 约 `5717/8188 MiB`，停止后约 `474/8188 MiB`。artifacts 保留，未读取 horizon/test metric。
- p2 结论/下一步：H30 adaptive-only 路线在 p1/p2 都 e0 崩坏；p1 的旧 raw-history full-H30 问题已修，但 p2 仍说明“只训练新增长历史 adaptive 分支”会快速破坏 validation。下一候选不要继续同形态 H30 adaptive-only。更稳的方向是：先做 H20 强结构升级或 H25/H30 的更保守 long-history branch（长窗只读、强 gate/null context、极小 residual scale/LR、最好先加 pre-train validation sanity），再决定是否训练；也可重新审视 TCNLSTM true-anchor velocity partial win 的后续结构。
- 最新接手状态（2026-05-13 11:55 CST）：当前 active training 已切到修正版 history-expanded `modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_adapthistfix_H30`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthistfix_H30_from_cont_e7_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260512_grutcn_rawtoktf_cont_H20_from_rawtok_e4_p1/checkpoints/model-epoch=07-best_valid_loss=0.47.pth`。
- 刚失败的 H30 p1：`modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1` e0 `valid_loss_epoch=0.6041508` / `best_valid_loss=0.6041512`，远高于 stop line `0.47190`，未生成 `train_summary.json`，未读取 horizon/test metric；无 OOM/NaN/Traceback。已只停止该实验训练 tmux、匹配该实验 id 的训练进程和 GPU watch，artifacts 保留，GPU 回到约 `453/8188 MiB`。
- p1 失败复盘与代码修复：H30 设计原意是 calibrated anchor path 只看 recent H20，full H30 只进 zero-init adaptive branch；但旧 `raw_history_proj(x).mean(dim=1)` 仍然吃 full H30。该分支来自 H20 rawtoktf checkpoint 且不是新 zero-init 分支，输入分布改变会破坏迁移校准。已在 `scripts/dynamics_learning/models/grutcn.py` 改为 `raw_history_proj(x_anchor).mean(dim=1)`，使旧 raw-history residual 也只看 recent H20。local py_compile、`git diff --check`、remote py_compile 通过；local/remote `grutcn.py` hash：`631e12d6b96120477b9108cf59b9fc5ee506e2b41ba29d28f6794474f3f112ac`。
- p2 结构/协议：`anchor_history_len=20`；`encoder` / `base_decoder` / `state_initializer` / raw-history residual / raw-token anchor 都保持 recent H20 校准；full H30 仅通过 `raw_token_adaptive_*` multi-scale short/mid/full context 进入。为了进一步保护 H20 checkpoint，p2 使用 `--trainable_parameter_patterns raw_token_adaptive`，只训练新增长历史 adaptive 分支，冻结旧 `raw_token_*` 主分支；physics loss OFF。remote smoke `smoke_20260513_grutcn_adapthist_H30_anchorrawfix_p1` 已从 H20 e7 checkpoint 加载成功，missing keys 仅新增 `raw_token_adaptive_*`，无 unexpected/shape mismatch，一批 train/valid finite，未读取 horizon/test metric。
- p2 配置：GRUTCN H30/F50，epochs `6`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1500`，early stopping patience `2`，min_delta `1e-5`，WANDB disabled。
- p2 gate：参考 rawtoktf cont e7 `best_valid_loss=0.4716687` 与 physkin e5 `best_valid_loss=0.4715962`。e0 `>0.47190` 或 q/v/omega/state_mse 同步明显回退则停止不 eval；e0 `0.47165-0.47190` 观察 e1/e2；e0 `<=0.47165` 健康。natural finish/validation best 后按协议读取 horizon/test MSE-schema，并比较 rawtoktf cont e7、physkin e5、strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。

- 最新接手状态（2026-05-13 11:05 CST）：当前 active training 已切到 history-expanded `modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1`。远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1`；训练 tmux：`modeldev_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1`；GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_adapthist_H30`；训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1/logs/train_phase1.log`；GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1/logs/gpu_watch.log`；初始化 checkpoint：`resources/experiments/modeldev_20260512_grutcn_rawtoktf_cont_H20_from_rawtok_e4_p1/checkpoints/model-epoch=07-best_valid_loss=0.47.pth`。
- 当前 H30 结构/协议：`scripts/dynamics_learning/models/grutcn.py` 使用 `anchor_history_len=min(history_len,20)`，主 `encoder`/TCN anchor path 只看最近 H20；新增 `raw_token_adaptive_*` branch 读取 full H30 history，并通过 short/mid/full multi-scale context 自适应注入 raw-token context。`scripts/train.py` 已加入 checkpoint shape filter，旧 checkpoint 中同名但 shape 不匹配参数会被跳过，避免跨 H20/H30/H50 初始化崩溃；本实验 physics loss OFF，只训练 `raw_token*`，其中包含 `raw_token_adaptive_*`。
- 检查/同步：local py_compile 与 `git diff --check` 通过；remote py_compile 通过；local/remote hash 一致：`grutcn.py=4fb3632d0c1eb9a41541576b062eb4818ea86b06e3ddbddcd01dd28648bc41bd`、`train.py=364c26c723e1904d450895ea3e3cac3172a65fc19b73dafb246fdac3c8d95cdf`。一次 `rsync` 目标少 `--relative` 误把 `grutcn.py/train.py` 放到远程 repo 根目录，已只删除本次误放文件并重新同步正确路径。
- Smoke：H30 p2 从 rawtoktf cont e7 H20 checkpoint 加载成功，missing keys 仅新增 `raw_token_adaptive_*`，无 unexpected/shape-mismatch crash；trainable parameter list 包含 `raw_token_adaptive_*`。H20 control smoke 同配置也通过，因此一批 loss 波动不作为 H30 失败依据。两个 smoke 均未读取 horizon/test metric。
- 当前训练配置：GRUTCN H30/F50，epochs `6`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1500`，early stopping patience `2`，min_delta `1e-5`，WANDB disabled，`--trainable_parameter_patterns raw_token`。
- 启动巡检：训练 tmux/process/GPU watch healthy，GPU 约 `7527/8188 MiB`、util `40%`，train steps finite，trainable params `4.1M`，missing keys 仅新增 adaptive 分支；horizon/test metric 未读取。显存余量较小，后续巡检重点看 OOM/NaN/Traceback；若 OOM，只停止本实验训练 tmux、匹配进程和 GPU watch，artifacts 默认保留。
- H30 gate：参考 rawtoktf cont e7 `best_valid_loss=0.4716687` 与 physkin e5 `best_valid_loss=0.4715962`。e0 `>0.47190` 或 q/v/omega/state_mse 同步明显回退则停止不 eval；e0 `0.47165-0.47190` 观察 e1/e2；e0 `<=0.47165` 健康。natural finish/validation best 后按协议读取 horizon/test MSE-schema，并比较 rawtoktf cont e7、physkin e5、strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。
- 最新完成训练：`modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1`，用于测试 rawtoktf e7 基础上加入小权重 physics-informed regularization + slack/reliability gate 是否能把长时域小增益放大。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1`。
- 训练 tmux：`modeldev_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1`。
- GPU watch tmux：`modeldev_gpu_watch_grutcn_rawtoktf_physkin_H20`。
- 训练日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260512_grutcn_rawtoktf_cont_H20_from_rawtok_e4_p1/checkpoints/model-epoch=07-best_valid_loss=0.47.pth`。
- 配置：GRUTCN H20/F50，epochs `6`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=5e-7`，`cosine_lr=1.5e-7`，`warmup_steps=50`，`cosine_steps=1200`，early stopping patience `2`，min_delta `1e-5`，WANDB disabled，`--trainable_parameter_patterns raw_token`。
- 结构/协议：沿用 checkpoint-safe `raw_token_*` raw-history token Transformer side branch，只训练 `raw_token*`；不替换 `encoder`、`base_decoder`、`state_initializer` 或 GRU memory，旧 observer/latent SE/motion 分支保留 key 兼容。本候选额外开启 train-only physics regularization：以 `p_{t+1}` 与 `p_t + 0.5*dt*(v_t+v_{t+1})` 的局部运动学一致性为主，并用 target 轨迹自身运动学残差作为 slack/reliability。当前权重：`physics_loss_weight=1000.0`、`physics_kinematic_weight=1.0`、`physics_quat_norm_weight=0.01`、`physics_v_smooth_weight=0.0`、`physics_omega_smooth_weight=0.0`、`physics_reliability_scale=10.0`、`physics_slack_margin=0.0`。没有上传或泄漏私有数据，只用聚合指标和误差趋势做决策。
- 检查/同步：local `python3 -m py_compile scripts/dynamics_learning/models/grutcn.py` 和 `git diff --check -- scripts/dynamics_learning/models/grutcn.py` 通过；local/remote `grutcn.py` hash 均为 `4e5fe4251d34dca3390e8c005cf08022cf09168dfd94d9236089403dda6e5248`；remote py_compile 通过。remote smoke `smoke_20260512_grutcn_rawtoktf_H20_from_joint_e4_p1` 从 jointobserver e4 加载成功，missing keys 为新增 `raw_token_*` 与 zero-init `latent_se_*` 兼容分支，无 unexpected keys；trainable parameters 仅 `raw_token*`；一批 train/valid finite：`train_loss_step=0.146`、`valid_loss_epoch=0.0819`；checkpoint、CSVLogger、`train_summary.json` 正常；smoke 未读取 horizon/test metric，只确认 `plots/testset` 存在。
- 启动巡检：训练 tmux/process/GPU watch healthy，GPU 约 `4702/8188 MiB`、util `30%`，训练 steps finite；horizon/test metric 尚未读取。Gate：参考 rawtoktf cont e7 `best_valid_loss=0.4716687` 与 e7 horizon/test 小幅全面 best；e0 `>0.47180` 或 physics objective 明显破坏 validation 即停止不 eval；e0 `0.47167-0.47180` 观察 e1/e2；e0 `<=0.47166` 健康。natural finish 或 validation best 后按协议读取 horizon/test，重点看 h50 `E_q/E_v/E_omega`、mean `E_q`、`MSE_1_to_F` 是否超过 e7。
- 最新巡检（2026-05-12 20:37 CST）：训练 tmux `modeldev_grutcn_rawtoktf_H20_from_joint_e4_p1` 与 GPU watch tmux `modeldev_gpu_watch_grutcn_rawtoktf_H20` 仍在，训练进程存在；GPU watch 约 `4720/8188 MiB`、util `30-41%`。CSVLogger e0/e1：`valid_loss_epoch=0.4719697 -> 0.4719442`，`best_valid_loss=0.4719442`；e1 submetrics 为 `valid_q=0.0389052`、`valid_p=0.0434147`、`valid_v=0.1604052`、`valid_omega=0.2292190`、`valid_state_mse=0.2503258`，相对 e0 同步小幅改善。checkpoint e0/e1 与 `last_model.pth` 已产生；`train_summary.json`、`horizon_summary.json`、`horizon_metrics.csv` 尚未生成；未读取 horizon/test metric 内容。判定：已经越过优先 eval/freeze 线 `best_valid_loss <=0.47195`，但训练还在运行，下一步等待 natural finish/summary 指向 best checkpoint 后启动 horizon/test MSE-schema evaluation，不手动抢跑。
- 最新状态（2026-05-12 22:22 CST）：`modeldev_20260512_grutcn_rawtoktf_H20_from_joint_e4_p1` 已自然跑满 `max_epochs=5`，`early_stopped=false`，`train_summary.json` 指向 e4 checkpoint `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。validation 曲线 e0/e1/e2/e3/e4 为 `0.4719697 -> 0.4719442 -> 0.4719169 -> 0.4718953 -> 0.4718790`，e4 submetrics `valid_q=0.0389019`、`valid_p=0.0434097`、`valid_v=0.1603578`、`valid_omega=0.2292095`、`valid_state_mse=0.2503134`，同步小幅改善，显著优于 jointobserver validation `0.4719877`。训练阶段未读取 horizon/test metric；artifacts 保留。已停止训练 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_H20`，启动 horizon/test MSE-schema eval：tmux `eval_grutcn_rawtoktf_H20_e4_mse_p1`，GPU watch `gpu_watch_eval_grutcn_rawtoktf_H20`，eval log `resources/experiments/modeldev_20260512_grutcn_rawtoktf_H20_from_joint_e4_p1/logs/eval_horizontest_20260512_grutcn_rawtoktf_H20_e4_b32_mse_p1.log`，eval GPU watch log `logs/gpu_watch_eval.log`。启动检查：eval process healthy，GPU 约 `795/8188 MiB`、util `35%`，eval log 已写 `Seed set to 10`；`horizon_summary.json` / `horizon_metrics.csv` 尚未生成，尚未读取 horizon/test metric。下一步：巡检 eval 完成后读取 h50/mean `E_q/E_v/E_omega` 与 `MSE_1_to_F`，记录是否影响后续结构决策，并和 jointobserver p2、strongest GRU targets 比较。
- 最新结果（2026-05-12 22:51 CST）：`modeldev_20260512_grutcn_rawtoktf_H20_from_joint_e4_p1` e4 horizon/test MSE-schema eval 已完成并读取 metric。checkpoint：`checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4718789756`，eval log `logs/eval_horizontest_20260512_grutcn_rawtoktf_H20_e4_b32_mse_p1.log`，average rollout loss `0.5631443858`；h50 `E_q=0.0853260663`、`E_v=0.3809074430`、`E_omega=0.3011899105`、`MSE_x=0.7803351939`；mean `E_q=0.0437284932`、`E_v=0.2160562063`、`E_omega=0.2426384474`；`MSE_1_to_F=0.3996152170`。相对 jointobserver p2（h50 `E_q=0.0853401803`、`E_v=0.3809713071`、`E_omega=0.3012216526`、mean `E_q=0.0437370777`、`MSE_1_to_F=0.3996392623`）小幅全面改善；但仍落后 strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。该 horizon/test 读取已影响后续结构决策：继续保留 rawtoktf 方向，优先从 e4 checkpoint 做 continuation/更长训练，必要时叠加小权重 physics-informed regularization + slack/reliability gate；不要回退到 latent SE 或单纯 output observer。artifacts 保留，eval tmux/GPU watch 已自然结束，无清理。
- 当前 continuation 启动状态（2026-05-12 22:56 CST）：tmux/process/GPU watch healthy，GPU 约 `4719/8188 MiB`、util `33%`，checkpoint 已加载，trainable parameters 仅 `raw_token*`，epoch 0 train steps finite；未读取新的 horizon/test metric。总控 automation 已更新为每 `40` 分钟巡检该实验。Gate：continuation e0 若 `valid_loss >0.47195` 或 valid q/v/omega/state_mse 同步回退则停止不 eval；e0 `<=0.47188` 或后续继续刷新则健康；natural finish/validation best 后按协议读取 horizon/test，并与 rawtoktf e4 与 strongest GRU targets 比较。
- 最新 continuation 巡检（2026-05-12 23:40 CST）：e0 validation 刷新 rawtoktf best，`valid_loss_epoch=0.4718495`、`best_valid_loss=0.4718494`，checkpoint `checkpoints/model-epoch=00-best_valid_loss=0.47.pth` 与 `last_model.pth` 已生成；训练已进入 e1，GPU watch healthy，GPU 约 `4698/8188 MiB`、util `30%`，无 OOM/NaN/Traceback。判定：e0 低于健康线 `<=0.47188`，支持 rawtoktf e4 欠训练假设；继续等待后续 epoch/natural finish。训练阶段未读取新的 horizon/test metric。
- 最新 continuation 巡检（2026-05-13 00:20 CST）：e1 validation 继续刷新，`valid_loss_epoch=0.4718124`、`best_valid_loss=0.4718124`，checkpoint `checkpoints/model-epoch=01-best_valid_loss=0.47.pth` 与 `last_model.pth` 已更新；训练进入 e2，GPU watch healthy，GPU 约 `4702/8188 MiB`、util `34%`，无 OOM/NaN/Traceback。判定：continuation 明确有效，继续等 natural finish/summary，再按协议跑 horizon/test MSE-schema eval；训练阶段未读取新的 horizon/test metric。
- 最新状态（2026-05-13 04:21 CST）：continuation 自然跑满 `max_epochs=8`，`early_stopped=false`，`train_summary.json` 指向 e7 checkpoint `checkpoints/model-epoch=07-best_valid_loss=0.47.pth`。e0-e7 validation 为 `0.4718495 -> 0.4718124 -> 0.4717747 -> 0.4717430 -> 0.4717138 -> 0.4716942 -> 0.4716801 -> 0.4716686`，`best_valid_loss=0.4716687`，比 rawtoktf e4 `0.4718790` 明显继续改善；训练阶段未读取新的 horizon/test metric，artifacts 保留，无 OOM/NaN/Traceback。已停止训练 GPU watch，启动 eval tmux `eval_grutcn_rawtoktf_cont_H20_e7_mse_p1` 与 eval GPU watch `gpu_watch_eval_grutcn_rawtoktf_cont_H20`；启动检查显示 eval process healthy，GPU 约 `529/8188 MiB`、util `2%`，`horizon_summary.json` / `horizon_metrics.csv` 尚未生成，尚未读取本次 horizon/test metric。
- 最新结果（2026-05-13 04:59 CST）：e7 horizon/test MSE-schema eval 完成并读取 metric。checkpoint：`checkpoints/model-epoch=07-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4716687`，eval log `logs/eval_horizontest_20260513_grutcn_rawtoktf_cont_H20_e7_b32_mse_p1.log`，average rollout loss `0.5630676746`；h50 `E_q=0.0853119630`、`E_v=0.3808232145`、`E_omega=0.3011252939`、`MSE_x=0.7802118972`；mean `E_q=0.0437197493`、`E_v=0.2160187981`、`E_omega=0.2426086313`；`MSE_1_to_F=0.3995643737`。相对 rawtoktf e4（h50 `E_q=0.0853260663`、`E_v=0.3809074430`、`E_omega=0.3011899105`、mean `E_q=0.0437284932`、`MSE_1_to_F=0.3996152170`）小幅全面改善；但仍明显落后 strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。该 horizon/test 读取已影响后续结构决策：rawtoktf 长训练方向有效但斜率很小，不能只靠继续训练追目标；下一步优先在 rawtoktf 基础上加入小权重 physics-informed regularization + slack/reliability gate，或升级更强 raw-token/长历史结构。Eval GPU watch 已停止，artifacts 保留，无清理。
- 代码/协议更新（2026-05-13 05:05 CST）：`scripts/config.py` 新增 physics regularization 参数；`scripts/dynamics_learning/lighting.py` 新增默认关闭的 train-only `physics_regularization`，以 `p_{t+1}` 与 `p_t + 0.5*dt*(v_t+v_{t+1})` 的局部运动学一致性为主，并用 target 轨迹自身运动学残差作为 slack/reliability，简单物理不可靠时自动减弱惩罚；`scripts/train.py` 把 physics 配置写入 `train_summary.json`。local `py_compile`、`git diff --check`、remote py_compile 均通过；local/remote hash：`config.py=63262b8cf5b79640b8ad8b7543acebf06e6d3215ea0e62f50eb031cb30834541`、`lighting.py=af55b5105dc5100e2e8e3fb5feea2526c0b868c38e37ddd2c022d82334fb9445`、`train.py=98760f8ac90446e8c1dd4f0476a9504963f10df0caebf9e554e29bb5636db655`。smoke `smoke_20260513_grutcn_rawtoktf_phys_H20_from_cont_e7_p1` 从 e7 checkpoint 加载成功，只训练 `raw_token*`，一批 train/valid finite：`train_loss_epoch=0.1463`、`train_objective_epoch=0.1463`、`train_physics_loss_epoch=1.0259e-06`、`train_physics_reliability_epoch=0.9910`、`valid_loss_epoch=0.0818`；未读取 horizon/test metric。
- 当前 active training 启动状态（2026-05-13 05:06 CST）：`modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1` 从 e7 checkpoint 初始化，GRUTCN H20/F50，epochs `6`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`、`limit_val_batches=0.5`，`warmup_lr=5e-7`、`cosine_lr=1.5e-7`，`warmup_steps=50`、`cosine_steps=1200`，early stopping patience `2`、min_delta `1e-5`，WANDB disabled，`--trainable_parameter_patterns raw_token`。Physics：`physics_loss_weight=1000.0`、`physics_kinematic_weight=1.0`、`physics_quat_norm_weight=0.01`、`physics_v_smooth_weight=0.0`、`physics_omega_smooth_weight=0.0`、`physics_reliability_scale=10.0`、`physics_slack_margin=0.0`。启动检查：tmux/process/GPU watch healthy，GPU 约 `4702/8188 MiB`、util `30%`，train steps finite；horizon/test metric 未读取。Gate：参考 e7 `best_valid_loss=0.4716687` 与 e7 horizon/test 小幅全面 best；e0 `>0.47180` 或 physics objective 明显破坏 validation 即停止不 eval；e0 `0.47167-0.47180` 观察 e1/e2；e0 `<=0.47166` 健康。natural finish 或 validation best 后按协议读取 horizon/test，重点看 h50 `E_q/E_v/E_omega`、mean `E_q`、`MSE_1_to_F` 是否超过 e7。
- 最新状态（2026-05-13 09:00 CST）：`modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1` 自然跑满 `max_epochs=6`，`early_stopped=false`，`train_summary.json` 指向 e5 checkpoint `checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。e0-e5 validation 连续刷新 rawtoktf best，`0.4716556 -> 0.4716390 -> 0.4716238 -> 0.4716125 -> 0.4716022 -> 0.4715962`，`best_valid_loss=0.4715962`；e5 train objective `0.2396219`，train physics loss `7.024e-06`，physics reliability `0.9758`。训练阶段 horizon/test metric 未读取，artifacts 保留，无 OOM/NaN/Traceback；训练 GPU watch `modeldev_gpu_watch_grutcn_rawtoktf_physkin_H20` 已停止。已启动 horizon/test MSE-schema eval：tmux `eval_grutcn_rawtoktf_physkin_H20_e5_mse_p1`，eval GPU watch `gpu_watch_eval_grutcn_rawtoktf_physkin_H20`，eval log `resources/experiments/modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1/logs/eval_horizontest_20260513_grutcn_rawtoktf_physkin_H20_e5_b32_mse_p1.log`；启动检查 healthy，日志写入 `Seed set to 10`，`horizon_summary.json` / `horizon_metrics.csv` 尚未生成，尚未读取本次 horizon/test metric。下一步：等 eval 完成后读取 h50/mean `E_q/E_v/E_omega` 与 `MSE_1_to_F`，记录是否影响后续结构决策，并和 rawtoktf cont e7、strongest GRU targets 比较。
- 最新结果（2026-05-13 09:38 CST）：e5 horizon/test MSE-schema eval 完成并读取 metric。checkpoint：`checkpoints/model-epoch=05-best_valid_loss=0.47.pth`，training `best_valid_loss=0.4715962`，eval log `logs/eval_horizontest_20260513_grutcn_rawtoktf_physkin_H20_e5_b32_mse_p1.log`，average rollout loss `0.5630497932`；h50 `E_q=0.0853132015`、`E_v=0.3808130120`、`E_omega=0.3010985338`、`MSE_x=0.7801643574`；mean `E_q=0.0437205195`、`E_v=0.2160128946`、`E_omega=0.2425938835`；`MSE_1_to_F=0.3995425400`。相对 rawtoktf cont e7（h50 `E_q=0.0853119630`、`E_v=0.3808232145`、`E_omega=0.3011252939`、mean `E_q=0.0437197493`、`MSE_1_to_F=0.3995643737`）仅 `h50_v`、`h50_omega`、`MSE_1_to_F` 极小改善，`h50_q` 与 mean `E_q` 极小回退；仍明显落后 strongest GRU targets（h50_q `0.0800042`、h50_v `0.353015`、h50_omega `0.260392`、mean_q `0.0420377`）。结论：physics kinematic regularization 可降 validation 与 full-state MSE，但未突破最强 long-horizon baseline；该 horizon/test 读取已影响后续结构决策，下一步不能只继续当前小权重物理 loss，应转向更强 raw-token/长历史结构或重新审视 TCNLSTM true-anchor velocity partial win。eval GPU watch `gpu_watch_eval_grutcn_rawtoktf_physkin_H20` 已停止，artifacts 保留，无清理。
- 待尝试协议候选：用户建议加入 physics loss，可参考 Saviolo/Li/Loianno 2022 `Physics-Inspired Temporal Learning of Quadrotor Dynamics for Accurate Model Predictive Trajectory Tracking` 和 Serrano et al. 2024 `Physics-Informed Neural Network for Multirotor Slung Load Systems Modeling`。核心思路：PI-TCN 把物理约束嵌入 temporal learning 训练以增强泛化；slung-load PINN 用一阶原理离散物理模型构造 physics-based loss，并加入 slack variables 允许预测与物理模型存在小偏差。项目落地建议：不要在当前 rawtoktf e4 eval 中改动；若 rawtoktf horizon/test 有正信号，优先从 e4 checkpoint 做 continuation，加入默认关闭、train-only、小权重的 physics regularization；使用可学习/可调 reliability gate 或 slack head，让模型在物理先验不可靠时自动降低约束权重。可先约束最基础的局部一致性：`p_{t+1}≈p_t+dt*v_t`、速度/角速度平滑、控制输入与 acceleration 的一致性、quaternion norm/小角速度一致性。所有新增 loss 必须记录文件、协议、权重、是否读取 horizon/test、观察结果；不得把数据样本/标签/轨迹硬编码或上传到外部。
- 上一完成实验：`modeldev_20260512_grutcn_latentse_H20_from_anchor_e4_p1` e2 horizon/test MSE-schema eval 已完成，结果不支持继续小调 latent SE。
- 最新状态（2026-05-12 19:12 CST）：训练 early stop 后的 e2 checkpoint 已完成 horizon/test MSE-schema eval，并读取 metric 内容。训练 e0/e1/e2 `valid_loss_epoch=0.4721178 -> 0.4721160 -> 0.4721141`，`best_valid_loss=0.4721142`，best checkpoint `checkpoints/model-epoch=02-best_valid_loss=0.47.pth`；eval average rollout loss `0.5632528663`，h50 `E_q=0.0853664371`、`E_v=0.3810531020`、`E_omega=0.3012494289`、`MSE_x=0.7804412069`，mean `E_q=0.0437534364`、`E_v=0.2161137365`、`E_omega=0.2426598685`，`MSE_1_to_F=0.3996510302`。判定：相对 jointobserver p2（h50 `E_q=0.0853401803`、`E_v=0.3809713071`、`E_omega=0.3012216526`、`MSE_1_to_F=0.3996392623`）更差，也没有接近 strongest GRU targets；latent SE 小 adapter 没形成有效长时域增益，不建议继续调 scale/lr。artifacts 保留，eval GPU watch 已停止，当前无 active training/eval。
- 下一步建议：转向更大表达力但 checkpoint-safe 的 raw-history token Transformer side branch，或按 `AGENTS.md` web research 引入更强历史记忆/频域/状态空间模块；原则是 zero-init 注入 observer/head，不替换 `encoder`、`base_decoder`、`state_initializer` 或 GRU memory，不上传/泄漏私有数据，只用聚合指标和误差趋势做结构决策。
- remote smoke：`smoke_20260512_grutcn_jointobserver_H20_from_anchor_e4_p1` 从 anchor e4 加载成功；missing keys 为新增/兼容 observer 分支，无 unexpected keys；trainable parameters 仅 `velocity_observer*` 与 `attitude_observer*`；一批 train/valid finite：`train_loss_step=0.138`、`valid_loss_epoch=0.0808`；只确认 `plots/testset` 存在，未读取 horizon/test metric。
- 已完成训练：2026-05-12 14:28 CST 左右自然跑满 `max_epochs=5`，`early_stopped=false`；e0/e1/e2/e3/e4 `valid_loss_epoch=0.4720957 -> 0.4720626 -> 0.4720305 -> 0.4720052 -> 0.4719877`，`best_valid_loss=0.4719877`，`train_summary.json` 指向 `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。训练阶段未读取 horizon/test metric，artifacts 保留，训练 GPU watch `modeldev_gpu_watch_grutcn_jointobserver_H20` 已停止。
- 指标代码状态：2026-05-12 14:52 CST 按用户要求新增截图中的 multi-step full-state MSE 指标。`scripts/eval.py` 现在输出每个 horizon 的 `MSE_x = mean_i ||x_{i,h}-xhat_{i,h}||_2^2`，并在 `horizon_summary.json` 里写 `MSE_1_to_F = mean_h MSE_x(h)`；`scripts/dynamics_learning/lighting.py` 额外记录 `train_state_mse`、`valid_state_mse`、`test_state_mse`，不改变训练目标和 `best_valid_loss`。local/remote py_compile 通过，hash 一致：`eval.py=473cc1104ec72a237e5f8ddccdf35a9364d2337cdccb5b88541a2d5a733e636e`，`lighting.py=411a2105b488df26baa2aebc4fa3cf92337fc39b13353f67717df4a4517bb21b`。
- 当前 eval 状态：首次 eval 在 MSE 代码同步前已完成，old-schema log 里读取到 `Average rollout loss=0.5631942749`，但 `horizon_metrics.csv` header 缺少 `MSE_x`；该读取只用于确认需要重跑新 schema，不用于结构结论。第一次 MSE-schema rerun `eval_grutcn_jointobserver_H20_e4_mse_p1` 暴露 wiring bug：`compute_horizon_metrics` 已产生 `MSE_x`，但 `run_prediction` 汇总列表仍是旧四指标，导致 `KeyError: 'MSE_x'`。已修复 `scripts/eval.py`：统一 `HORIZON_METRICS = ["E_p", "E_v", "E_q", "E_omega", "MSE_x"]` 供 summary、CSV、aggregation 使用；remote py_compile 通过，hash `eval.py=69631944db1592c51934ff0a0d58079f319dcfdbd7c8b128335aaaa89872e657`。MSE-schema rerun p2 已完成并读取 metric：checkpoint `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，average rollout loss `0.5631942749`，h50 `E_q=0.0853401803`、`E_v=0.3809713071`、`E_omega=0.3012216526`、`MSE_x=0.7803988811`，mean `E_q=0.0437370777`、`E_v=0.2160828500`、`E_omega=0.2426515404`，`MSE_1_to_F=0.3996392623`。与 attobserver e4 相比只在 `h50_v` / `h50_omega` 有约 `2e-5` / `8e-7` 量级小改善，`h50_q` 与 mean `E_q` 小幅回退；仍明显落后 strongest GRU targets。该 horizon/test 读取已影响后续结构决策：停止继续堆 output observer，下一候选转向 latent SE residual。
- 下一阶段结构候选（不要忘）：用户提出 SENet/SE gate 与 Transformer 输入分支。总控判断为可行，但要 checkpoint-safe。若 jointobserver horizon/test 仍只是小幅提升，下一候选优先做 latent SENet/SE residual，接在 TCN encoder 输出 `enc_seq` 后，形式 `enc_seq = enc_seq + scale * enc_seq * zero_init_se_delta`，不直接放在原始 `x[:, :, feature]` 前，避免缩放 quaternion、omega、control 原始物理量。若 SE 有信号，再尝试 raw-history token Transformer side branch：原始历史 `x` 投到 latent tokens，经 1-2 层 lightweight self-attention 得到长时域 context，再 zero-init 注入 observer/head；不要替换 `encoder`、`base_decoder`、`state_initializer` 或 GRU memory。优先级：latent SE 是下一步小实验，Transformer side context 是后续结构升级；避免继续无结构故事地堆 output observer。
- 最新 horizon/test 结果：2026-05-12 11:14 CST `modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1` e4 evaluation 完成并读取 metric。checkpoint：`checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，eval log `logs/eval_horizontest_20260512_grutcn_attobserver_H20_e4_b32.log`，average rollout loss `0.5632079840`；h50 `E_q=0.0853396507`、`E_v=0.3809912724`、`E_omega=0.3012224156`；mean `E_q=0.0437367480`、`E_v=0.2160949155`、`E_omega=0.2426517642`。相对 GRUTCN anchor e4 和 velocity_observer e4 均为小幅正增益，但离 strongest GRU targets 仍很远；该读取已用于结构决策。结论：frozen output observer 方向有信号，单一姿态分支表达力不足；下一步无代码改动地从 GRUTCN anchor e4 初始化、联合训练 `attitude_observer,velocity_observer`，测试两个 zero-init output observer 的叠加效果。Eval GPU watch 已停止，artifacts 保留。
- Purpose：新协议允许读取 horizon/test；本次评估用于判断 GRUTCN frozen-anchor attitude/omega observer 是否虽然 validation 只小幅改善，但在 h50 `E_q` / `E_omega` 或 mean q 上提供有用长时域改善，同时检查 h50 `E_v` 是否明显退化。
- Startup check：eval tmux/process/GPU watch healthy，GPU 约 `729/8188 MiB`、util `34%`；`horizon_summary.json` / `horizon_metrics.csv` 尚未生成，尚未读取 metric 内容。完成后必须记录 checkpoint、average rollout loss、h50/mean 关键指标、是否影响后续结构决策，以及结论。
- 刚完成训练：`modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1` 自然跑满 `max_epochs=5`，`early_stopped=false`；e4 best `best_valid_loss=0.4720434`，`train_summary.json` 指向 `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。CSVLogger 本实验未写出 `valid_q/valid_p/valid_v/valid_omega` 分列，只能确认 total validation 与 train loss。训练阶段未读取 horizon/test metric；artifacts 保留；训练 GPU watch `modeldev_gpu_watch_grutcn_attobserver_H20` 已停止。
- 当前 active training：`modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1`，2026-05-11 22:35 CST 已在远程 tmux 启动。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1`。
- 训练 tmux：`modeldev_grutcn_attobserver_H20_from_anchor_e4_p1`。
- GPU watch tmux：`modeldev_gpu_watch_grutcn_attobserver_H20`。
- 训练日志：`resources/experiments/modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。
- 配置：GRUTCN H20/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled，`--trainable_parameter_patterns attitude_observer`。
- 当前 GRUTCN 代码/结构：`scripts/dynamics_learning/models/grutcn.py` 在 GRUTCN H20 anchor e4 上新增 checkpoint-safe `attitude_observer_*`，输入 `base_feature + context + history_context + projected_x_last + decoder_feature + dx_last + base_delta[:, 6:12]`，final projection zero-init，`attitude_observer_scale=0.003`，只在最终输出后写 `y[:, 6:12]`；不写 `y[:, 0:6]`，不改 `encoder`、`base_decoder`、`state_initializer` 或 GRU memory。失败的 motion/velocity observer 分支保留 key 兼容；本候选冻结 anchor，只训练 `attitude_observer*`。
- 检查/同步：local py_compile、`git diff --check` 和 remote py_compile 通过；local/remote code hash 一致：`grutcn.py=9328db7e9e1c6349155bd56d982c23e5576d522c0dda44ab579319d4e91902c9`、`eval.py=aa8971b415136a53725df7cd19ee16e7daa6c3485f79dcd3870abf8656c17007`。一次错误多文件 `rsync` 把 `grutcn.py/eval.py` 放到远程 repo 根目录，已删除这两个本次误放文件并重同步到正确路径；`Prompt.md` / `MODEL_DEV_HANDOFF.md` 后续按本记录同步。
- remote smoke：`smoke_20260511_grutcn_attobserver_H20_from_anchor_e4_p1` 从 GRUTCN anchor e4 checkpoint 加载成功；missing keys 为新增/保留兼容分支（含 `attitude_observer_*`），无 unexpected keys；只训练 `attitude_observer*`；一批 train/valid finite：`train_loss_step=0.138`、`valid_loss_epoch=0.0817`；checkpoint、CSVLogger、`train_summary.json` 正常；只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- 启动巡检：训练 tmux、GPU watch tmux、训练进程存在，GPU 约 `2509/8188 MiB`、util `37%`，epoch 0 train steps finite；`plots/testset` 目录存在但未读取 horizon/test metric。Gate：e0 `>0.4728` 停止；e0 `0.47210-0.47280` 观察 e1/e2；e0 `<=0.47205` 健康。训练自然结束或 validation best 后按新协议读取 horizon/test metric，重点看 h50 `E_q`、`E_omega` 是否改善，同时检查 h50 `E_v` 不明显退化；每次读取必须写回 `Prompt.md` 和本交接文档。
- 最新异常：2026-05-11 22:24 CST 尝试评估旧候选 `modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1` e2 checkpoint，但启动后中止，未读取 horizon/test metric。checkpoint 为 `checkpoints/model-epoch=02-best_valid_loss=0.46.pth`，log 为 `logs/eval_horizontest_20260511_tcnlstm_lagobserver_H10_e2_b32.log`。原因：当前 `tcnlstm.py` active forward 是 velocity residual，不是 lagobserver；旧 checkpoint 在 strict load 下缺少后续新增模块 key，即使用 non-strict 也会在错误 forward 下评估，结果不能代表 lagobserver。处理：已停止该 eval tmux 和 GPU watch，artifacts 保留。继承建议：旧候选 horizon/test 只能在恢复匹配代码叙事/forward 后评估，不要直接用当前 forward 读取旧 lagobserver 指标做调参。
- 最近完成 horizon/test evaluation：`modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1` e2 checkpoint，2026-05-11 22:19 CST 完成并读取 metric。average rollout loss `0.5291858315`；h50 `E_q=0.0908154197`、`E_v=0.3454937450`、`E_omega=0.3339743749`；mean `E_q=0.0459003559`、`E_v=0.1858658490`、`E_omega=0.2568162155`。相对 TCNLSTM attitude H10 e3 locked audit 没有改善，velocity/q/omega 均略差；该结果已用于决策，关闭 velocity-only residual 分支，不继续细调。Eval GPU watch 已停止，artifacts 保留。
- 最近 horizon/test evaluation 启动记录（已完成，见上一条）：`modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1` e2 checkpoint，2026-05-11 22:04 CST 在远程 tmux 启动。
- Eval tmux：`eval_tcnlstm_velres_H10_e2_p1`。
- GPU watch tmux：`gpu_watch_eval_tcnlstm_velres_H10`。
- Eval log：`resources/experiments/modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1/logs/eval_horizontest_20260511_tcnlstm_velres_H10_e2_b32_retry2.log`。
- Eval command：`scripts/eval.py --dataset neurobemfullstate --predictor_type full_state --accelerator cuda --gpu_id 0 --eval_batch_size 32 --eval_horizons 1,10,25,50 --wandb_mode disabled --experiment_path resources/experiments/modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1`。
- Purpose：新协议允许读取 horizon/test；本次评估用于验证 TCNLSTM velocity-only 分支是否虽然 validation/q/omega gate 失败，但在 h50 velocity 或其他 long-horizon 指标上提供有用信号。
- Startup check：eval log 确认加载 `checkpoints/model-epoch=02-best_valid_loss=0.46.pth`，test windows `50904`，H10/F50，GPU 正在计算；截至记录时 `horizon_summary.json` / `horizon_metrics.csv` 尚未生成。完成后必须记录关键 horizon/test 指标、是否影响后续调参/结构决策，以及结论。
- Eval wiring fix：2026-05-11 修复 `scripts/eval.py` 的 checkpoint 选择 bug。旧正则把 `best_valid_loss=0.46.pth` 解析为 `0.46.` 并导致 `ValueError`；新正则只捕获数字小数，且 rounded loss 相同、无 `train_summary.json` 时用 checkpoint mtime 选择较新的 non-last checkpoint，以便手动 gate 停止实验可评估真实 validation-best checkpoint。local/remote hash `aa8971b415136a53725df7cd19ee16e7daa6c3485f79dcd3870abf8656c17007`，local/remote py_compile 通过。
- 最近完成 horizon/test evaluation：`modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1` e4 best，2026-05-11 21:50 CST 完成并读取 metric。average rollout loss `0.5632427931`；h50 `E_q=0.0853668251`、`E_v=0.3810359016`、`E_omega=0.3012500490`；mean `E_q=0.0437536468`、`E_v=0.2161035278`、`E_omega=0.2426603199`。相对 GRUTCN anchor e4 只有 h50 velocity 极微小改善，未接近 strongest targets；该结果已用于决策，判定 frozen-anchor `velocity_observer` 表达力不足，不继续单独细调该分支。Eval GPU watch 已停止，artifacts 保留。
- 最近 horizon/test evaluation 启动记录（已完成，见上一条）：`modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1` e4 best checkpoint，2026-05-11 21:22 CST 在远程 tmux 启动。
- Eval tmux：`eval_grutcn_velobserver_H20_e4_p1`。
- GPU watch tmux：`gpu_watch_eval_grutcn_velobserver_H20`。
- Eval log：`resources/experiments/modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1/logs/eval_horizontest_20260511_grutcn_velobserver_H20_e4_b32.log`。
- Eval command：`scripts/eval.py --dataset neurobemfullstate --predictor_type full_state --accelerator cuda --gpu_id 0 --eval_batch_size 32 --eval_horizons 1,10,25,50 --wandb_mode disabled --experiment_path resources/experiments/modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1`。
- Purpose：新协议允许读取 horizon/test；本次评估用于验证 `best_valid_loss=0.4720605` 虽未过旧冻结线，但 long-horizon 指标可能更好的假设。
- Startup check：eval log 确认加载 `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，test windows `50784`，H20/F50，GPU 正在计算；截至记录时 `horizon_summary.json` / `horizon_metrics.csv` 尚未生成，尚未读取 metric 内容。完成后必须记录关键 horizon/test 指标、是否影响后续调参/结构决策，以及结论。
- 当前 active training：无（最近完成 `modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1`，2026-05-11 20:42 CST 自然结束）。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1`。
- 训练 tmux：`modeldev_grutcn_velobserver_H20_from_anchor_e4_p1`（已退出）。
- GPU watch tmux：`modeldev_gpu_watch_grutcn_velobserver_H20`（已关闭）。
- 训练日志：`resources/experiments/modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`，即 GRUTCN H20 anchor e4。
- 配置：GRUTCN H20/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled，`--trainable_parameter_patterns velocity_observer`。
- 当前 GRUTCN 代码/结构：`scripts/dynamics_learning/models/grutcn.py` 新增 checkpoint-safe `velocity_observer_*`，输入 `base_feature + context + history_context + projected_x_last + dx_last + base_delta[:, 3:6]`，final projection zero-init、`velocity_observer_scale=0.003`，只在最终输出后写 `y[:, 3:6]`。不写 `y[:, 0:3]`，不写 `y[:, 6:12]`，不改 `encoder`、`base_decoder`、`state_initializer` 或 GRU decoder memory。失败的 `motion_encoder/motion_fusion`、temporal refiner、dual/raw 分支保留 key 兼容；训练时冻结 anchor，只训练 observer，用于验证此前 TCNLSTM velres 的 q/omega 回退是否主要来自共享参数漂移。
- train wiring：`scripts/config.py` 和 `scripts/train.py` 新增 `--trainable_parameter_patterns`，按逗号分隔子串筛选 trainable 参数，并把实际 trainable parameter names 写入 `train_summary.json`。当前正式训练只训练 `model.velocity_observer*` 参数。
- 检查/同步：local py_compile、`git diff --check` 通过；remote py_compile 通过；最终 local/remote hash 一致：`grutcn.py=fe649b2d138dceea90fb9107e983e784c9acd72a60dee24b7439e855132d824e`、`train.py=f73e558e6295910412d65284c02c2f53137224ac58ab3cfa94af342ae3ea716b`、`config.py=e4be9bdc3574b3ade4c896c1a58082b3041d28efc3e4ddd9ff22b86f5e48ead8`；一次错误 `rsync --relative` 生成的远程 `scripts/scripts` 已删除，正式训练文件 hash 已校正。
- remote smoke：`smoke_20260511_grutcn_velobserver_H20_from_anchor_e4_p1` 从 GRUTCN anchor e4 checkpoint 加载成功；missing keys 为新增 `velocity_observer_*` 及保留兼容分支，无 unexpected keys；只训练 `velocity_observer*` 参数；一批 train/valid finite：`train_loss_step=0.144`、`valid_loss_epoch=0.0821`；checkpoint、CSVLogger、`train_summary.json` 正常生成。只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- 训练结束：2026-05-11 20:42 CST 自然跑满 `max_epochs=5`（`early_stopped=false`），best 为 e4：`best_valid_loss=0.4720605`（`valid_q=0.0389170`、`valid_p=0.0434236`、`valid_v=0.1604807`、`valid_omega=0.2292394`）；best checkpoint `checkpoints/model-epoch=04-best_valid_loss=0.47.pth`；`train_summary.json` 已生成。判定：未达冻结候选线（`best_valid_loss <=0.47200` / `<=0.47180`），不做 locked audit；训练阶段未读取 horizon/test metric 内容，仅确认 `plots/testset` 目录存在；GPU watch tmux 已关闭，artifacts 保留。
- Gate：参考 GRUTCN anchor e4 best `0.4720816` 和 motiondiff e2 `0.4723428`。e0 `>0.4728` 直接停止不 audit；e0 `0.47210-0.47280` 最多观察 e1/e2；e0 `<=0.47205` 算健康。冻结候选优先要求 `best_valid_loss <=0.47180`，或 `<=0.47200` 且 `valid_q <=0.0390`、`valid_omega <=0.22935`、`valid_v <=0.16030`、`valid_p <=0.04330` 同步成立。若 `valid_q/valid_omega` 连续两轮回退或 `valid_v >0.1610`，按 validation-only failure 停止，不做 locked audit，不读取 horizon/test metric 内容。locked audit 只能在 natural finish 或 `train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 最近失败实验：`modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1` 已于 2026-05-11 18:33 CST 按 validation-only gate 停止。e0 `valid_loss_epoch=0.4618285`、`best_valid_loss=0.4618286`、`valid_q=0.0412260`、`valid_p=0.0322214`、`valid_v=0.1505020`、`valid_omega=0.2378793`；e1 `valid_loss_epoch=0.4618840`，best 未刷新，`valid_q=0.0412363`、`valid_p=0.0322240`、`valid_v=0.1504817`、`valid_omega=0.2379420`；e2 `valid_loss_epoch=0.4617192`、`best_valid_loss=0.4617194`、`valid_q=0.0412369`、`valid_p=0.0322030`、`valid_v=0.1503207`、`valid_omega=0.2379586`。判定：e2 total/v 改善但 q/omega 从 e0 到 e2 连续回退；best 未达冻结线，不做 locked audit。执行：仅停止该实验训练 tmux、匹配实验路径进程和 GPU watch；artifacts 保留；训练阶段未读取 horizon/test metric 内容，仅确认 `plots/testset` 存在。
- 下面较旧的 active-training 条目保留为历史记录；继承时以本节最上方的 `grutcn_velobserver` 为准。
- 当前 active training：`modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1`，2026-05-11 17:11 CST 已在远程 tmux 启动。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1`。
- 训练 tmux：`modeldev_tcnlstm_velres_H10_from_attitude_e3_p1`。
- GPU watch tmux：`modeldev_gpu_watch_tcnlstm_velres_H10`。
- 训练日志：`resources/experiments/modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260511_tcnlstm_velres_H10_from_attitude_e3_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`，即当前最稳 `TCNLSTM attitude H10 e3` anchor。
- 配置：TCNLSTM H10/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
- 当前 TCNLSTM 代码/结构：`scripts/dynamics_learning/models/tcnlstm.py` 在 H10 attitude e3 anchor 上新增 checkpoint-safe `velocity_residual_*` 分支，只写 `y[:, 3:6]` velocity slice；final projection zero-init，`velocity_residual_scale=0.005`。该分支用 `attitude_input + base_delta[:, :6]` 生成小尺度 velocity damping residual，目标是吸收 actuator lag / aero drag / hidden disturbance 的慢变量速度误差。不写 `y[:, 6:12]`，不改 `encoder`、TCN-compatible `decoder`、`base_feature = anchor_seq[:, -1, :]`、`state_initializer` 或已训练 attitude path；`gru_context_bridge`、`attitude_output_residual_v2`、`coupled_residual_*` 等失败候选仍只保留 key 兼容，不参与 forward。
- 检查/同步：local py_compile、`git diff --check` 通过；local/remote `tcnlstm.py` hash 均为 `94ad150e9a80eee0ab16fe1b5be3514b93dd9970d18d7273de5f978e79027c10`；remote `python3 -m py_compile` 通过。remote smoke `smoke_20260511_tcnlstm_velres_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 为新增/保留兼容分支（含 `velocity_residual_*`），无 unexpected keys；一批 train/valid finite：`train_loss_step=0.181`、`valid_loss_epoch=0.563`（1-batch smoke），checkpoint、CSVLogger、`train_summary.json` 正常；只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- 启动巡检：训练 tmux、GPU watch tmux、训练进程均存在，日志实时写入，GPU 约 `2053/8188 MiB`、util `37-40%`，e0 train steps finite；`train_summary.json` 尚未生成；训练阶段没有读取 horizon/test metric 内容，只确认 `plots/testset` 路径存在。
- Gate：参考 H10 attitude e3 best `0.4615005`。e0 `>0.46185` 或 e0 同时 `valid_q >0.04123` 且 `valid_omega >0.23785` 直接停止不 audit；e0 `0.46155-0.46185` 且未触发 q/omega 保护线则最多观察 e1/e2；e0 `<=0.46150` 算健康。冻结候选优先要求 `best_valid_loss <=0.46130`，或 `<=0.46145` 且 `valid_q <0.04118`、`valid_omega <0.23746`、`valid_v <=0.15030` 同步成立。若 q/omega 连续两轮回退，按 validation-only failure 停止，不做 locked audit，不读取 horizon/test metric 内容。locked audit 只能在 natural finish 或 `train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 当前 active training：`modeldev_20260511_tcnlstm_consolidation_H10_from_attitude_e3_p1`，2026-05-11 14:30 CST 已在远程 tmux 启动。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260511_tcnlstm_consolidation_H10_from_attitude_e3_p1`。
- 训练 tmux：`modeldev_tcnlstm_consolidation_H10_from_attitude_e3_p1`。
- GPU watch tmux：`modeldev_gpu_watch_tcnlstm_consolidation_H10`。
- 训练日志：`resources/experiments/modeldev_20260511_tcnlstm_consolidation_H10_from_attitude_e3_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260511_tcnlstm_consolidation_H10_from_attitude_e3_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`，即当前最稳 `TCNLSTM attitude H10 e3` anchor。
- 配置：TCNLSTM H10/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=8e-7`，`cosine_lr=2e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
- 当前 TCNLSTM 代码/结构：`scripts/dynamics_learning/models/tcnlstm.py` 将刚失败的 `coupled_residual_*` 从 forward 旁路，只保留模块/key 兼容；`attitude_output_residual_v2`、`attitude_fine_*`、`decoder_state_residual*`、`gru_context_bridge*`、`lag_context*`、`long_delta*` 也不参与 forward。当前 forward 回到 H10 attitude e3 的 TCN anchor + context_delta + attitude_delta 路线做 ultralow consolidation；不改 `encoder`、TCN-compatible `decoder`、`base_feature = anchor_seq[:, -1, :]`、`state_initializer`。
- 检查/同步：local py_compile、`git diff --check` 通过；local/remote `tcnlstm.py` hash 均为 `300bfb2af70240924a8c9d66d3f199f2660c31a17a15c7e5ddfa1e0ade93674d`；remote `python3 -m py_compile` 通过。remote smoke `smoke_20260511_tcnlstm_consolidation_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 为不参与 forward 的兼容分支（含 `coupled_residual_*`），无 unexpected keys；一批 `train_loss_step=0.176`、`valid_loss_epoch=0.1178` finite，checkpoint、CSVLogger、`train_summary.json` 正常；只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- 启动巡检：训练 tmux、GPU watch tmux、训练进程均存在，日志实时写入，GPU 约 `1979/8188 MiB`、util `30%`，e0 train steps finite；`train_summary.json` 尚未生成；训练阶段没有读取 horizon/test metric 内容，只确认 `plots/testset` 路径存在。
- Gate：参考 H10 attitude e3 best `0.4615005`。e0 `>0.46185` 直接停止不 audit；e0 `0.46155-0.46185` 最多观察 e1/e2；e0 `<=0.46150` 算健康。冻结候选优先要求 `best_valid_loss <=0.46130`，或 `<=0.46145` 且 `valid_q <0.04118`、`valid_omega <0.23746`、`valid_v <=0.15030` 同步成立。若 q/omega 连续两轮回退，按 validation-only failure 停止，不做 locked audit，不读取 horizon/test metric 内容。locked audit 只能在 natural finish 或 `train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 最新巡检：2026-05-11 16:02 CST consolidation H10 e2 validation 完成并触发 “q/omega 连续两轮回退” gate，训练已 early stop 结束，`train_summary.json` 已生成且 `best_model_path` 指向 e0 checkpoint。e0：`train_loss_epoch=0.3324556`、`valid_loss_epoch=0.4616246`、`best_valid_loss=0.4616245`、`valid_q=0.0412200`、`valid_p=0.0322120`、`valid_v=0.1503584`、`valid_omega=0.2378343`。e1：`train_loss_epoch=0.3321951`、`valid_loss_epoch=0.4617252`、best 未刷新，`valid_q=0.0412524`、`valid_p=0.0321666`、`valid_v=0.1502967`、`valid_omega=0.2380092`。e2：`valid_loss_epoch=0.4617007`、best 仍为 e0，`valid_q=0.0412584`、`valid_p=0.0321514`、`valid_v=0.1502234`、`valid_omega=0.2380675`。判定：q/omega e0->e1->e2 连续回退，validation-only failure；不冻结、不 locked audit。执行：训练进程已退出，GPU watch tmux 已关闭；artifacts 保留。训练阶段未读取 horizon/test metric 内容，仅确认 `plots/testset` 存在。
- 最近失败实验：`modeldev_20260511_tcnlstm_coupledlowrank_H10_from_attitude_e3_p1`，2026-05-11 14:18 CST 按 validation-only gate 停止。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260511_tcnlstm_coupledlowrank_H10_from_attitude_e3_p1`。
- 已停止训练 tmux：`modeldev_tcnlstm_coupledlowrank_H10_from_attitude_e3_p1`。
- 已停止 GPU watch tmux：`modeldev_gpu_watch_tcnlstm_coupledlowrank_H10`。
- 训练日志：`resources/experiments/modeldev_20260511_tcnlstm_coupledlowrank_H10_from_attitude_e3_p1/logs/train_phase1.log`。
- GPU watch 日志：`resources/experiments/modeldev_20260511_tcnlstm_coupledlowrank_H10_from_attitude_e3_p1/logs/gpu_watch.log`。
- 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`，即当前最稳 `TCNLSTM attitude H10 e3` anchor。
- 配置：TCNLSTM H10/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=1.5e-6`，`cosine_lr=4e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
- 关键 validation：e0 `train_loss_epoch=0.3328158`、`valid_loss_epoch=0.4621493`、`best_valid_loss=0.4621493`，`valid_q=0.0412490`、`valid_p=0.0322318`、`valid_v=0.1506851`、`valid_omega=0.2379835`。
- 判定：e0 超过预声明 stop line `0.4620`，且不满足 `best_valid_loss <=0.4613` 或 `<=0.46145` 加 q/omega/v 同步健康的冻结条件；不做 locked audit，不读取 horizon/test metric 内容（仅确认 `plots/testset` 目录存在）。
- 清理范围：只停止该实验训练 tmux、对应 GPU watch 和匹配该实验路径的训练进程；artifacts 保留（e0 checkpoint、`last_model.pth`、CSV/logs），`train_summary.json` 因手动 gate 停止不存在；GPU 回到约 `504/8188 MiB`、util `2%`。
- 当前 TCNLSTM 代码/结构：`scripts/dynamics_learning/models/tcnlstm.py` 新增 checkpoint-safe `coupled_residual_*` low-rank latent residual。它使用 `attitude_input + base_delta` 生成低秩系数，再经 zero-init `coupled_residual_basis` 投到完整 `output_size`，`coupled_residual_scale=0.01`，作为全输出空间耦合修正；初始为 no-op。刚失败的 `attitude_output_residual_v2` forward 路径已旁路但模块/key 保留；不改 `encoder`、TCN `decoder` anchor、`base_feature = anchor_seq[:, -1, :]`、`state_initializer`。
- 检查/同步：local py_compile、`git diff --check` 通过；local/remote `tcnlstm.py` hash 均为 `160804053c79e04f33512005880caece5370651c2dd9d7aead4a56df2958d434`；remote `python3 -m py_compile` 通过。remote smoke `smoke_20260511_tcnlstm_coupledlowrank_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 为新增/保留兼容分支（含 `coupled_residual_*`），无 unexpected keys；一批 `train_loss_step=0.180`、`valid_loss_epoch=0.676` finite，checkpoint、CSVLogger、`train_summary.json` 正常；只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- 最近完成实验 `modeldev_20260511_tcnlstm_attoutv2_H10_from_attitude_e3_p1` 已于 2026-05-11 12:55 CST 因 validation-only gate 失败停止；不做 locked audit，不读取 horizon/test metric 内容。关键 validation：e0 `valid_loss_epoch=0.4621532` / `best_valid_loss=0.4621533`，`valid_q=0.0412496`、`valid_p=0.0322328`、`valid_v=0.1506867`、`valid_omega=0.2379841`；e1 `valid_loss_epoch=0.4621446` / `best_valid_loss=0.4621447`，`valid_q=0.0412693`、`valid_p=0.0322439`、`valid_v=0.1505715`、`valid_omega=0.2380598`；artifacts 保留，`train_summary.json` 因手动 gate 停止不存在。
- 当前 TCNLSTM 代码/结构：`scripts/dynamics_learning/models/tcnlstm.py` 新增 checkpoint-safe `attitude_output_residual_v2`。它在最终输出 `y` 形成后，用 `attitude_input + y[:, 6:12]` 生成 output-space residual，只加到 `y[:, 6:12]`；final projection zero-init，`attitude_output_residual_scale=0.005`。不改 `encoder`、TCN `decoder` anchor、`base_feature = anchor_seq[:, -1, :]`、`state_initializer`。`gru_context_bridge` 不属于 attitude H10 e3 checkpoint 的已训练路径，forward 已旁路，只保留 key 兼容。
- 检查/同步：local py_compile、`git diff --check` 通过；local/remote `tcnlstm.py` hash 均为 `bc7e5b7e8753a67109c08272ce027cd630e10cb8f562706c26d8265e5d5dcdaf`；remote `python3 -m py_compile` 通过。remote smoke `smoke_20260511_tcnlstm_attoutv2_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 为新增/保留兼容分支，无 unexpected keys，一批 `train_loss_step=0.181`、`valid_loss_epoch=0.118`，checkpoint、CSVLogger、`train_summary.json` 正常；只确认 `plots/testset` 存在，未读取 horizon/test metric 内容。
- Gate：参考 H10 attitude best `0.4615005`。e0 `>0.4620` 直接停止不 audit；e0 `0.4616-0.4620` 最多观察 e1/e2；e0 `<=0.46155` 算健康。冻结候选优先要求 `best_valid_loss <=0.4613`，或 `<=0.46145` 且 `valid_q <0.04118`、`valid_omega <0.23746`、`valid_v <=0.1505` 同步成立。若 q/omega 连续两轮回退，按 validation-only failure 停止，不做 locked audit，不读取 horizon/test metric 内容。locked audit 只能在 natural finish 或 `train_summary.json` 指向 validation-selected best checkpoint 后启动。
- Automation 职责纠偏：`AGENTS.md` 第 12/13 条要求核心 automation 是模型冲刺总控，不是单实验 watcher。单个 validation-only failure 后，只能停止对应实验运行态并转入下一候选；总控 heartbeat 应继续存在，在无 active training 时推进下一结构/训练方案，在有 active training 时巡检该实验。误删旧单实验 automation 后，已恢复 heartbeat automation `automation`（名称 `继续模型冲刺总控`，每 30 分钟）。巡检只检查 horizon/test 文件或目录是否存在，不能读取 metric 内容。
- 上一状态：此前无 active training；最近实验 `modeldev_20260511_grutcn_motiondiff_H20_from_anchor_e4_p1` 已因 validation-only gate 失败停止。
- 远程路径：`resources/experiments/modeldev_20260511_grutcn_motiondiff_H20_from_anchor_e4_p1`。
- 训练 tmux：`modeldev_grutcn_motiondiff_H20_from_anchor_e4_p1`。
- GPU watch tmux：`modeldev_gpu_watch_grutcn_motiondiff_H20`。
- 训练日志：`resources/experiments/modeldev_20260511_grutcn_motiondiff_H20_from_anchor_e4_p1/logs/train_phase1.log`。
- 状态：2026-05-11 10:55 CST 已停止该实验训练 tmux、匹配实验路径的训练进程和 GPU watch；GPU 回到约 `495/8188 MiB`、util `0%`。关键 validation：e0 `train_loss_epoch=0.2315615`、`valid_loss_epoch=0.4725016`、`best_valid_loss=0.4725017`、`valid_q=0.0389248`、`valid_p=0.0435194`、`valid_v=0.1608349`、`valid_omega=0.2292226`；e1 `train_loss_epoch=0.2325923`、`valid_loss_epoch=0.4726623`、`best_valid_loss=0.4725017`、`valid_q=0.0389161`、`valid_p=0.0435071`、`valid_v=0.1610198`、`valid_omega=0.2292190`；e2 刷新本实验 best：`train_loss_epoch=0.2322338`、`valid_loss_epoch=0.4723428`、`best_valid_loss=0.4723428`、`valid_q=0.0389410`、`valid_p=0.0435035`、`valid_v=0.1606968`、`valid_omega=0.2292014`。判定：e2 未达到冻结候选线 `best_valid_loss <=0.4718`，也不满足 `<=0.4720` 且 q/omega/v/p 同步健康，按预声明 `e2 未达到冻结线` 处理为 validation-only failure；不做 locked audit，horizon/test metric 内容未读取，只确认 `plots/testset` 路径存在；artifacts 保留。
- 当前新代码：`scripts/dynamics_learning/models/grutcn.py` 新增 checkpoint-safe `motion_encoder` + `motion_fusion`，用 `[x, dx]` 差分观测流在 encoder-side 融合 actuator lag / aero lag / filter-state 证据，final projection zero-init；不直接写 `y`、不改 decoder state。local/remote hash `7897bb1d60f6a8d8f696418ebfbcd5900077440921cc89e6b37bf674d58e5b9d`；local py_compile、`git diff --check`、remote py_compile 通过。
- remote smoke：`smoke_20260511_grutcn_motiondiff_H20_from_anchor_e4_p1` 从 `modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth` 加载，missing keys 仅新增 motion 分支及既有 refiner/dual/raw 分支，无 unexpected keys；`train_loss_step=0.119`、`valid_loss_epoch=0.0772`，checkpoint 与 `train_summary.json` 正常生成；horizon/test metric 内容未读取，只确认 `plots/testset` 路径存在。
- motiondiff 配置：GRUTCN H20/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`、`limit_val_batches=0.5`，`warmup_lr=1.5e-6`、`cosine_lr=4e-7`、`warmup_steps=50`、`cosine_steps=1000`，early stopping patience `2`、min_delta `2e-5`，WANDB disabled；init checkpoint 为 `resources/experiments/modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。
- motiondiff gate：参考 GRUTCN anchor e4 best `0.4720816`、anchor-noise e5 `0.4722003`、dual/raw e2 `0.4721887`。e0 `>0.4732` 判定 motion-diff fusion 破坏 GRUTCN 校准，停止不 audit；e0 `0.4722-0.4732` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v`、`valid_p` 是否同步健康；e0 `<=0.4721` 算健康。冻结候选优先要求 `best_valid_loss <=0.4718`；或 `<=0.4720` 且 `valid_q <=0.0390`、`valid_omega <=0.22935`、`valid_v <=0.1606`、`valid_p <=0.04330` 同步成立。若 q/omega 连续两轮回退或 `valid_v >0.1612`，不读取 horizon/test metric 内容、不做 locked audit；locked audit 只能在 natural finish/`train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 最近完成实验：`modeldev_20260511_tcnlstm_grubridge_H10_from_attitude_e3_p1`。
- grubridge 远程路径：`resources/experiments/modeldev_20260511_tcnlstm_grubridge_H10_from_attitude_e3_p1`。
- grubridge 状态：2026-05-11 09:03 CST 因 validation-only gate 失败停止；训练 tmux `modeldev_tcnlstm_grubridge_H10_from_attitude_e3_p1`、匹配该实验路径的训练进程和 GPU watch `modeldev_gpu_watch_tcnlstm_grubridge_H10` 已停止，GPU 回到约 `495/8188 MiB`、util `0%`。
- grubridge 关键结果：e0 `train_loss_epoch=0.3323885`，`valid_loss_epoch=0.4620850`，`best_valid_loss=0.4620851`，`valid_q=0.0412587`，`valid_p=0.0322910`，`valid_v=0.1505874`，`valid_omega=0.2379481`。e0 超过预声明 stop line `0.4620`，判定 GRU context bridge 破坏 H10 anchor 校准，停止不 audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在；checkpoint `model-epoch=00-best_valid_loss=0.46.pth` 和 `last_model.pth` 保留，artifacts 保留。
- 最近失败实验：`modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1`，已于 2026-05-11 08:19 CST 因 validation-only gate 失败停止。关键结果：e2 `valid_loss_epoch=0.4614547`，`best_valid_loss=0.4614547`，`valid_q=0.0412437`，`valid_p=0.0320399`，`valid_v=0.1501071`，`valid_omega=0.2380640`；未达到冻结候选线，不做 locked audit，不读取 horizon/test metric 内容，artifacts 保留。
- 当前 TCNLSTM 代码/协议：`scripts/dynamics_learning/models/tcnlstm.py` 仍包含 checkpoint-safe `gru_context` + `gru_context_bridge` 等兼容模块，但 grubridge 已验证 validation-only 失败；不要继续训练、audit 或细调该 TCNLSTM grubridge 候选。
- latest completed candidate：`modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1`。
- 2026-05-10 20:58 CST 启动 TCNLSTM true-anchor screening：从 TCN H10 best checkpoint 初始化，H10/F50，batch `64`、accumulate `8`、effective batch `512`，epochs `6`，`warmup_lr=1e-5`、`cosine_lr=3e-6`。tmux：`modeldev_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1`；GPU watch：`modeldev_gpu_watch_tcnlstm_trueanchor_H10`；train log：`resources/experiments/modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1/logs/train_phase1.log`。
- 2026-05-10 21:18 CST TCNLSTM true-anchor e0 validation 完成且健康：`train_loss_epoch=0.3388692`，`valid_loss_epoch=0.4623863`，`best_valid_loss=0.4623865`，`valid_q=0.0414418`，`valid_p=0.0322573`，`valid_v=0.1502742`，`valid_omega=0.2384134`。该结果低于 e0 停止线 `0.58`、健康线 `0.55`，也低于 TCN H10 validation baseline `0.533292`；checkpoint `model-epoch=00-best_valid_loss=0.46.pth` 已产生。
- 2026-05-10 21:41 CST TCNLSTM true-anchor e1 validation 完成：`train_loss_epoch=0.3363273`，`valid_loss_epoch=0.4637655`，`best_valid_loss=0.4623865`，`valid_q=0.0413985`，`valid_p=0.0323606`，`valid_v=0.1515890`，`valid_omega=0.2384175`。e1 未刷新 e0 best，但仍显著低于 TCN H10 validation baseline；训练已进入 e2，继续等 e2 验证作为是否冻结/继续的关键窗口。当前不 audit，horizon/test metric 内容未读取。
- 2026-05-10 22:02 CST TCNLSTM true-anchor e2 validation 完成：`train_loss_epoch=0.3350681`，`valid_loss_epoch=0.4640823`，`best_valid_loss=0.4623865`，`valid_q=0.0413460`，`valid_p=0.0324449`，`valid_v=0.1518466`，`valid_omega=0.2384447`。e2 仍远低于 TCN H10 validation baseline `0.533292`，但未刷新 e0 best 且没有形成持续下降；训练已进入 e3，继续观察自然收敛/early stopping，不冻结、不 audit，horizon/test metric 内容未读取。
- 2026-05-10 22:50 CST TCNLSTM true-anchor 已自然 early stop：e3 `valid_loss_epoch=0.4634408`，`best_valid_loss=0.4623865`，`train_summary.json` 显示 `early_stopped=True`、`stopped_epoch=4`，`best_model_path` 指向 `resources/experiments/modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth`。该 e0 checkpoint 已冻结为 locked audit 候选；训练阶段未读取 horizon/test metric 内容，实验目录暂未发现 horizon/test 文件。
- 2026-05-10 22:54 CST 已启动 locked audit：audit id `lockaudit_20260510_tcnlstm_trueanchor_H10_e0_p1`；tmux `audit_tcnlstm_trueanchor_H10_e0_p1`；日志 `resources/experiments/modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1/logs/eval_lockaudit_20260510_tcnlstm_trueanchor_H10_e0_p1_b32.log`。Eval 命令使用 `scripts/eval.py --dataset neurobemfullstate --predictor_type full_state --accelerator cuda --gpu_id 0 --eval_batch_size 32 --eval_horizons 1,10,25,50 --wandb_mode disabled --experiment_path resources/experiments/modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1`；启动检查显示 tmux/进程 alive、GPU 进程已创建、日志开始于 `Seed set to 10`，horizon/test metric 内容尚未读取。若 batch `32` OOM 或崩溃，按既有协议用 batch `16` 重试同一冻结 checkpoint，并记录 `_b16.log`。
- 2026-05-10 23:04 CST locked audit 完成：batch `32` 无 OOM retry，average rollout loss `0.5286130309`；h1 `E_q=0.0014212`、`E_v=0.0099771`、`E_omega=0.0537535`；h10 `E_q=0.0173599`、`E_v=0.0862777`、`E_omega=0.2137942`；h25 `E_q=0.0445177`、`E_v=0.1855211`、`E_omega=0.2686314`；h50 `E_q=0.0909456`、`E_v=0.3444223`、`E_omega=0.3350359`；mean `E_q=0.0458820`、`E_v=0.1851967`、`E_omega=0.2572233`。结论：partial locked win，h50 velocity 同时优于 `gru_H20=0.353015` 和 `tcn_H10=0.347036`，但 h50 quaternion、h50 omega、mean quaternion 未击败 strongest GRU targets；不是全面 winner。horizon/test metric 内容仅在 locked audit 完成后读取；artifacts 全部保留，未清理；audit GPU watch `modeldev_gpu_watch_tcnlstm_audit_H10` 已停止。
- 2026-05-10 20:48 CST 已完成保守 resume64 确认：从原实验 `last_model.pth` resume 到 epoch 4，micro batch `64`、accumulate `8`、effective batch `512`；训练自然到 `max_epochs=6` 结束，未再出现整机重启或普通 PyTorch OOM。

实验 id：

- `modeldev_20260510_grutcn_dualraw_H20_resume64_from_last_p1`

远程路径：

- `resources/experiments/modeldev_20260510_grutcn_dualraw_H20_resume64_from_last_p1`

tmux：

- `modeldev_grutcn_dualraw_H20_resume64_from_last_p1`

日志：

- `logs/train_resume64.log`

resume checkpoint：

- `resources/experiments/modeldev_20260510_grutcn_dualraw_H20_from_ultralow_e4_p1/checkpoints/last_model.pth`

GPU watch：

- tmux：`modeldev_gpu_watch_dualraw_resume64_H20`
- log：`resources/experiments/modeldev_20260510_grutcn_dualraw_H20_resume64_from_last_p1/logs/gpu_watch.log`

配置：

- model：GRUTCN
- history length：H20
- unroll：F50
- epochs：`6`
- batch：`64`
- accumulate：`8`
- effective batch：`512`
- `limit_train_batches=0.25`
- `limit_val_batches=0.5`
- seed：`10`
- `warmup_lr=4e-6`
- `cosine_lr=1e-6`
- `warmup_steps=50`
- `cosine_steps=1000`
- early stopping patience：`3`
- min_delta：`5e-5`
- no SWA
- WANDB disabled

最新状态：

- 正式训练已于 2026-05-10 16:26 CST 在 tmux 启动。
- startup patrol：tmux/process alive；RTX 4060 约 `5525/8188 MiB`、util `39%`。
- 日志确认从 ultralow e4 checkpoint 初始化。
- missing keys 只包含新增 temporal/dual/raw 模块；无 unexpected keys。
- e3 validation 已完成但未刷新 best，epoch 4 已开始。
- e0 `train_loss=0.233`，`valid_loss=0.4729816`，`best_valid_loss=0.4729815`，`valid_q=0.0389585`，`valid_p=0.0433674`，`valid_v=0.1613430`，`valid_omega=0.2293125`。
- e1 `train_loss=0.23257`，`valid_loss=0.4730309`，`best_valid_loss=0.4729815`，`valid_q=0.0390640`，`valid_p=0.0434037`，`valid_v=0.1609298`，`valid_omega=0.2296333`。
- e2 `train_loss=0.232`，`valid_loss=0.4721888`，`best_valid_loss=0.4721887`，`valid_q=0.0390779`，`valid_p=0.0433664`，`valid_v=0.1599812`，`valid_omega=0.2297631`。
- e3 `valid_loss=0.4732611`，`best_valid_loss=0.4721887`，`valid_q=0.0391039`，`valid_p=0.0434005`，`valid_v=0.1609785`，`valid_omega=0.2297781`。
- 决策：e3 未刷新 e2 best，且仍未达到 preferred audit band `<=0.4720`；继续等 e4 作为后续确认，不做 locked audit。若 e4 不刷新到 `<=0.4720` 或 submetrics 明显改善不足，则优先停止该候选并 pivot。
- artifacts：当前可见 checkpoints 包括 `checkpoints/model-epoch=01-best_valid_loss=0.47.pth`、`checkpoints/model-epoch=02-best_valid_loss=0.47.pth`、`checkpoints/model-epoch=03-best_valid_loss=0.47.pth` 和 `last_model.pth`。
- horizon/test metric 内容未读取；这是当时旧协议下的记录。自 2026-05-11 新协议起，训练/开发阶段允许读取 horizon/test metric 并用于模型改进，但必须记录指标和决策影响。
- 2026-05-10 17:28 CST 巡检异常：SSH 连接 `gpu4060` 超时，未能确认 e4 最新状态；未读取 horizon/test metric 内容，下次心跳重试。
- 2026-05-10 17:50 CST 远程连通性升级异常：`ping -c 2 192.168.1.108` 为 `100% packet loss`，`nc -vz -G 4 192.168.1.108 22` 连接 22 端口超时；当前从本机看 `gpu4060` 主机不可达，不只是 SSH 命令慢。未读取 horizon/test metric 内容；下一步先重试连通性，恢复后立刻检查 e4/e5 validation。
- 2026-05-10 18:38 CST 远程恢复后确认：`gpu4060` 已重启，`uptime` 约 0 分钟，GPU 空闲约 `381/8188 MiB`，tmux 缺失且训练进程不存在；训练日志停在 epoch 4 约 `554/771`（约 72%）并出现 NUL 尾部，未完成 e4 validation。CSV 仍只有 e0-e3 validation rows，checkpoint 只有 e1/e2/e3/last；日志未出现 `CUDA out of memory`、`Traceback`、`Killed`。判断为整机/驱动/WSL/电源级硬中断或重启，不是普通 PyTorch OOM；未读取 horizon/test metric 内容。当前 best 仍是 e2 `best_valid_loss=0.4721887`，未达到 preferred audit band，暂不 audit。
- 2026-05-10 18:40 CST 已启动远程 GPU watch tmux：`modeldev_gpu_watch_dualraw_H20`，日志 `resources/experiments/modeldev_20260510_grutcn_dualraw_H20_from_ultralow_e4_p1/logs/gpu_watch.log`，每 10 秒记录 `nvidia-smi` GPU memory/util/temp/power 和 compute apps，用于后续判断是否显存、驱动或整机问题。
- 2026-05-10 19:24 CST 按用户要求保守恢复：新实验 `modeldev_20260510_grutcn_dualraw_H20_resume64_from_last_p1` 保留原断点 artifacts，不覆盖旧目录；显式激活 conda `dynamics_learning`，remote py_compile 通过，`grutcn.py` hash 仍为 `8c5ae1489fa1961df585f15789e64932cc9b9f2e10e6a2a25d163a76cf7a7f92`；训练命令使用 `--resume_from_checkpoint` 指向原实验 `last_model.pth`，`--batch_size 64 --accumulate_grad_batches 8`，W&B disabled。未读取 horizon/test metric 内容。
- 2026-05-10 19:50 CST resume64 e4 validation 完成：`valid_loss=0.4732777`，`best_valid_loss=0.4732777`，`valid_q=0.0389887`，`valid_p=0.0435931`，`valid_v=0.1611713`，`valid_omega=0.2295246`；未刷新原 dual/raw e2 best `0.4721887`，也未到 preferred audit band `<=0.4720`。训练已进入 e5，继续等 e5 最终确认；当前不 audit，未读取 horizon/test metric 内容。
- 2026-05-10 20:48 CST resume64 e5 validation 完成：`valid_loss=0.4728829`，`best_valid_loss=0.4728828`，`valid_q=0.0389932`，`valid_p=0.0435362`，`valid_v=0.1608482`，`valid_omega=0.2295049`；仍未刷新原 dual/raw e2 best `0.4721887`，也未达到 preferred audit band `<=0.4720`。结论：dual/raw resume64 不是冻结候选，不做 locked audit；horizon/test metric 内容未读取；artifacts 保留；专属 GPU watch tmux `modeldev_gpu_watch_dualraw_resume64_H20` 已停止。
- 下一步：停止继续围绕 dual/raw 细调，不 audit；pivot 到 TCNLSTM 温启动或下一种更强 latent-context 结构，仍遵守只看 train/validation/log/checkpoint/架构信号的规则。
- 2026-05-10 20:58 CST TCNLSTM 结构 pivot：按 Harvey 子 agent 建议修正 `scripts/dynamics_learning/models/tcnlstm.py`，让 TCN-compatible anchor 分支命名为 `decoder`，从而旧 TCN checkpoint 的 `model.encoder.*` / `model.decoder.*` 直接加载到 base predictor；将 latent correction head 改名为 `context_decoder` 并 final zero-init；新增 checkpoint-safe temporal refiner，但 anchor decoder 使用未 `LayerNorm` 的 raw TCN feature，以保持 TCN checkpoint 行为。local/remote hash `ab3f30a6dd0eaa02369a7752739bf26e1210d5708cb9b3668efd2ccc50d8a6ab`；local/remote py_compile 通过；H10 true-anchor smoke 从 TCN H10 checkpoint 加载无 unexpected keys，missing 仅新增 context/refiner/LSTM 分支，一批次 `train_loss_step=0.189`、`valid_loss_epoch=0.123`，有限且量级正常。未读取 horizon/test metric 内容。
- 2026-05-10 23:12 CST TCNLSTM attitude-aware 改动：根据结果分析 agent 和架构 agent 的策略节点建议，在 `scripts/dynamics_learning/models/tcnlstm.py` 新增 attitude attention + attitude correction branch；分支输入包含 `decoder_feature`、`projected_x_last`、`history_context`、通用 attention context 和独立 attitude attention context，只给输出 `[:, 6:12]` 的 `dtheta/delta_omega` 写 residual。保持 TCN checkpoint anchor：不改 `encoder`、不改名 `decoder`、不把 normalized/refined feature 接到 base decoder，`base_feature = anchor_seq[:, -1, :]` 不变。新增 `attitude_decoder` final zero-init、`attitude_delta_scale=0.05`，从旧 TCNLSTM e0 checkpoint 初始化时是 no-op。local/remote hash `527599fd329a5f4c9f010da276bdc34905e2297a73aa1bdfb01ea1471c80cc7d`；local/remote py_compile 和 `git diff --check` 通过；remote smoke `smoke_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1` 从 e0 checkpoint 加载，missing 只包含新增 attitude keys、无 unexpected keys，一批 train/valid finite：`train_loss_step=0.175`，`valid_loss_epoch=0.590`。未读取 horizon/test metric 内容。
- 2026-05-10 23:12 CST 启动 active 训练 `modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1`：从 `modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth` 初始化，H10/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`、`limit_val_batches=0.5`，`warmup_lr=3e-6`、`cosine_lr=8e-7`、`warmup_steps=50`、`cosine_steps=1000`，early stopping patience `2`、min_delta `2e-5`，WANDB disabled。tmux `modeldev_tcnlstm_attitude_H10_from_trueanchor_e0_p1`；GPU watch `modeldev_gpu_watch_tcnlstm_attitude_H10`；log `resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/logs/train_phase1.log`。启动巡检显示 tmux/process/GPU healthy，RTX 4060 约 `1957/8188 MiB`、util `41%`，train steps finite，horizon/test metric 内容未读取。screen gate：e0 `<=0.55` 才健康；继续训练需 best 优于 `0.4623865` 且 validation q/omega 不回退，尤其 `valid_q < 0.04135`、`valid_omega < 0.2383`，同时 `valid_v` 不明显回退；只有 total validation + q/omega 同步过门才考虑 locked audit。
- 2026-05-10 23:36 CST TCNLSTM attitude e0 validation 完成并刷新 true-anchor e0：`train_loss_epoch=0.3352483`，`valid_loss_epoch=0.4619423`，`best_valid_loss=0.4619424`，`valid_q=0.0411794`，`valid_p=0.0324030`，`valid_v=0.1508975`，`valid_omega=0.2374622`。相对 true-anchor e0 `best_valid_loss=0.4623865`、`valid_q=0.0414418`、`valid_omega=0.2384134` 有小幅同步改善，`valid_v` 从 `0.1502742` 轻微回退但仍接近；训练已进入 e1，继续等 e1/e2 确认趋势，当前不冻结、不 locked audit；horizon/test metric 内容未读取，artifacts 保留。
- 2026-05-11 00:04 CST TCNLSTM attitude e1 validation 完成并小幅刷新 best：`train_loss_epoch=0.3338704`，`valid_loss_epoch=0.4618453`，`best_valid_loss=0.4618454`，`valid_q=0.0412394`，`valid_p=0.0321995`，`valid_v=0.1505699`，`valid_omega=0.2378366`。相对 e0 总 loss 和 `valid_v` 改善，`valid_q/valid_omega` 较 e0 略回退但仍优于 true-anchor e0，且仍在 attitude gate 内；训练已进入 e2，继续等 e2 确认趋势，当前不冻结、不 locked audit；horizon/test metric 内容未读取，artifacts 保留。
- 2026-05-11 00:25 CST TCNLSTM attitude e2 validation 完成但未刷新 e1 best：`train_loss_epoch=0.3332587`，`valid_loss_epoch=0.4620062`，`best_valid_loss=0.4618454`，`valid_q=0.0412714`，`valid_p=0.0321716`，`valid_v=0.1506382`，`valid_omega=0.2379250`。e2 总 loss、`valid_q`、`valid_omega` 较 e1 回退，但 q/omega 仍优于 true-anchor e0 且在 attitude gate 内；训练已进入 e3，继续观察自然收敛/early stopping。当前不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在；artifacts 保留。
- 2026-05-11 00:32 CST 本窗口 heartbeat automation 已更新为当前 active experiment `modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1`。只读交叉巡检确认训练仍在 e3，GPU watch 正常，`train_summary.json` 尚未生成，当前 validation-selected best 是 e1 checkpoint `checkpoints/model-epoch=01-best_valid_loss=0.46.pth`。不要手动停止训练后直接 audit，因为无 summary 时 eval 的 checkpoint 自动选择可能不稳定；等待 natural early stop/max epoch 写出 `train_summary.json` 后再冻结 audit。horizon/test metric 内容未读取。
- 2026-05-11 00:47 CST TCNLSTM attitude e3 validation 刷新 best：`train_loss_epoch=0.3327240`，`valid_loss_epoch=0.4615006`，`best_valid_loss=0.4615005`；checkpoint `checkpoints/model-epoch=03-best_valid_loss=0.46.pth` 已产生，训练已进入 e4。当前不要手动停止，不要 locked audit；等 natural finish/max epoch 写出 `train_summary.json`，确认 `best_model_path` 指向 e3 validation-selected checkpoint 后再冻结 audit。horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 01:12 CST TCNLSTM attitude 训练自然到 `max_epochs=5` 结束，`early_stopped=false`；e4 `valid_loss_epoch=0.4616226`、`best_valid_loss=0.4615005`、`valid_q=0.0412516`、`valid_p=0.0321305`、`valid_v=0.1503205`、`valid_omega=0.2379201`，未刷新 e3。`train_summary.json` 显示 `best_model_path` 指向 `/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`，`best_model_score=0.4615004957`。训练阶段 horizon/test metric 内容未读取，artifacts 保留。
- 2026-05-11 01:12 CST 已冻结 e3 checkpoint 并启动 locked audit：audit id `lockaudit_20260511_tcnlstm_attitude_H10_e3_p1`；tmux `audit_tcnlstm_attitude_H10_e3_p1`；日志 `resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/logs/eval_lockaudit_20260511_tcnlstm_attitude_H10_e3_p1_b32.log`。Eval 命令使用 `scripts/eval.py --dataset neurobemfullstate --predictor_type full_state --accelerator cuda --gpu_id 0 --eval_batch_size 32 --eval_horizons 1,10,25,50 --wandb_mode disabled --experiment_path resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1`；启动检查显示 tmux/process alive，horizon/test metric 内容尚未读取。若 batch `32` OOM 或崩溃，按协议用 batch `16` 重试同一冻结 checkpoint，并记录 `_b16.log`。
- 2026-05-11 01:25 CST locked audit 完成：batch `32`，无 OOM retry，average rollout loss `0.5288807750`；h1 `E_q=0.0014257`、`E_v=0.0100072`、`E_omega=0.0537533`；h10 `E_q=0.0174112`、`E_v=0.0865228`、`E_omega=0.2137843`；h25 `E_q=0.0445511`、`E_v=0.1861041`、`E_omega=0.2681405`；h50 `E_q=0.0907420`、`E_v=0.3453549`、`E_omega=0.3335756`；mean `E_q=0.0458700`、`E_v=0.1857792`、`E_omega=0.2566082`。结论：partial locked win，`h50_E_v` 仍优于 `gru_H20=0.353015` 和 `tcn_H10=0.347036`，但 h50 quaternion、h50 omega、mean quaternion 未击败 strongest GRU targets；不是全面 winner。horizon/test metric 内容仅在 locked audit 完成后读取；artifacts 全部保留，未清理；不从 audit 细节反向调参。
- 2026-05-11 01:27 CST 结果/结构 agent 交叉建议一致：下一步先做无代码改动的 `TCNLSTM attitude H20/F50` history extension，不先改模型代码。理由：`tcnlstm.py` 参数基本不依赖 `history_length`，forward 在实际输入序列长度上运行；H20 可提供更多 actuator lag / hidden dynamics 历史，但有跨 history mismatch 风险，必须严格 gate。remote smoke `smoke_20260511_tcnlstm_attitude_H20_from_H10e3_p1` 从 H10 e3 checkpoint 启动成功，H20 training/validation 数据窗口正常，一批 train/valid finite：`train_loss_step=0.169`、`valid_loss_epoch=0.115`，`train_summary.json` 和 checkpoint 正常生成。训练阶段未读取 horizon/test metric 内容。
- 2026-05-11 01:29 CST 启动 active training `modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1`：从 `modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth` 初始化，H20/F50，epochs `5`，batch `64`、accumulate `8`、effective batch `512`，`limit_train_batches=0.25`、`limit_val_batches=0.5`，`warmup_lr=2e-6`、`cosine_lr=5e-7`、`warmup_steps=50`、`cosine_steps=1000`，early stopping patience `2`、min_delta `2e-5`，WANDB disabled。tmux `modeldev_tcnlstm_attitude_H20_from_H10e3_p1`；GPU watch `modeldev_gpu_watch_tcnlstm_attitude_H20`；log `resources/experiments/modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1/logs/train_phase1.log`。启动巡检显示 tmux/process/GPU healthy，RTX 4060 约 `2950/8188 MiB`、util `31%`，epoch 0 train steps finite；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- H20 gate：e0 `>0.465` 直接判定跨 history mismatch，停止不 audit；`0.4625-0.465` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康；`<=0.4625` 算健康。冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.4615` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1506` 同步成立。任何 q/omega 连续两轮回退或 `valid_v > 0.1510`，即使 total loss 接近也停止。
- 2026-05-11 01:54 CST H20 e0 validation 完成：`train_loss_epoch=0.335`，`valid_loss_epoch=0.4630005`，`best_valid_loss=0.4630004`，`valid_q=0.0413083`，`valid_p=0.0323164`，`valid_v=0.1509458`，`valid_omega=0.2384300`；checkpoint `resources/experiments/modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth` 已产生，`last_model.pth` 同步更新，`train_summary.json` 尚未生成，训练已进入 e1。判定：e0 位于 `0.4625-0.465` 灰区，不是即时跨 history mismatch 失败，但也不是健康/冻结候选；q/omega/v 均未达到冻结同步线。下一步按预声明继续观察 e1/e2，不手动停止、不 audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 02:17 CST H20 e1 validation 完成并小幅刷新 best：`train_loss_epoch=0.3348033`，`valid_loss_epoch=0.4628834`，`best_valid_loss=0.4628835`，`valid_q=0.0413141`，`valid_p=0.0322572`，`valid_v=0.1507668`，`valid_omega=0.2385453`；checkpoint `resources/experiments/modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1/checkpoints/model-epoch=01-best_valid_loss=0.46.pth` 已产生，`last_model.pth` 同步更新，`train_summary.json` 尚未生成，训练已进入 e2。判定：e1 仍在灰区，不是冻结候选；`valid_v` 较 e0 改善且未触发 `>0.1510` 停止线，但 `valid_q/valid_omega` 较 e0 回退，若 e2 再次回退则按 H20 gate 停止不 audit。下一步继续观察 e2，不手动停止、不 audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 02:44 CST H20 e2 validation 完成并小幅刷新 best：`train_loss_epoch=0.3340761`，`valid_loss_epoch=0.4627466`，`best_valid_loss=0.4627467`，`valid_q=0.0413706`，`valid_p=0.0321600`，`valid_v=0.1504421`，`valid_omega=0.2387740`；checkpoint `resources/experiments/modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1/checkpoints/model-epoch=02-best_valid_loss=0.46.pth` 已产生，`last_model.pth` 同步更新，训练已进入 e3。判定：total loss 和 `valid_v` 继续改善，但未到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.4615` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1506` 的同步健康条件；`valid_q`、`valid_omega` 连续两轮回退，按 H20 gate 停止。执行结果：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_attitude_H20_from_H10e3_p1` 和 GPU watch `modeldev_gpu_watch_tcnlstm_attitude_H20`，artifacts 保留（checkpoints e0/e1/e2 和 `last_model.pth`），`train_summary.json` 未生成。下一步建议：基于 train/validation/架构信号，在 `scripts/dynamics_learning/models/tcnlstm.py` 评估 checkpoint-safe decoder-state residual initializer，不使用 H20 audit 或 horizon/test 信息反向调参。
- 2026-05-11 02:52 CST TCNLSTM decoder-state residual initializer 已实现：
  - 文件：`scripts/dynamics_learning/models/tcnlstm.py`。
  - 结构：在原 `state_initializer` 之后新增 `decoder_state_residual_norm`、`decoder_state_residual`、`decoder_state_residual_scale=0.05`。
  - 输入：`enc_seq` last/mean、`history_context`、以及用 `history_context+x_last` 查询的 attitude context。
  - 输出：LSTM decoder `h0/c0` residual，最后一层 zero-init，因此旧 checkpoint 初始为 no-op。
  - 保持不变：TCN-compatible anchor 路径 `encoder`、`decoder`、`base_feature = anchor_seq[:, -1, :]`；没有使用 H20 audit 或 horizon/test 信息调参。
- 检查和同步：
  - local `python -m py_compile scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - 本地 forward smoke 因 Mac Python 缺少 `torch` 未跑；已改用远程 `dynamics_learning` 环境验证。
  - local/remote `tcnlstm.py` hash：`55ac4a3f86de7256d70bbbc2ee55351a5f1fd9bb1efdb19b149ccc5f317b3bb1`。
  - remote py_compile 通过。
  - remote one-batch smoke `smoke_20260511_tcnlstm_stateinit_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 仅新增 `decoder_state_residual*`，无 unexpected keys；`train_loss_step=0.176`、`valid_loss_epoch=0.122`，checkpoint 和 `train_summary.json` 正常生成。
  - horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 02:53 CST 启动 active training `modeldev_20260511_tcnlstm_stateinit_H10_from_attitude_e3_p1`：
  - 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`。
  - 配置：H10/F50，epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=2e-6`，`cosine_lr=5e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
  - tmux：`modeldev_tcnlstm_stateinit_H10_from_attitude_e3_p1`。
  - GPU watch：`modeldev_gpu_watch_tcnlstm_stateinit_H10`。
  - log：`resources/experiments/modeldev_20260511_tcnlstm_stateinit_H10_from_attitude_e3_p1/logs/train_phase1.log`。
  - 启动巡检：tmux/process/GPU healthy，RTX 4060 约 `1974/8188 MiB`、util `31%`，train steps finite；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- stateinit H10 gate：
  - 参考 H10 attitude best `0.4615005`。
  - e0 `>0.4625` 判定破坏校准，停止不 audit。
  - e0 `0.4617-0.4625` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。
  - e0 `<=0.4615` 算健康。
  - 冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立。
  - locked audit 只能在 natural finish/summary 指向 validation-selected checkpoint 后启动。
- 2026-05-11 03:21 CST stateinit H10 e0 validation 完成：
  - `valid_loss_epoch=0.4617348`，`best_valid_loss=0.4617349`。
  - submetrics：`valid_q=0.0412232`，`valid_p=0.0321486`，`valid_v=0.1504455`，`valid_omega=0.2379177`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_stateinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成，训练已进入 e1；tmux/process/GPU watch 健康。
  - 判定：e0 位于预声明灰区 `0.4617-0.4625`，不触发 `>0.4625` 破坏校准停止线，但未达到健康线/冻结候选线；继续最多观察 e1/e2，看 total loss、`valid_q`、`valid_omega`、`valid_v` 是否同步健康。当前不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 03:40 CST stateinit H10 e1 validation 完成并刷新本实验 best：
  - `valid_loss_epoch=0.4615996`，`best_valid_loss=0.4615997`，`train_loss_epoch=0.332`。
  - submetrics：`valid_q=0.0412552`，`valid_p=0.0321055`，`valid_v=0.1502778`，`valid_omega=0.2379611`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_stateinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=01-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成，训练已进入 e2；tmux/process/GPU watch 健康。
  - 判定：e1 的 total loss 和 `valid_v` 较 e0 改善，但仍未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立；`valid_q/valid_omega` 较 e0 回退。当前继续观察 e2，不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 04:06 CST stateinit H10 e2 validation 完成但未刷新 e1 best：
  - `valid_loss_epoch=0.4617738`，`best_valid_loss=0.4615997`。
  - submetrics：`valid_q=0.0413049`，`valid_p=0.0320988`，`valid_v=0.1501462`，`valid_omega=0.2382236`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_stateinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=02-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新；`train_summary.json` 未生成。
  - 判定：e2 未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立；`valid_q/valid_omega` 从 e0 到 e2 连续回退，触发 stateinit H10 gate 失败。
  - 执行结果：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_stateinit_H10_from_attitude_e3_p1` 和 GPU watch `modeldev_gpu_watch_tcnlstm_stateinit_H10`，artifacts 保留（checkpoints e0/e1/e2 和 `last_model.pth`），无活跃训练进程。
  - 下一步建议：不要基于 horizon/test 内容调参；只基于 train/validation/架构信号讨论下一种 TCNLSTM/GRUTCN 结构，优先复盘 stateinit 对 q/omega 的不利影响，选择新的 checkpoint-safe 分支或回到 GRUTCN/TCNLSTM 架构候选。
- 2026-05-11 04:12 CST TCNLSTM cell-memory residual 已实现：
  - 文件：`scripts/dynamics_learning/models/tcnlstm.py`。
  - 结构：保留 `decoder_state_residual*` 参数形状，但 `_init_decoder_state()` 不再把 residual 同时加到 LSTM `h0/c0`；现在 `h0 = base_h`，只用 `residual_gate` 调制 `residual_c` 后加到 `c0`。
  - 动机：stateinit H10 的 e0-e2 显示 `valid_v` 受益但 `valid_q/valid_omega` 连续回退，说明扰动 `h0` 可能污染 attention query；cell-only 版本保留 Mohajerin-style decoder memory 初始化叙事，同时让 attention query 和 anchor 初始化更稳定。
  - 保持不变：TCN-compatible anchor 路径 `encoder`、`decoder`、`base_feature = anchor_seq[:, -1, :]`；没有使用 H20/stateinit audit 或 horizon/test 信息调参。
- 检查和同步：
  - local `python -m py_compile scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local `git diff --check -- scripts/dynamics_learning/models/tcnlstm.py Prompt.md MODEL_DEV_HANDOFF.md` 通过。
  - local/remote `tcnlstm.py` hash：`239d693579fa11d594d3bc9e163e69ed73bc6c25f1862bcace907b16656ddd84`。
  - remote py_compile 通过。
  - remote one-batch smoke `smoke_20260511_tcnlstm_cellinit_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 仅新增 `decoder_state_residual*`；一批 train/valid finite：`train_loss_step=0.169`、`valid_loss_epoch=0.118`，checkpoint 和 `train_summary.json` 正常生成。
  - horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 04:14 CST 启动 active training `modeldev_20260511_tcnlstm_cellinit_H10_from_attitude_e3_p1`：
  - 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`。
  - 配置：H10/F50，epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=2e-6`，`cosine_lr=5e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
  - tmux：`modeldev_tcnlstm_cellinit_H10_from_attitude_e3_p1`。
  - GPU watch：`modeldev_gpu_watch_tcnlstm_cellinit_H10`。
  - log：`resources/experiments/modeldev_20260511_tcnlstm_cellinit_H10_from_attitude_e3_p1/logs/train_phase1.log`。
  - 启动巡检：tmux/process/GPU healthy，RTX 4060 约 `1928/8188 MiB`、util `25%`，epoch 0 train steps finite；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- cellinit H10 gate：
  - 参考 H10 attitude best `0.4615005`。
  - e0 `>0.4622` 判定 cell residual 破坏校准，停止不 audit。
  - e0 `0.4616-0.4622` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。
  - e0 `<=0.46155` 算健康。
  - 冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立。
  - q/omega 连续两轮回退即 validation-only 失败，不读取 horizon/test metric 内容、不做 locked audit。
  - locked audit 只能在 natural finish/summary 指向 validation-selected checkpoint 后启动。
- 2026-05-11 04:41 CST cellinit H10 e0 validation 完成：
  - `train_loss_epoch=0.333`，`valid_loss_epoch=0.4617287`，`best_valid_loss=0.4617288`。
  - submetrics：`valid_q=0.0412230`，`valid_p=0.0321477`，`valid_v=0.1504408`，`valid_omega=0.2379174`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_cellinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成，训练已进入 e1；04:44 CST 巡检显示 tmux/process/GPU watch 健康，RTX 4060 约 `1968/8188 MiB`、util `34%`，e1 训练 steps finite。
  - 判定：e0 位于预声明灰区 `0.4616-0.4622`，不触发 `>0.4622` 停止线，但未达到 `<=0.46155` 健康线/冻结候选线；`valid_v` 已满足同步线，`valid_q/valid_omega` 尚未过门。下一步继续观察 e1/e2 的 total loss 和 q/omega/v 是否同步健康；当前不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 05:01 CST cellinit H10 e1 validation 完成并刷新本实验 best：
  - `valid_loss_epoch=0.4615999`，`best_valid_loss=0.4616000`。
  - submetrics：`valid_q=0.0412553`，`valid_p=0.0321055`，`valid_v=0.1502781`，`valid_omega=0.2379610`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_cellinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=01-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成，训练已进入 e2；05:03 CST 巡检显示 tmux/process/GPU watch 健康，RTX 4060 约 `1968/8188 MiB`、util `36%`，e2 训练 steps finite，CSVLogger 只有 e0/e1 validation rows。
  - 判定：e1 的 total loss 和 `valid_v` 较 e0 改善，但仍未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立；`valid_q/valid_omega` 较 e0 回退。下一步继续观察 e2，若 q/omega 再次回退则按 cellinit H10 gate 判为 validation-only 失败，停止本实验训练 tmux/GPU watch，不读取 horizon/test metric 内容、不做 locked audit；当前不冻结、不 audit，只确认 `plots/testset` 目录存在。
- 2026-05-11 05:25 CST cellinit H10 e2 validation 完成但未刷新 e1 best：
  - `valid_loss_epoch=0.4617757`，`best_valid_loss=0.4616000`。
  - submetrics：`valid_q=0.0413049`，`valid_p=0.0320990`，`valid_v=0.1501480`，`valid_omega=0.2382238`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_cellinit_H10_from_attitude_e3_p1/checkpoints/model-epoch=02-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新；`train_summary.json` 未生成。
  - 判定：e2 未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立；`valid_q/valid_omega` 从 e0 到 e2 连续回退，触发 cellinit H10 validation-only failure。
  - 执行结果：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_cellinit_H10_from_attitude_e3_p1`、匹配该实验路径的训练进程和 GPU watch `modeldev_gpu_watch_tcnlstm_cellinit_H10`；artifacts 保留（checkpoints e0/e1/e2 和 `last_model.pth`）。
  - 下一步建议：不要基于 horizon/test 内容调参；只基于 train/validation/架构信号讨论下一种 TCNLSTM/GRUTCN 结构。stateinit 和 cellinit 都呈现 `valid_v` 改善但 `valid_q/valid_omega` 连续回退，下一步应优先考虑更局部的 attitude/omega path、门控 residual 的训练幅度/位置，或回到 GRUTCN/TCNLSTM 的其他 checkpoint-safe 分支。
- 2026-05-11 05:34 CST TCNLSTM local attitude fine residual 已实现：
  - 文件：`scripts/dynamics_learning/models/tcnlstm.py`。
  - 结构：保留 `decoder_state_residual*` 模块以兼容近期 checkpoint，但 `_init_decoder_state()` 不再把 residual 加到 LSTM `h0/c0`；新增 `attitude_fine_norm`、`attitude_fine_gate`、`attitude_fine_decoder`、`attitude_fine_delta_scale=0.02`，只在输出端为 `[:, 6:12]` 的 `dtheta/delta_omega` 生成小 residual。
  - 输入：已有 `attitude_input` 加上 TCN anchor 的 `base_delta[:, 6:12]`，让细分支只看姿态/角速度局部误差方向，不再扰动 decoder hidden/cell memory。
  - 初始化：`attitude_fine_decoder` final projection zero-init，因此从 H10 attitude e3 checkpoint 加载时初始为 no-op；`encoder`、`decoder`、`base_feature = anchor_seq[:, -1, :]` anchor 路径不变。
  - 动机：stateinit/cellinit 都显示 total loss/`valid_v` 可改善，但 `valid_q/valid_omega` 连续回退；因此把表达力移动到更局部的 attitude/omega output path，而不是改 decoder state。
  - 未使用外部 trick、web research 或 horizon/test 信息。
- 检查和同步：
  - local `python -m py_compile scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local `git diff --check -- scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local/remote `tcnlstm.py` hash：`be9793501f16aab8aad4e45c0e72cf78462c42eb124e4fce23c59b8b822f2df5`。
  - remote py_compile 通过。
  - remote one-batch smoke `smoke_20260511_tcnlstm_attfine_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 仅新增/保留 `decoder_state_residual*` 与 `attitude_fine*`，无 unexpected keys；`train_loss_step=0.181`、`valid_loss_epoch=0.118`，checkpoint 和 `train_summary.json` 正常生成。
  - horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 05:37 CST 启动 active training `modeldev_20260511_tcnlstm_attfine_H10_from_attitude_e3_p1`：
  - 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`。
  - 配置：H10/F50，epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=2e-6`，`cosine_lr=5e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
  - tmux：`modeldev_tcnlstm_attfine_H10_from_attitude_e3_p1`。
  - GPU watch：`modeldev_gpu_watch_tcnlstm_attfine_H10`。
  - log：`resources/experiments/modeldev_20260511_tcnlstm_attfine_H10_from_attitude_e3_p1/logs/train_phase1.log`。
  - 启动巡检：tmux/process/GPU healthy，RTX 4060 约 `1978/8188 MiB`、util `24%`，e0 train steps finite；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- attfine H10 gate：
  - 参考 H10 attitude best `0.4615005`。
  - e0 `>0.4622` 判定 fine residual 破坏校准，停止不 audit。
  - e0 `0.4616-0.4622` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。
  - e0 `<=0.46155` 算健康。
  - 冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立。
  - q/omega 连续两轮回退即 validation-only 失败；不读取 horizon/test metric 内容、不做 locked audit。
  - locked audit 只能在 natural finish/`train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 2026-05-11 06:03 CST attfine H10 e0 validation 完成并触发 gate：
  - `train_loss_epoch=0.3329748`，`valid_loss_epoch=0.4623119`，`best_valid_loss=0.4623121`。
  - submetrics：`valid_q=0.0412572`，`valid_p=0.0322411`，`valid_v=0.1507487`，`valid_omega=0.2380650`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_attfine_H10_from_attitude_e3_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新；`train_summary.json` 未生成。
  - 判定：e0 `0.4623119 > 0.4622`，local attitude fine residual 破坏校准，validation-only failure。
  - 执行结果：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_attfine_H10_from_attitude_e3_p1`、匹配该实验路径的训练进程和 GPU watch `modeldev_gpu_watch_tcnlstm_attfine_H10`；artifacts 保留（e0 checkpoint、`last_model.pth`、CSV/logs）。
  - 下一步建议：不要继续沿输出端 tiny attitude residual 堆叠；它未解决 q/omega 方向且总 loss 直接过 stop line。下一步应只基于 train/validation/架构信号，重新评估 TCNLSTM 的更强结构候选或回到 GRUTCN，优先让结果分析 agent 与架构 agent 交叉提出方案。
- 2026-05-11 06:12 CST 结果/架构 agent 交叉建议后的 TCNLSTM dual-view long-history translational residual 已实现：
  - 文件：`scripts/dynamics_learning/models/tcnlstm.py`。
  - 动机：H20/stateinit/cellinit/attfine 的共同模式是 total loss 或 `valid_v` 偶有改善，但 `valid_q/valid_omega` 容易退化；因此不再扰动 decoder state，也不再直接写姿态输出，而是让 full H20 只作为非姿态慢变量旁路。
  - 结构：新增 `anchor_history_len=min(history_len,10)`，`encoder`、`decoder`、`base_feature=anchor_seq[:, -1, :]` 只使用最后 10 帧，保持 H10 attitude e3 checkpoint 的 TCN anchor 校准；新增 `long_history_lstm`、`long_history_norm`、`long_delta_norm`、`long_delta_gate`、`long_delta_decoder`、`long_delta_scale=0.03`，full H20 只写 `long_delta[:, :6]`。
  - 初始化：`long_delta_decoder` final projection zero-init，因此从 H10 attitude e3 checkpoint 加载时初始为 no-op；保留 `decoder_state_residual*` 和 `attitude_fine*` 参数以兼容近期 checkpoint，但 forward 不再应用失败的 `attitude_fine` residual。
  - 未使用外部 trick、web research 或 horizon/test 信息。
- 检查和同步：
  - local `python -m py_compile scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local `git diff --check -- scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local/remote `tcnlstm.py` hash：`55717404b8786d64fcc5e96c8b742a30ee8b869a60e25d47a2a11df8e37581b6`。
  - remote py_compile 通过。
  - remote one-batch smoke `smoke_20260511_tcnlstm_dualview_longtrans_H20_from_attitude_e3_p1` 从 H10 attitude e3 checkpoint 加载，missing keys 仅新增/保留 `decoder_state_residual*`、`attitude_fine*`、`long_history*`、`long_delta*`，无 unexpected keys；`train_loss_step=0.178`、`valid_loss_epoch=0.115`，checkpoint 和 `train_summary.json` 正常生成。
  - horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 06:14 CST 启动 active training `modeldev_20260511_tcnlstm_dualview_longtrans_H20_from_attitude_e3_p1`：
  - 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`。
  - 配置：H20/F50，epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=2e-6`，`cosine_lr=5e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
  - tmux：`modeldev_tcnlstm_dualview_longtrans_H20_p1`。
  - GPU watch：`modeldev_gpu_watch_tcnlstm_dualview_H20`。
  - log：`resources/experiments/modeldev_20260511_tcnlstm_dualview_longtrans_H20_from_attitude_e3_p1/logs/train_phase1.log`。
  - 启动巡检：tmux/process alive；GPU watch 修正为 `/usr/lib/wsl/lib/nvidia-smi` 后可记录显存/利用率，约 `2442/8188 MiB`、util `30%`；train steps finite；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- dualview longtrans H20 gate：
  - 参考 H10 attitude best `0.4615005`。
  - e0 `>0.4622` 判定长历史旁路破坏校准，停止不 audit。
  - e0 `0.4616-0.4622` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。
  - e0 `<=0.46155` 算健康。
  - 冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1505` 同步成立。
  - q/omega 连续两轮回退即 validation-only 失败，不读取 horizon/test metric 内容、不做 locked audit。
  - locked audit 只能在 natural finish/`train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 2026-05-11 06:43 CST e0 validation 完成并触发 stop line：
  - `train_loss_epoch=0.3349663`，`valid_loss_epoch=0.4628875`，`best_valid_loss=0.4628872`。
  - submetrics：`valid_q=0.0413091`，`valid_p=0.0322539`，`valid_v=0.1509052`，`valid_omega=0.2384190`。
  - checkpoint：`checkpoints/model-epoch=00-best_valid_loss=0.46.pth` 和 `last_model.pth` 已产生；`train_summary.json` 未生成。
  - 判定：e0 `0.4628875 > 0.4622`，长历史 translational bypass 破坏 H10 attitude anchor 校准，validation-only failure。
  - 执行：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_dualview_longtrans_H20_p1`、匹配该实验路径的训练进程和 GPU watch `modeldev_gpu_watch_tcnlstm_dualview_H20`；artifacts 保留。
  - 下一步建议：只基于 train/validation/架构信号讨论 TCNLSTM/GRUTCN 结构，优先复盘 H20/history/long-branch 与 q/omega 校准的冲突；不要用任何 horizon/test 内容反向调参。
- 2026-05-11 06:52 CST 选择并实现下一候选 `TCNLSTM H10 Lag-Observer Context`：
  - 文件：`scripts/dynamics_learning/models/tcnlstm.py`。
  - 动机：H20/长历史、stateinit/cellinit、attfine、dualview 的共同模式是 total loss 抬高或 `valid_q/valid_omega` 退化；因此不再继续堆 output tiny adapter，也不触碰 LSTM `h0/c0`。
  - 结构：保留 H10 attitude e3 的 TCN anchor 路径 `encoder`、`decoder`、`base_feature=anchor_seq[:, -1, :]`；failed `long_delta` 不再加到输出；新增 H10 内 `lag_observer`，输入 `[x_anchor, dx_anchor]`，提取 actuator lag / aero lag / filter-state latent context，只通过 zero-init `lag_context_proj` 注入 `head_input`，不直接写 `y`。
  - 初始化：`lag_context_proj` final projection zero-init，`lag_context_scale=0.05`；从 H10 attitude e3 checkpoint 初始化时初始为 no-op。
  - 未使用外部 web research、trick 行为变化或 horizon/test 信息；决策来自 train/validation/架构信号和 agent 交叉建议。
- 检查和同步：
  - local `python -m py_compile scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local `git diff --check -- scripts/dynamics_learning/models/tcnlstm.py` 通过。
  - local/remote `tcnlstm.py` hash：`d5857bca84d7535e1022f14f999cd9630ea09b9a30dbe37e7580a94ce9ea6790`。
  - remote py_compile 通过。
  - remote one-batch smoke `smoke_20260511_tcnlstm_lagobserver_H10_from_attitudee3_p1` 从 H10 attitude e3 checkpoint 加载成功，missing keys 为新增/保留的 `lag_*`、`decoder_state_residual*`、`attitude_fine*`、`long_history*`、`long_delta*`，无运行错误；一批 train/valid finite：`train_loss_step=0.168`、`valid_loss_epoch=0.124`。
  - horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 06:55 CST 启动 active training `modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1`：
  - 初始化 checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`。
  - 配置：H10/F50，epochs `5`，batch `64`，accumulate `8`，effective batch `512`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=1.5e-6`，`cosine_lr=4e-7`，`warmup_steps=50`，`cosine_steps=1000`，early stopping patience `2`，min_delta `2e-5`，WANDB disabled。
  - tmux：`modeldev_tcnlstm_lagobserver_H10_from_attitude_e3_p1`。
  - GPU watch：`modeldev_gpu_watch_tcnlstm_lagobserver_H10`。
  - log：`resources/experiments/modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1/logs/train_phase1.log`。
  - 启动巡检：tmux/process/GPU watch healthy，GPU 约 `2134/8188 MiB`、util `40%`；e0 train steps finite；`train_summary.json` 尚未生成；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- lagobserver H10 gate：
  - 参考 H10 attitude best `0.4615005`。
  - e0 `>0.4620` 判定 lag observer 破坏校准，停止不 audit。
  - e0 `0.4616-0.4620` 最多给 e1/e2，看 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。
  - e0 `<=0.46155` 算健康。
  - 冻结候选优先要求 `best_valid_loss <= 0.4613`；或 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1503` 同步成立。
  - q/omega 连续两轮回退即 validation-only failure，不读取 horizon/test metric 内容、不做 locked audit。
  - locked audit 只能在 natural finish/`train_summary.json` 指向 validation-selected best checkpoint 后启动。
- 2026-05-11 07:23 CST lagobserver H10 e0 validation 完成并进入 e1：
  - `train_loss_epoch=0.3326606`，`valid_loss_epoch=0.4618268`，`best_valid_loss=0.4618267`。
  - submetrics：`valid_q=0.0412433`，`valid_p=0.0321268`，`valid_v=0.1503970`，`valid_omega=0.2380596`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1/checkpoints/model-epoch=00-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成；tmux/process/GPU watch 健康，e1 train steps finite。
  - 判定：e0 位于预声明灰区 `0.4616-0.4620`，不触发 `>0.4620` 停止线，但未达到 `<=0.46155` 健康线/冻结候选线；继续观察 e1/e2 的 total loss 和 `valid_q`、`valid_omega`、`valid_v` 是否同步健康。当前不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 07:49 CST lagobserver H10 e1 validation 完成并进入 e2：
  - `train_loss_epoch=0.3322823`，`valid_loss_epoch=0.4615656`，`best_valid_loss=0.4615656`。
  - submetrics：`valid_q=0.0412526`，`valid_p=0.0321126`，`valid_v=0.1501766`，`valid_omega=0.2380238`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1/checkpoints/model-epoch=01-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新。
  - `train_summary.json` 尚未生成；tmux/process/GPU watch 健康，e2 train steps finite。
  - 判定：e1 刷新 best，total loss、`valid_v`、`valid_omega` 较 e0 改善，但仍未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 q/omega/v 同步健康；`valid_q` 较 e0 回退。继续观察 e2，若 e2 未达到冻结线或 q/omega 形成连续回退，则 validation-only failure；当前不冻结、不 locked audit；horizon/test metric 内容未读取，只确认 `plots/testset` 目录存在。
- 2026-05-11 08:16 CST lagobserver H10 e2 validation 完成并触发 validation-only failure：
  - `train_loss_epoch=0.3323109`，`valid_loss_epoch=0.4614547`，`best_valid_loss=0.4614547`。
  - submetrics：`valid_q=0.0412437`，`valid_p=0.0320399`，`valid_v=0.1501071`，`valid_omega=0.2380640`。
  - checkpoint：`resources/experiments/modeldev_20260511_tcnlstm_lagobserver_H10_from_attitude_e3_p1/checkpoints/model-epoch=02-best_valid_loss=0.46.pth`，`last_model.pth` 同步更新；`train_summary.json` 未生成。
  - 判定：e2 total loss 与 `valid_v` 较 e1 改善，但未达到冻结候选线 `best_valid_loss <= 0.4613`，也不满足 `<=0.46145` 且 `valid_q < 0.04118`、`valid_omega < 0.23746`、`valid_v <= 0.1503` 同步成立；`valid_q/valid_omega` 仍未过门。按预声明 e2 未改善到冻结线处理为 validation-only failure。
  - 执行：不做 locked audit，不读取 horizon/test metric 内容；已仅停止训练 tmux `modeldev_tcnlstm_lagobserver_H10_from_attitude_e3_p1`、匹配该实验路径的训练进程和 GPU watch `modeldev_gpu_watch_tcnlstm_lagobserver_H10`。artifacts 保留（checkpoints e0/e1/e2 和 `last_model.pth`）。
- TCNLSTM true-anchor gates：e0 若 `>0.58` 直接停；e0 `<=0.55` 视为健康；e2 需低于 TCN H10 validation baseline `0.533292`，理想 `<=0.525` 才继续；只有 validation 大幅改善并呈现 submetrics 同步改善时才考虑冻结 audit。

## 8. 当前 GRUTCN dual/raw 结构

改动文件：

- `scripts/dynamics_learning/models/grutcn.py`

结构变化：

- 保留 checkpoint-safe temporal refiner：`LayerNorm -> MultiheadAttention -> LayerNorm -> FFN`。
- `_encode()` 在 `encoder_norm(enc_seq)` 之后、`context_gru(enc_seq)` 之前使用 refiner。
- 新增 dual latent context：把 attention `context`、`history_context`、`enc_seq` last/mean 融合成第二 context residual，加在 `decoder_input` 前。
- 新增 raw history residual fusion：从完整 raw `[x,u]` history 汇总 latent residual，加到 `head_input`。
- temporal refiner、dual context、raw history residual 的输出层均 zero-init，因此旧 checkpoint 初始保持校准行为，同时允许新分支学习更强 temporal/raw-history context。

设计动机：

- temporal refiner validation-only 失败后，不继续堆 tiny output adapter。
- 当前 locked audit 失败说明单纯 validation 降低不够，模型需要更强的 latent context / recurrent state/raw-history 表达力。
- dual context 在 decoder attention 后增加第二 latent context，raw-history residual 让完整 `[x,u]` history 可直接补充 head latent，符合 hidden actuator/aero/disturbance/filter-state 叙事。

检查和同步：

- local `python -m py_compile scripts/dynamics_learning/models/grutcn.py` 通过。
- local `git diff --check -- scripts/dynamics_learning/models/grutcn.py Prompt.md MODEL_DEV_HANDOFF.md` 通过。
- 只同步 `scripts/dynamics_learning/models/grutcn.py` 到 `gpu4060`，未使用 `--delete`。
- local/remote `grutcn.py` hash：`8c5ae1489fa1961df585f15789e64932cc9b9f2e10e6a2a25d163a76cf7a7f92`。
- remote py_compile 通过。
- remote one-batch CUDA smoke 通过：从 ultralow e4 checkpoint 加载，missing keys 只包含新增 temporal/dual/raw 模块，无 unexpected keys；`train_loss_step=0.150`，`valid_loss_epoch=0.0807`。

## 9. 当前监控门槛

训练阶段：

- 可以读取 horizon/test metric 内容，并可用于候选挑选、结构判断和后续模型改进。
- 每次读取都要记录实验 id、checkpoint、关键 horizon/test 指标、是否用于后续调参/结构决策，以及结论。
- 仍需同时检查 train/validation/log/checkpoint/smoke/架构分析，避免只追单一 horizon 指标。

dual/raw gates：

- e0 `>0.485`：停止，无 audit。
- e0 `0.4732-0.485`：最多给 e1/e2。
- e0 `<=0.4732`：健康。
- e2 `>0.4725`：停止，除非它是 `0.47250-0.47255` 内的新低，可预声明给 e3 确认。
- locked audit 优先只给 `best_valid_loss <= 0.4720`。
- 理想 audit gate：`best_valid_loss <= 0.4718`。
- audit 前 validation submetrics 不应回退，参考线：
  - `valid_v <= 0.1605`
  - `valid_omega <= 0.2294`
  - `valid_q <= 0.0390`

如果触发停止：

- 先记录 validation-only 失败原因。
- 确认 exact experiment id、tmux、process、path。
- 只停止该 tmux/process。
- 默认保留 artifacts，除非用户明确要求清理该失败实验产物。
- 是否继续跑 audit 或 horizon/test evaluation 由当前策略决定；如果已读 horizon/test metric，必须记录读取内容和对决策的影响。

## 10. 最近关键实验脉络

### 10.1 `modeldev_20260510_grutcn_anchor_e7_ultralow_p2`

定位：

- 当前最强 validation checkpoint 的来源。
- pure stable anchor GRUTCN H20。

结果：

- best validation：e4 `valid_loss=0.4720815`，`best_valid_loss=0.4720816`。
- checkpoint：`resources/experiments/modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。

locked audit：

- audit id：`lockaudit_20260510_grutcn_anchor_e7_ultralow_p2`。
- average rollout loss：`0.5632579327`。
- h50 `E_q=0.0853667619`、`E_v=0.3810590338`、`E_omega=0.3012506581`。
- mean `E_q=0.0437535938`。

结论：

- validation 很强，但不是 locked winner。
- 未击败 `gru_H10 h50_E_q=0.0800042`。
- 未击败 `gru_H20 h50_E_v=0.353015`。
- 未击败 `gru_H20 h50_E_omega=0.260392`。
- 未击败 `gru_H20 mean_E_q=0.0420377`。
- 保留 checkpoint。
- 不从 audit 细节反向调参。

### 10.2 H50 / H10 跨 history 尝试

`modeldev_20260510_grutcn_anchor_H50_fromH20_ultralow_p1`：

- 目的：测试 H20 checkpoint 跨到 H50 是否改善 long-horizon behavior。
- e0/e1 validation：`0.9261976` -> `0.8817936`。
- 结论：明显跨 history mismatch，停止，无 audit。

`modeldev_20260510_grutcn_anchor_H10_fromH20_ultralow_p1`：

- 目的：测试 H20 checkpoint 跨到 H10。
- e0/e1 validation：`0.6045368` -> `0.5921974`。
- 结论：仍远高于 `0.50` sprint gate，停止，无 audit。

### 10.3 `modeldev_20260510_grutcn_no_output_gain_H20_from_ultralow_p1`

代码变化：

- 删除 GRUTCN 整体输出 `output_gain`。
- 目的：避免强 TCN `base_delta` anchor 被全局重缩放。

结果：

- e0 validation：`1.3612511`。

结论：

- 删除 `output_gain` 破坏 checkpoint 输出尺度。
- 立即停止，无 audit。
- 后续恢复 `output_gain`。

### 10.4 `modeldev_20260510_grutcn_refreshboost_H20_from_ultralow_p1`

代码变化：

- 恢复 `output_gain`。
- 添加 `memory_refresh_boost=1.15`。
- 让 `_blend_memory` 更偏向当前窗口隐状态，减少旧 decoder memory 在 rollout 中的拖累。

结果：

- e0/e1/e2 validation：`0.4836859` -> `0.4813507` -> `0.4818282`。

结论：

- 方向略改善，但未恢复到 `<=0.4725` audit screening band。
- 停止，无 audit。
- 未读取 horizon/test metric 内容。

### 10.5 `modeldev_20260510_grutcn_anchor_noise_H20_from_ultralow_p1`

trick 来源：

- `/Users/lixiang/Documents/Obsidian Vault/trick/trick.md` 的 physical/noise robustness 思路。

代码/协议变化：

- `scripts/config.py` 增加默认关闭 `input_noise_std` 和 `input_noise_loss_weight`。
- `scripts/dynamics_learning/lighting.py` 增加 train-only input noise rollout objective。
- 记录 `train_noisy_loss` 和 `train_objective`。
- validation/test/eval 保持 clean。

结果：

- e5 `valid_loss=0.4722001`，`best_valid_loss=0.4722003`。
- checkpoint：`resources/experiments/modeldev_20260510_grutcn_anchor_noise_H20_from_ultralow_p1/checkpoints/model-epoch=05-best_valid_loss=0.47.pth`。

locked audit：

- audit id：`lockaudit_20260510_grutcn_anchor_noise_H20_e5_p1`。
- average rollout loss：`0.5636401176`。
- h1 `E_q=0.0015411`、`E_v=0.0135520`、`E_omega=0.0658329`。
- h10 `E_q=0.0177362`、`E_v=0.1057881`、`E_omega=0.2108654`。
- h25 `E_q=0.0425928`、`E_v=0.2209661`、`E_omega=0.2514102`。
- h50 `E_q=0.0854911`、`E_v=0.3807892`、`E_omega=0.3017333`。
- mean `E_q=0.0438164`、`E_v=0.2160809`、`E_omega=0.2430137`。

结论：

- 诚实 locked audit 失败。
- 保留 artifacts。
- 不从 audit 细节反向调参。

### 10.6 `modeldev_20260510_grutcn_feedbacktail_H20_from_noisee5_p1`

代码/协议变化：

- `scripts/config.py` 增加默认关闭 `feedback_noise_std`、`rollout_loss_tail_weight`。
- `scripts/dynamics_learning/lighting.py` 增加 quaternion-safe state noise、train-only feedback noise、可选 normalized linear tail weighting。
- existing input-noise 扰动后会重新归一化 quaternion history。
- validation/test/eval 保持 clean。

初始化：

- physical-noise e5 checkpoint。

结果：

- e0/e1/e2 validation：`0.4727978` -> `0.4730892` -> `0.4725238`。

结论：

- e2 是新低，但仍高于预声明 `<=0.4725` audit screening band 约 `2.4e-5`。
- 按门槛停止，无 audit。
- tmux/process 已停止。
- artifacts 保留。
- 未读取 horizon/test metric 内容。

后续逻辑：

- 不要为了 `1e-4` 级 validation-only gain 过度 audit。
- 转向增强 encoder-side temporal context 表达力，因此启动 temporal refiner。

### 10.7 `modeldev_20260510_grutcn_temporalrefine_H20_from_ultralow_e4_p1`

代码变化：

- `scripts/dynamics_learning/models/grutcn.py` 新增 checkpoint-safe encoder-side temporal refiner。
- 结构为 `LayerNorm -> MultiheadAttention -> LayerNorm -> FFN`。
- zero-init output residual 接在 `encoder_norm(enc_seq)` 后、`context_gru(enc_seq)` 前。

初始化：

- `modeldev_20260510_grutcn_anchor_e7_ultralow_p2/checkpoints/model-epoch=04-best_valid_loss=0.47.pth`。

结果：

- e0/e1/e2/e3 validation：`0.4727792` -> `0.4732321` -> `0.4731706` -> `0.4729020`。
- best `0.4727792442`，未进入 `<=0.4725` screening band。

结论：

- natural early stop，无 locked audit。
- tmux/process 已退出，artifacts 保留。
- 未读取 horizon/test metric 内容。
- temporal refiner 没带来 validation 改善，因此转向 decoder-side dual latent context + raw history residual，继续保持 zero-init/checkpoint calibration。

## 11. 较早模型开发脉络

### TCNLSTM 系列

`modeldev_20260508_tcnlstm_gate_H20_p1`：

- validation gate 失败并清理。
- 不恢复。

`modeldev_20260509_tcnlstm_context_residual_H20_p1`：

- best `0.6649035` at epoch 12。
- epoch 16 `valid_loss=0.6828123`、`valid_v=0.2939221`。
- validation-only 失败并清理。
- 不恢复。

结论：

- 当前 sprint 暂时以 GRUTCN 为主。
- TCNLSTM 不是永久放弃；只有在有更强结构故事时再继续。

### GRUTCN anchored context residual

`modeldev_20260509_grutcn_anchored_context_residual_H20_p1`：

- natural early stop。
- best e22 `best_valid_loss=0.6142662`。
- locked audit average rollout loss `0.5747820139`。
- h50 `E_q=0.08536557`、`E_v=0.40000114`、`E_omega=0.29806767`。

结论：

- 比 TCN H10 部分 quaternion 指标好。
- 未击败最强 GRU targets。
- 保留 checkpoint。
- 不标记 winner。

### GRUTCN low-LR / SWA anchor

`modeldev_20260509_grutcn_anchor_H20_e10_lowlr_swaft_p1`：

- best validation e7 `0.4776078`。
- locked audit average rollout loss `0.5678076148`。
- h50 `E_q=0.0856633007`、`E_v=0.3885666346`、`E_omega=0.3016224673`。

结论：

- validation 很强，但 locked audit 失败。
- 后续 tiny/ultralow LR 继续降低 validation，但仍没有 locked winner。

## 12. 已知代码/协议变更

训练/评估基础设施：

- `scripts/dynamics_learning/lighting.py` 支持 non-finite validation 传播到 `best_valid_loss`，避免 NaN run 逃过 early stopping。
- `scripts/run_neurobem_sweep.sh` 避免从 non-finite log 对应的 `last_model.pth` 恢复；rerun 日志不覆盖旧日志。
- `scripts/sync_to_gpu.sh` 默认排除 `resources/experiments/`，避免 Mac-to-remote sync 覆盖远程实验结果。
- `scripts/config.py` / `scripts/train.py` 支持 `--init_from_checkpoint`。
- `--init_from_checkpoint` 以 `strict=False` 加载模型权重，不恢复 optimizer/scheduler/callback state，并打印 missing/unexpected keys。

trick/训练目标：

- SWA trick 用过，来源 `trick.md` sections 3.3/3.4，目的是平滑 late-training 参数。
- input-noise 已加入为默认关闭选项，train-only。
- feedback-noise 已加入为默认关闭选项，train-only。
- tail-weighted rollout loss 已加入为默认关闭选项。
- validation/test/eval 必须保持 clean，除非明确记录协议变化。

GRUTCN 模型状态：

- 当前 active 版本包含 temporal refiner + dual latent context + raw history residual fusion。
- 不能直接回到 no-output-gain 版本，因为它已证明破坏 checkpoint scale。
- memory_refresh_boost 分支已停止，不是当前 active 代码目标。

## 13. 当前开放问题

- TCNLSTM true-anchor 已拿到 partial locked win：`h50_E_v=0.3444223`，优于 `gru_H20=0.353015` 和 `tcn_H10=0.347036`；但不是全面 winner。
- 下一阶段重点是保持 TCNLSTM 的 velocity 优势，同时改善 h50 quaternion、h50 omega 和 mean quaternion。
- GRUTCN validation 曾被压到 `0.4720816` 附近，但 locked h50 quaternion/velocity/omega 仍不够。
- temporal refiner 已验证：best 只到 `0.4727792442`，未进入 `<=0.4725` screening band，不 audit。
- GRUTCN dual/raw residual 已确认不是冻结候选；优先沿 TCNLSTM true-anchor 方向做下一轮结构改进或训练协议筛选。
- 下一步应让结果分析 agent + 架构 agent 判断 TCNLSTM 后续结构。
- 避免继续堆 tiny adapter；除非有清晰物理/结构故事和验证门槛。

## 14. 下一步执行建议

第一步：确认当前无 active training 后，基于 train/validation/horizon/test/架构信号讨论下一候选。

当前关键状态：`modeldev_20260511_grutcn_motiondiff_H20_from_anchor_e4_p1` 已 validation-only 失败并停止。e2 虽刷新本实验 best 到 `0.4723428`，但仍高于冻结线，且 `valid_v=0.1606968`、`valid_p=0.0435035` 未满足同步健康线；不冻结、不 audit，horizon/test metric 内容未读取。

自动巡检节奏：2026-05-11 10:15 CST 起按用户要求改为每 20 分钟一次；状态变化、失败停止或 locked audit 仍需即时记录到 `Prompt.md` 与本文件。

巡检/分析范围：

- 若下一次唤醒仍无 active training，先只读确认远程没有遗留训练进程/GPU 占用，再推进下一结构候选。
- 下一候选可以基于 train/validation/horizon/test/架构信号；允许读取 horizon/test metric 做挑选和改模型。
- 普通巡检频率按总控 automation 保持每 30 分钟；若启动新训练，更新本文件和 automation 到新的 active experiment。

第二步：不要对 motiondiff 做 locked audit，转入下一结构候选讨论。

motiondiff 失败要点：

- e2 best `0.4723428` 只小幅优于 e0，但仍弱于 anchor e4 best `0.4720816`、anchor-noise e5 `0.4722003`、dual/raw e2 `0.4721887`。
- `valid_q` / `valid_omega` 维持健康，但 `valid_p` / `valid_v` 未同步过门，说明 motion-diff encoder-side fusion 仍没有把 GRUTCN 的 locked 候选 validation band 推开。
- 该实验已停止，未做 locked audit；下一步不要围绕这个失败实验重试微小 LR/scale，而应基于 train/validation/架构信号考虑更结构性的 checkpoint-safe 候选。

第三步：不要回头继续 grubridge H10。

原因：

- grubridge e0 `valid_loss_epoch=0.4620850` / `best_valid_loss=0.4620851`，超过预声明 stop line `0.4620`。
- q/v/omega 也没有同步健康：`valid_q=0.0412587`、`valid_v=0.1505874`、`valid_omega=0.2379481`。
- 该实验没有形成冻结候选，未做 locked audit，horizon/test metric 内容未读取，artifacts 已保留。
- 2026-05-11 09:03 CST 已停止该实验训练 tmux、匹配实验路径的训练进程和 GPU watch；只清理了对应失败实验运行态，没有删除产物。

第四步：新状态变化写入 `Prompt.md`。

只记录长期有用信息：

- validation 新 best 或失败门槛。
- 停止/继续/audit 决策。
- 如果 audit，记录 audit id、checkpoint、batch、horizon 设置和 pass/fail。
- 如果代码变更，记录文件、结构故事、检查/同步/hash、smoke 结果。

## 15. 新 agent 不要做的事

- 不要隐式读取或使用 horizon/test metric；凡读取都必须在 `Prompt.md` / 本文件记录实验 id、checkpoint、关键指标和决策影响。
- 不要只因为 validation 接近就下结论；应结合 horizon/test、validation submetrics 和架构分析做判断。
- 不要删除 artifacts，除非实验已确认失败且用户允许清理对应失败实验产物。
- 不要用 `git reset --hard` 或 checkout 回滚用户改动。
- 不要同步覆盖远程 `resources/experiments/`。
- 不要把普通 20 分钟巡检流水写进 `Prompt.md`。
- 不要把文档写回英文。

## 16. 写入模板

把下面模板用于 `Prompt.md`，但只在有状态变化时写：

### <experiment_id> - <YYYY-MM-DD HH:mm CST>

- 类型：training / locked audit / code change / cleanup / decision。
- 当前状态：
- 模型与配置：
- 初始化 checkpoint：
- 最新 validation：
- 当前 best checkpoint：
- 决策：
- 是否读取 horizon/test metric 内容：
- 文件改动/同步/hash/smoke：
- artifacts 处理：
