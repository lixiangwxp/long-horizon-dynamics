# 当前模型冲刺短状态

最后更新：2026-05-16 21:16 CST。

用途：同一聊天窗口和 heartbeat 自动化优先读取本文件，避免反复完整读取 `Prompt.md` / `MODEL_DEV_HANDOFF.md` 造成上下文膨胀。只有新聊天、上下文压缩后状态不明、当前状态冲突、或需要历史复盘时，才读取完整交接文档。

## 当前 Active

- 状态：active none。
- 当前阶段：post-run decision pending。
- 实验 id：`modeldev_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1`
- 代码 base：`main@818e18990b8bf1aee4e493d238d1d0f912936652`。本地 repo 启动前 clean；远端原 repo 仍保留旧 dirty 现场，因此本次正式训练使用 clean worktree `/home/ubuntu/Developer/long-horizon-dynamics_run_818e189`，该 worktree HEAD 为 `818e18990b8bf1aee4e493d238d1d0f912936652` 且 `git status --short` clean。`resources` 通过本地 git exclude 的 symlink 指向原资源目录，不改变源码。
- 远程连接：当前通过 LAN `ubuntu@192.168.1.108` 可稳定执行 SSH 命令；Tailscale/WSL/SSH 异常只作为背景诊断，不再阻塞本次训练。
- 远程代码 worktree：`/home/ubuntu/Developer/long-horizon-dynamics_run_818e189`
- 实验路径：`/home/ubuntu/Developer/long-horizon-dynamics_run_818e189/resources/experiments/modeldev_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1`（`resources` symlink 到原 repo resources）。
- train tmux：`modeldev_tcnlstm_geoactctx_H10_nulltrust_s005_p1`
- GPU watch tmux：`modeldev_gpu_watch_tcnlstm_geoactctx_H10_s005`
- train log：`logs/train_phase1.log`
- GPU watch log：`logs/gpu_watch.log`
- init checkpoint：`resources/experiments/modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1/checkpoints/model-epoch=03-best_valid_loss=0.46.pth`
- 关键 CLI 配置：`model_type=tcnlstm`，`history_length=10`，`unroll_length=50`，`adaptive_history_context=true`，`adaptive_history_short_window=10`，`adaptive_history_mid_window=10`，`tcnlstm_side_history_selector_prior=null_short`，`tcnlstm_side_history_scale_init=0.005`，`history_context_mode=dmot_vbat`，`state_update_mode=residual_full_state`，只训练 `tcnlstm_side_history`，`batch_size=16`，`accumulate_grad_batches=32`，effective batch `512`，epochs `4`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`warmup_lr=1.5e-6`，`cosine_lr=4e-7`，early stopping patience `2`，`min_delta=2e-5`，WANDB disabled。首轮禁止 future context、`a/alpha` 和 true seq2seq。
- smoke：`smoke_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1` 已通过；one-batch train/valid finite；`history_context_dim=5`；trainable names 仅 24 个 `model.tcnlstm_side_history_*`；valid gate stats：null `0.5252`、short `0.4118`、mid `0.0485`、full `0.0145`、`gate_saturation=0.0`、`reliability_mean=0.4976`、`reliability_std=0.0445`、`side_residual_norm=1.15e-06`。
- 训练结果：训练 tmux 已退出，`train_summary.json` 已生成，`early_stopped=True`，`stopped_epoch=3`，`best_valid_loss=0.4615608752`，best checkpoint 为 `/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260516_tcnlstm_geoactctx_H10_nulltrust_s005_from_attitude_e3_p1/checkpoints/model-epoch=01-best_valid_loss=0.46.pth`。checkpoint 目录现有 `last_model.pth`、`model-epoch=00-best_valid_loss=0.46.pth`、`model-epoch=01-best_valid_loss=0.46.pth`、`model-epoch=02-best_valid_loss=0.46.pth`。训练日志无 OOM/NaN/Traceback；GPU watch tmux 仍在，可后续按用户决定清理。
- 正式 validation：
  - e0：`valid_loss_epoch=0.4615604877`，`valid_q_loss_epoch=0.0412399694`，`valid_v_loss_epoch=0.1502761692`，`valid_omega_loss_epoch=0.2378911227`，`valid_state_mse_epoch=0.1953031570`
  - e1：`valid_loss_epoch=0.4615604281`，`valid_q_loss_epoch=0.0412399657`，`valid_v_loss_epoch=0.1502761543`，`valid_omega_loss_epoch=0.2378911227`，`valid_state_mse_epoch=0.1953031123`
  - best epoch（epoch 1）：`valid_loss_epoch=0.4615604281`，`valid_q_loss_epoch=0.0412399657`，`valid_v_loss_epoch=0.1502761543`，`valid_omega_loss_epoch=0.2378911227`，`valid_state_mse_epoch=0.1953031123`
- selector / reliability（best epoch 观察值）：`null=0.5198`、`short=0.4173`、`mid=0.0482`、`full=0.0147`、`gate_saturation=0.0`、`reliability_mean=0.4969`、`reliability_std=0.0471`、`reliability_saturation=0.0`、`side_residual_norm=3.14e-05`。结论上未见 full gate 抢主导，side residual 保持很小。
- Gate：TCNLSTM H10 attitude reference `best_valid_loss=0.4615005`、`valid_q=0.0412516`、`valid_v=0.1503205`、`valid_omega=0.2379201`。Green：e0 `<=0.46155`，或 e1 `<=0.46145` 且 q/omega/v 至少两个指标改善。Yellow：`0.46155 < e0 <=0.46220` 且 q/omega 没有同步明显变坏，最多观察 e1。Red：e0 `>0.46220`，或 q/omega/state_mse 同步明显变坏，停止不读 horizon/test。
- 当前判断：`Yellow weak positive`。q/v/omega 都比 TCNLSTM H10 reference 略好，但整体 `valid_loss` 仍略差于 reference `0.4615005`，因此不触发 Green/freeze，也不触发 Red。
- horizon/test：未读取，当前明确保持 `no`。
- 下一步：等待用户 / GPT Pro 决策；不要自动启动新候选，不要自动读 horizon/test，不要恢复旧候选。
- 禁止动作：不要恢复旧 GRUTCN rawtokgeo，不要启动 true seq2seq，不要先上 H20/H50；不要读取 horizon/test，除非用户或 GPT Pro 明确允许。

## 本轮代码/协议变更（2026-05-15 23:40 CST）

- 改动文件：`scripts/dynamics_learning/models/tcnlstm.py`、`scripts/dynamics_learning/registry.py`。
- `TCNLSTM.forward()` 现在接受 `context_hist=None`，避免 `DynamicsLearning.forward(..., context_hist=...)` 在 TCNLSTM 上 TypeError。
- 新增默认关闭的 `adaptive_history_context` 支持：`tcnlstm_side_history_*` 分支把 H50 raw state/control、SO(3) geometric motion delta 和 past-only `dmot/vbat` 编码为 side context，通过 `null/short/mid/full` selector 与 reliability gate 产生 zero-init residual 注入 `head_input`。旧 checkpoint 初始为 no-op；旧路径不启用时行为不变。
- `registry.py` 将 `adaptive_history_context`、窗口参数和 `history_context_dim` 传给 TCNLSTM。检查：本地/远程 `py_compile` 和 `git diff --check` 通过；local/remote SHA256 一致；远程 dummy forward 与 one-batch smoke 通过。
- 2026-05-16 补充默认不改变旧行为的 null-trust 开关：`--tcnlstm_side_history_scale_init`（默认 `0.05`）和 `--tcnlstm_side_history_selector_prior uniform|null_short`（默认 `uniform`）。本地/远程 `py_compile`、`git diff --check` 和 H10 null-trust smoke 均通过。

## 最近停止候选

- `modeldev_20260515_tcnlstm_geoactctx_H50_from_attitude_e3_p1`：训练 early-stopped，`best_valid_loss=0.4656468630`，对比 H10 baseline 整体回退；不跑 horizon/test；watch 已清理，artifacts 保留。
- `modeldev_20260515_grutcn_adaptivehist_H50_from_vomega_e3_p1` 已按 e1 validation gate 停止，不跑 horizon/test。e0->e1 `valid_loss_epoch=0.6879014373 -> 0.6870185733`，e1 `valid_v=0.1602650136`、`valid_q=0.0390152447`、`valid_omega=0.2308619916`、`valid_state_mse=0.2525024712`，仍明显差于 current GRUTCN best；adaptive gate e1 `null=0.0084`、`short=0.7560`、`full=0.1937`、`gate_saturation=0.3051`，长历史 side branch 没显示泛化收益。训练/watch tmux 已停止，artifacts 保留。

## 本轮代码/协议变更（2026-05-15 16:45 CST）

- 改动文件：`scripts/dynamics_learning/models/grutcn.py`、`scripts/config.py`、`scripts/dynamics_learning/registry.py`、`scripts/dynamics_learning/lighting.py`、`scripts/train.py`。
- 新增 `--adaptive_history_context`、`--adaptive_history_short_window`、`--adaptive_history_mid_window`，默认关闭以保持旧行为。GRUTCN 新增 checkpoint-safe `adaptive_history_*` 分支：H20 anchor 主路径不变，长历史用 raw token Transformer side branch 做 `null/short/mid/full` selector，再通过 reliability gate 和 zero-init residual 修正 `history_context`。
- `DynamicsLearning` 新增 `log_adaptive_history_stats()`，训练/验证 CSVLogger 记录 gate mean/std/saturation 与 reliability mean/std/saturation；`train_summary.json` 写入 adaptive history 配置。
- 检查/同步：本地 `py_compile` 通过；远程 `python3 -m py_compile` 通过；远程 conda env dummy forward finite；远程脚本级 smoke 通过。同步时曾误把一份副本放到远程 `scripts/scripts/`，已只删除该误同步副本并按 repo 根目录重新同步正确路径。

## 本轮代码/协议变更（2026-05-15 14:10 CST）

- 改动文件：`scripts/dynamics_learning/models/grutcn.py`、`scripts/config.py`、`scripts/dynamics_learning/data.py`、`scripts/dynamics_learning/registry.py`、`scripts/dynamics_learning/lighting.py`、`scripts/train.py`、`scripts/eval.py`。
- 新增 `--history_context_mode none|dmot_vbat`，默认 `none` 保持旧行为。`dmot_vbat` 模式下 `DynamicsDataset` 只选历史 `dmot/vbat` 作为 `context_hist`；`DynamicsLearning.full_state_rollout()` 将 `context_hist` 传给模型；GRUTCN 新增 `history_context_*` gated/zero-init 分支，把历史 actuator/battery context 融入 `history_context`；train/eval summary 记录 `history_context_mode` 与 `history_context_dim`。
- 检查/同步：本地 `py_compile` 通过；远程 `py_compile` 通过；local/remote SHA256 一致；远程 smoke 通过，确认 `history_context_dim=5`、trainable count `19`、one-batch train/valid finite。

## 刚停止的候选

- `modeldev_20260515_grutcn_dmotvbatctx_H20_from_vomega_e3_p1` 已按 validation gate 停止，不跑 horizon/test。结构为 `history_context_mode=dmot_vbat`、past-only `dmot/vbat` context、`state_update_mode=residual_full_state`，只训练 `history_context`（trainable count `19`）。e0/e1 `valid_loss_epoch=0.6827901006 -> 0.6828653216`，e1 `valid_v=0.1596235037`、`valid_omega=0.2291074544`、`valid_q=0.0388034955`、`valid_state_mse=0.2502800524`，相对 current best 的 v/omega/q/state_mse 均无正信号。训练 tmux/GPU watch 已停止，匹配实验路径的训练进程已停止，artifacts 保留。结论：当前 tiny context branch 失败；下一步先诊断 context 字段/归一化/分支表达力，再决定 adaptive history selector 或 TCNLSTM 迁移。
- `modeldev_20260515_grutcn_rawtokgeo_H20_from_vomega_e3_p1` 已按 validation gate/early stop 判失败，不跑 horizon/test。训练 early-stopped，best 停在 e0：e0/e1/e2 `valid_loss_epoch=0.6829264164 -> 0.6830326319 -> 0.6831049919`，e2 `valid_v=0.1597713530`、`valid_q=0.0388241075`、`valid_omega=0.2290891409`、`valid_state_mse=0.2502748668`；相对 current best 的 `valid_loss/v/q/state_mse` 均无正信号。训练 tmux 已退出，GPU watch 已停止，artifacts 保留；horizon/test 未读取。结论：SO(3) raw-token representation fix 单独未带来 validation 正信号，不做 rawtokgeo continuation，转 `dmot/vbat` context。
- `modeldev_20260515_grutcn_softvomega_H20_from_hard_e1_p1` 已按 e0 validation gate 停止，不跑 horizon/test。e0 `valid_v=0.1629530936`、`valid_omega=0.2299810797`、`valid_q=0.0454349183`、`valid_state_mse=0.2749448121`，相对 hard e1 和当前 GRUTCN best 均明显恶化。只停止对应训练/GPU watch tmux，artifacts 保留。
- `modeldev_20260514_grutcn_hardvomega_multistep_H20_from_vomega_e3_p1` 已按 gate 停止，不跑 horizon/test。e1 曾反转，但 e2 把 v/omega 信号打回：e2 `valid_v=0.1598088145`、`valid_omega=0.2292566448`，相对当前 best reference 明显回退。只停止对应训练/GPU watch tmux，artifacts 保留。
- `modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1` 已按 gate 停止，不跑 horizon/test。e0/e1 validation 相对当前 GRUTCN best 参考 `valid_v=0.1595940590`、`valid_omega=0.2290888280`、`valid_q=0.0388018563` 没有反转；e1 `valid_loss_epoch=0.4708360136`、`valid_v=0.1596593559`、`valid_q=0.0388144702`、`valid_omega=0.2290989012`。训练 tmux 和 GPU watch tmux 已停止，只保留 artifacts，不删除实验产物。

## 当前代码/协议状态（2026-05-14 17:55 CST）

- 已实现 `eval no-silent-skip guard`：`scripts/eval.py` 在 eval 开始前检查 `unroll_length >= max(eval_horizons)`；`horizon_summary.json` 会记录 `actual_unroll_length`、`requested_eval_horizons`、`computed_eval_horizons`、`skipped_eval_horizons` 和 `state_update_mode`。若用 `unroll_length=2` 请求 `1,10,25,50`，现在会直接 `ValueError`，不再 silent skip h50。
- 已实现 `state_update_mode` wiring：`scripts/config.py` 新增 `--state_update_mode residual_full_state|hard_vomega_kinematic|soft_vomega_kinematic` 和 `--state_update_soft_residual_scale`；`scripts/dynamics_learning/lighting.py` 默认 `residual_full_state` 完全保留旧行为，hard 模式只用 `delta_v/delta_omega` 更新 v/omega 并用梯形积分恢复 p/q，soft 模式在 hard 结果上加小 p/theta residual；`scripts/train.py` 将这两个字段写入 `train_summary.json`。
- 验证/同步：本地和远程 `py_compile` 均通过；`rsync -avR` 已同步 `scripts/config.py`、`scripts/dynamics_learning/lighting.py`、`scripts/eval.py`、`scripts/train.py` 以及三份状态文档到 `gpu4060`，本地/远程 SHA256 一致。远程 smoke 通过：guard 正确拒绝 `unroll_length=2 + eval_horizons=1,10,25,50`；`summarize_horizon_metrics` 对 50 步输出 summary fields；hard/soft kinematic update 输出有限且 quaternion norm 为 1。
- 下一步：当前 rawtokgeo 与首版 `dmot/vbat` context 均已按 gate 失败。先做 context 失败复盘：确认 `context_hist` 字段 past-only、归一化/尺度、进入 forward、gate/zero-init 和 trainable scope 是否过弱；若实现无误且 tiny 分支表达力不足，下一候选优先做 adaptive history selector / multi-scale reliability gate（短窗 H20 anchor，长历史只作 gated side branch，并记录 gate mean/std/saturation），或将成熟 geometry/context trick 迁移到 `tcnlstm.py`。不要继续同类 tiny context/低 LR 微调。

## GPT Pro 策略更新（2026-05-15）

- 来源：用户提供的 `/Users/lixiang/Desktop/codex_priority_reorder_keep_seq2seq_last.md`。该建议覆盖 2026-05-14 “优先 Delta v / Delta omega 主预测”的优先级，但保留已实现的 eval guard 与 `state_update_mode` 作为可用 ablation。
- 核心判断：hard/soft vomega 失败只能说明当前“在已有 full-state residual head 上硬接/软接 v/omega kinematic update”的实现失败，不能否定 `Delta v / Delta omega` 父级思想；但 `true seq2seq Delta v / Delta omega` 更像 finite-horizon decoder，不应作为当前第一创新点，而应放到最后 fallback / late-stage extension。
- 新优先级：1) 完成当前 `raw_token_geometric_delta`；2) 若有接近正信号，最多做一次 conservative scope 或短 continuation；3) history-only `dmot/vbat` context branch；4) adaptive history selector / multi-scale reliability gate；5) 将成熟 geometry/context trick 迁移到 `tcnlstm.py`；6) 只有前面方向均无 horizon/test 聚合收益时，才启动完整 true seq2seq `Delta v[1:F] / Delta omega[1:F]` predictor。
- Gate 更新：rawtokgeo 属于 representation fix，除非 e0/e1 多项关键指标灾难性恶化，否则至少看 e1 或 natural finish；`dmot/vbat` context 重点看 `valid_v/valid_omega`、h50 `E_v/E_omega` 与 `MSE_1_to_F`，并先检查字段进入 forward、归一化和 leakage；adaptive selector 必须记录 gate mean/std/saturation；true seq2seq 不能用 e0 直接杀死，至少给 1-2 个完整 validation 周期和结构匹配 LR/scope。

## 旧 Active 归档

- 状态：当前无 active training/evaluation。`modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1` e2 horizon/test eval 已完成，但评估只展开到 2 步，h10/h25/h50 被脚本跳过，因此不能直接作为 long-horizon locked audit 结论。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1`
- evaluated checkpoint：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1/checkpoints/model-epoch-02-best_valid_loss=0.17.pth`
- eval tmux：已退出
- GPU watch tmux：已停止
- eval 日志：`logs/eval_horizontest_20260514_grutcn_multistepdelta_H20_e2_b32_mse_p1.log`
- GPU watch 日志：`logs/gpu_watch_eval.log`
- 训练结果：`train_summary.json` 显示 `best_valid_loss=0.1660599709`，`valid_v_loss_epoch=0.0156964753`、`valid_omega_loss_epoch=0.0856593177`、`valid_q_loss_epoch=0.0021088682`、`valid_state_mse_epoch=0.0295935068`；训练已自然结束，`trainable_parameter_names` 仅 `model.multi_step_*`，smoke checkpoint-safe load 和 one-batch finite 都通过，missing 仅新增 `multi_step_*` key，无旧主路径 shape mismatch。
- 已读取 horizon/test：`average rollout loss=0.17696784436702728`；`h=1` 仅有 `E_p=0.0027821236194022192`、`E_v=0.013420002662026935`、`E_q=0.0015253809243973744`、`E_omega=0.06523779146431298`、`MSE_x=0.015020398139281596`；`mean_1_to_F=0.003992972929253514`、`MSE_1_to_F=0.029530191901839857`。`h=10/25/50` 被脚本明确跳过，原因是 `unroll_length=2`。结论：这轮 multi-step 分支在当前配置下只验证了 2 步展开，不能拿来和 long-horizon baselines 做最终 locked audit 对比；下一步应先修正/增加 unroll length，再做真正的 horizon/test 判定。
- 最新完成实验：`modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1` e3 horizon/test MSE-schema eval p1。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`
- evaluated checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`（来自 `train_summary.json`）。
- eval log：`logs/eval_horizontest_20260514_grutcn_vomegaweight_cont_H20_e3_b32_mse_p1.log`
- 训练结果：自然结束，训练 tmux/watch 退出；e0-e3 加权 validation `0.6827921271 -> 0.6827870011 -> 0.6827825308 -> 0.6827781796`，`best_valid_loss=0.6827782393`；e3 unweighted `valid_p=0.0432733931`、`valid_v=0.1595940590`、`valid_q=0.0388018563`、`valid_omega=0.2290888280`、`valid_state_mse=0.2502373457`。
- horizon/test：average rollout loss `0.8086838126`（加权 loss，不与未改权重实验直接比较）；h50 `E_q=0.0852212122`、`E_v=0.3800343019`、`E_omega=0.3008244657`、`MSE_x=0.7797312295`；mean_1_to_F `E_q=0.0436578958`、`E_v=0.2156254028`、`E_omega=0.2425080560`；`MSE_1_to_F=0.3993478401`。
- 结论：相对 vomegaweight e3，`MSE_1_to_F 0.3993592611 -> 0.3993478401`，h50 `E_v/E_omega`、mean q/v/omega 和 h50 `MSE_x` 继续小幅改善；h50 `E_q` 基本持平但第 8 位略回退。该实验是新的 GRUTCN 开发期 best，但提升只有 `1e-5` 量级且仍远落后 strongest GRU targets，因此停止同类 reweight/latentse 微调，下一步转真正 `multi-step delta_v/delta_omega predictor`。eval/watch 已停止，artifacts 保留。

## 刚完成 Eval

- 实验 id：`modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`
- checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.47.pth`
- training best：`best_valid_loss=0.4708260596`
- eval log：`logs/eval_horizontest_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_e3_b32_mse_p1.log`
- average rollout loss：`0.5625137091`
- h50：`E_q=0.0852223037`、`E_v=0.3801424071`、`E_omega=0.3008873714`、`MSE_x=0.7797898716`
- mean_1_to_F：`E_q=0.0436600007`、`E_v=0.2156833140`、`E_omega=0.2425306675`
- `MSE_1_to_F=0.3993805727`
- 结论：cont2 是新的 GRUTCN 开发期 best，但只比 cont e5 微小改善，仍明显落后 strongest GRU targets；该 horizon/test 读取影响后续决策：停止同类 latentse/raw-token 低 LR 微调，转 v/omega 主导路线。

## 刚完成 Eval

- 实验 id：`modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`
- checkpoint：`checkpoints/model-epoch=03-best_valid_loss=0.68.pth`
- training best：加权 `best_valid_loss=0.6828041077`；unweighted e3 `valid_v=0.1596023738`、`valid_q=0.0388069339`、`valid_omega=0.2290926725`
- eval log：`logs/eval_horizontest_20260514_grutcn_vomegaweight_H20_e3_b32_mse_p1.log`
- average rollout loss：`0.8087291121`（加权 loss，不与未改权重实验直接比较）
- h50：`E_q=0.0852212047`、`E_v=0.3800689345`、`E_omega=0.3008467932`、`MSE_x=0.7797522271`
- mean_1_to_F：`E_q=0.0436584481`、`E_v=0.2156444532`、`E_omega=0.2425160258`
- `MSE_1_to_F=0.3993592611`
- 结论：相对 cont2 e3，`MSE_1_to_F`、h50 q/v/omega 和 mean q/v/omega 都小幅改善，是新的 GRUTCN 开发期 best；该 horizon/test 读取影响后续决策：允许一次更低 LR 的同权重 continuation 验证收益是否延续；若 continuation 没有继续改善聚合指标，则转 `multi-step delta_v/delta_omega predictor`。

## 当前候选

- 当前无 active 候选。最近两个 Pro 路线前置候选 `raw_token_geometric_delta` 与首版 `dmot/vbat` context 均未过 validation gate，且未读取 horizon/test。
- 下一步候选不应再是 tiny adapter 或低 LR continuation。优先级：1) 复盘 context wiring/normalization/trainable scope；2) 若 context 仍有合理假设，做 adaptive history selector / multi-scale reliability gate；3) 若 GRUTCN 继续卡住，将成熟 geometry/context trick 迁移到 `tcnlstm.py`；4) true seq2seq `Delta v / Delta omega` 仍放最后 fallback。

## Gate

- rawtokgeo gate：e0/e1 若多项关键指标同步灾难性恶化（尤其 q/state_mse 明显差于 current best 且 v/omega 同步回退）才停止不 eval；若只是轻微回退，至少观察 e1 或 natural finish。
- context/adaptive gate：`dmot/vbat` context 必须先 smoke，确认 past-only、字段进入 forward、归一化和 leakage 风险；adaptive history selector 除 validation/horizon 外要记录 gate mean/std/saturation。
- horizon/test 对比线：当前 GRUTCN best `MSE_1_to_F=0.3993478401`，h50 `E_q=0.0852212122`、`E_v=0.3800343019`、`E_omega=0.3008244657`。任一新候选若不能改善 `MSE_1_to_F` 或 h50 q/v/omega 的聚合表现，停止该结构分支并复盘。

## 下一结构候选

- Phase 1：context 失败复盘。检查 `dmot/vbat` past-only 字段、归一化/尺度、是否进入 forward、`history_context_*` gate 是否近似关闭、trainable count `19` 是否过弱。
- Phase 2：adaptive history selector。短窗 anchor 保持 H20，长历史只进 gated side branch，必须有 gate/attention/null context/reliability 抑制长历史噪声，并记录 gate mean/std/saturation。
- Phase 3：将成熟 raw-token geometry/context trick 迁移到 `tcnlstm.py`；暂不迁移 hard/soft vomega。
- Phase 4：只有前述方向均无 horizon/test 聚合收益时，才启动完整 true seq2seq `Delta v[1:F] / Delta omega[1:F]` predictor，并单独设置 LR/warmup/trainable scope，不能用 e0 直接杀死。

## 巡检规则

- 同一窗口普通 heartbeat：只读本文件 + 必要的 `AGENTS.md` 规则行；不要完整读取 `Prompt.md` / `MODEL_DEV_HANDOFF.md`。
- 远程巡检用一次性压缩脚本返回：tmux/进程/GPU/错误 grep/CSVLogger 最新 validation rows/checkpoint/train_summary/horizon 文件存在性。
- 避免 `tail` 训练进度条刷屏。
- 普通无变化巡检不写文档。
- 状态变化、新 best、失败、eval 完成、horizon/test metric 读取、异常或代码/协议变更时，更新 `MODEL_DEV_CURRENT.md`、`Prompt.md`、`MODEL_DEV_HANDOFF.md` 和 automation。
- 实验推进按“明确假设 -> 小验证/smoke -> validation gate -> horizon/test 聚合判断 -> 失败就换结构”循环；连续同类微调无聚合收益时应主动收缩方向并复盘，不继续堆相似 adapter。
