# 当前模型冲刺短状态

最后更新：2026-05-14 12:03 CST。

用途：同一聊天窗口和 heartbeat 自动化优先读取本文件，避免反复完整读取 `Prompt.md` / `MODEL_DEV_HANDOFF.md` 造成上下文膨胀。只有新聊天、上下文压缩后状态不明、当前状态冲突、或需要历史复盘时，才读取完整交接文档。

## 当前 Active

- 状态：当前 active training 为 `modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1`。这是把 `multi_step_delta_vomega` 结构正式重跑到 `unroll_length=50` 的 long-horizon 候选；前一次错误启动因 init checkpoint 路径少写一个 `=` 已确认失败并清理对应空运行目录和 GPU watch tmux，当前这次是修正后的重启版。
- 远程路径：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1`
- 训练 tmux：`modeldev_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1`
- GPU watch tmux：`modeldev_gpu_watch_grutcn_multistepdelta_u50_H20`
- 训练日志：`logs/train_phase1.log`
- GPU watch 日志：`logs/gpu_watch.log`
- init checkpoint：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1/checkpoints/model-epoch=03-best_valid_loss=0.68.pth`
- 当前配置：`history_length=20`、`unroll_length=50`、`epochs=4`、`batch_size=16`、`accumulate_grad_batches=32`、`limit_train_batches=0.25`、`limit_val_batches=0.5`、`warmup_lr=8e-8`、`cosine_lr=3e-8`、`warmup_steps=50`、`cosine_steps=1500`、`early_stopping=true`、`early_stopping_patience=2`、`early_stopping_min_delta=5e-6`、`WANDB_MODE=disabled`、`trainable_parameter_patterns=multi_step`、`multi_step_delta_vomega=true`、`multi_step_kinematic_update=true`
- 启动检查：tmux/watch alive，GPU 约 `1795/8188 MiB`、util `32%`；日志已确认成功加载 `model-epoch=03-best_valid_loss=0.68.pth`，缺失项仅新增 `multi_step_*` key，无旧主路径 shape mismatch，trainable 参数也只剩 `model.multi_step_*`；训练已进入 epoch 0，尚未生成 validation row，horizon/test metric 尚未读取。
- 下一步：等 e0/e1 validation，再根据 unweighted `valid_v/valid_omega/valid_q` 与 cont2 / vomegaweight 基准比较；只有在自然结束或 validation-selected checkpoint 后才跑/read horizon/test MSE-schema。若聚合指标没有继续改善，就停止同类微调，转真正 `multi-step delta_v/delta_omega predictor` 的结构审计。

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

- 下一候选：真正 `multi-step delta_v/delta_omega predictor`。
- 假设：当前 reweight/latentse 路线能稳定压低 `valid_v/valid_omega`，但 horizon/test 改善已进入 `1e-5` 量级，难以靠同类微调追上 GRU strongest targets；需要把结构从“单步 state correction”改为“显式预测未来速度/角速度增量，并用运动学恢复 p/q”，让长时域误差的主驱动项直接被建模。
- 初始实现边界：优先只改 `scripts/dynamics_learning/models/grutcn.py` 和必要 train/test/eval wiring；保持 checkpoint-safe，旧 raw-token/latent_se 主路径可作为 anchor，新增分支默认 zero-init 或小尺度注入。

## Gate

- 下一结构必须先 smoke：从当前 vomegaweight_cont e3 checkpoint 加载时只能 missing 新增 `multi_step_*` / `delta_*` keys，不允许旧主路径 shape mismatch；one-batch train/valid finite；trainable patterns 必须明确包含新分支，避免白训。
- validation gate 初期参考当前 best 的 unweighted e3：`valid_v=0.1595940590`、`valid_omega=0.2290888280`、`valid_q=0.0388018563`。若 e0 明显破坏 q 或 v/omega，停止该候选；若 v/omega 有正信号且 q 可控，natural finish 后跑/read horizon/test MSE-schema。
- horizon/test 对比线：当前 GRUTCN best `MSE_1_to_F=0.3993478401`，h50 `E_q=0.0852212122`、`E_v=0.3800343019`、`E_omega=0.3008244657`。新结构若不能改善 `MSE_1_to_F` 或 h50 v/omega，停止该结构分支并复盘。

## 下一结构候选

- 若 cont2 失败或 horizon/test 不再改善，优先转 `multi-step delta_v/delta_omega predictor`：输入历史 `[p,v,q,omega]`、历史控制和 future controls，核心输出整段 `delta_v_{t:t+F-1}` 与 `delta_omega_{t:t+F-1}`；`p/q` 默认通过运动学积分恢复，即 `p` 用梯形积分、`q` 用 `omega` 梯形积分。
- 小 `delta_p_res` / `delta_theta_res` 只作为 ablation：若加入后 `E_v`、`E_omega`、`E_q` 或 `MSE_1_to_F` 更好则保留，否则删掉，避免重新变成自由预测完整 state。
- 低风险前置验证可先做 `lambda_v/lambda_omega` 提权、`lambda_p/lambda_q` 降低但不归零，检验是否改善 v/omega 主导误差。

## 巡检规则

- 同一窗口普通 heartbeat：只读本文件 + 必要的 `AGENTS.md` 规则行；不要完整读取 `Prompt.md` / `MODEL_DEV_HANDOFF.md`。
- 远程巡检用一次性压缩脚本返回：tmux/进程/GPU/错误 grep/CSVLogger 最新 validation rows/checkpoint/train_summary/horizon 文件存在性。
- 避免 `tail` 训练进度条刷屏。
- 普通无变化巡检不写文档。
- 状态变化、新 best、失败、eval 完成、horizon/test metric 读取、异常或代码/协议变更时，更新 `MODEL_DEV_CURRENT.md`、`Prompt.md`、`MODEL_DEV_HANDOFF.md` 和 automation。
- 实验推进按“明确假设 -> 小验证/smoke -> validation gate -> horizon/test 聚合判断 -> 失败就换结构”循环；连续同类微调无聚合收益时应主动收缩方向并复盘，不继续堆相似 adapter。
