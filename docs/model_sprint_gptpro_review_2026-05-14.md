# long-horizon-dynamics 模型冲刺复盘文档（给 GPT Pro 分析瓶颈用）

生成时间：2026-05-14 13:27 CST  
本地仓库：`/Users/lixiang/Developer/long-horizon-dynamics`  
远程训练仓库：`/home/ubuntu/Developer/long-horizon-dynamics`  
远程 GPU：`gpu4060` / RTX 4060 8GB  

本文档目的：把当前任务定义、代码协议、数据流、指标公式、已做实验、每条实验线的结构改动和结果尽量自包含地整理出来，供 GPT Pro 判断目前模型冲刺的瓶颈在哪里，以及下一步该优先改哪里。本文不会包含私有原始轨迹、标签、checkpoint、完整日志或可复现实验样本；只使用仓库代码、长期实验记录和聚合指标。

## 1. 项目任务到底是什么

原始仓库是 Rao et al. `Learning Long-Horizon Predictions for Quadrotor Dynamics` 的代码框架。我们现在的目标不是复现实验，而是在这个代码框架上开发两个更强的新模型：`scripts/dynamics_learning/models/grutcn.py` 和 `scripts/dynamics_learning/models/tcnlstm.py`。

核心任务是四旋翼 full-state open-loop dynamics prediction。输入过去一段状态/控制历史和未来控制序列，在没有 ground-truth state correction 的情况下递归预测未来状态：

```text
x_t = [p_t, v_t, q_t, omega_t]
u_t = control input

given:
  X_hist = x_{t-H+1:t}
  U_hist = u_{t-H+1:t}
  U_future = u_{t:t+F-1}

predict:
  X_hat_{t+1:t+F}
```

注意评估是 open-loop rollout：第 `h+1` 步输入的是第 `h` 步预测出的状态，不再用真实状态纠正。因此 validation one-step/multi-step loss 好看不一定等价于 h50 指标好。

状态维度在当前实现里是 13：

| 片段 | 维度 | 含义 |
| --- | ---: | --- |
| `p` | 3 | 世界系位置 |
| `v` | 3 | 线速度 |
| `q` | 4 | quaternion，`wxyz` 顺序 |
| `omega` | 3 | 机体系角速度 |

模型实际输出不是 13 维完整状态，而是 12 维 delta：

```text
delta = [delta_p, delta_v, dtheta, delta_omega]
```

代码在 `DynamicsLearning.apply_full_state_update()` 中把 delta 变成下一状态：

```text
p_next     = p_t + delta_p
v_next     = v_t + delta_v
q_next     = normalize(q_t ⊗ Exp(dtheta))
omega_next = omega_t + delta_omega
```

所以当前不是“直接预测 quaternion 四元数”，而是预测局部姿态增量 `dtheta` 再通过指数映射更新 quaternion。这个设计本身是合理的。

## 2. 当前仓库结构和关键文件

仓库主要目录：

```text
AGENTS.md                         实验总规则
MODEL_DEV_CURRENT.md              当前 active/gate/路径/下一步的单页总控
MODEL_DEV_HANDOFF.md              详细交接文档
Prompt.md                         长期压缩实验记忆
README.md                         原始仓库说明
docs/archive/                     旧 Prompt/HANDOFF 归档
scripts/config.py                 CLI 参数、loss 权重、physics、multi-step flags
scripts/train.py                  训练入口、checkpoint 初始化、train_summary
scripts/eval.py                   horizon/test open-loop eval 和指标输出
scripts/dynamics_learning/data.py full-state HDF5 window dataset
scripts/dynamics_learning/lighting.py LightningModule、rollout、loss、physics loss
scripts/dynamics_learning/models/grutcn.py  当前主力 GRUTCN 改造模型
scripts/dynamics_learning/models/tcnlstm.py 当前 TCNLSTM 改造模型
scripts/aggregate_horizon_results.py        本地聚合 horizon 结果脚本
```

本地 `resources/experiments/` 只保留了少量早期实验。最近几天的训练/eval 主要在远程 `gpu4060` 上运行，结果通过 `Prompt.md`、`MODEL_DEV_HANDOFF.md`、`MODEL_DEV_CURRENT.md` 记录。因此本文的最近实验数据主要来自这些记录，而不是本地 artifacts 全量扫描。

## 3. 数据和训练协议

### 3.1 数据窗口

`scripts/dynamics_learning/data.py` 读取 trajectory-level HDF5，并构造滑动窗口。每个样本返回：

```python
{
    "x_hist": x[start:history_end],          # [H, 13]
    "u_hist": u[start:history_end],          # [H, 4]
    "u_roll": u[history_end-1 : history_end-1+F],  # [F, 4]
    "y_future": x[history_end : history_end+F],    # [F, 13]
    "context_hist": context[start:history_end],
}
```

模型输入 `z_hist = concat(x_hist, u_hist)`，所以 `input_size=17`。当前 full-state pipeline 强制 `predictor_type=full_state`，旧的 `velocity/attitude` Rao-style predictor 已被禁止，以避免和当前 full-state rollout 协议混用。

重要协议修复：`load_dataset()` 现在只在 training mode 下 shuffle：

```python
shuffle = args.shuffle and mode == "training"
```

这是一个关键修复。之前 validation/test 也可能被 shuffle，导致某些 continuation 的 e0 valid loss 异常到 `~0.600`，后来确认是协议 bug，不是模型真实崩坏。修复后 no-op checkpoint 可以回到原本 validation 数值。

### 3.2 训练 rollout

`scripts/dynamics_learning/lighting.py` 的 `full_state_rollout()` 递归展开：

1. 用当前 `x_hist_curr/u_hist_curr` 组成 `z_hist`。
2. 模型输出 `delta`。
3. `apply_full_state_update()` 得到 `x_next_pred`。
4. 和 `y_future[:, step]` 计算 step loss。
5. 把 `x_next_pred` 追加进 history，继续下一步。
6. `u_hist` 也滚动更新，使用已知 future control。

这意味着训练和 evaluation 都是 coupled full-state open-loop rollout，不是 teacher-forcing one-step。

### 3.3 训练 loss

当前基础训练 loss 是：

```text
L = lambda_p * E_p
  + lambda_v * E_v
  + lambda_q * E_q
  + lambda_omega * E_omega
```

其中：

```text
E_p     = mean ||p_pred - p_true||_2
E_v     = mean ||v_pred - v_true||_2
E_q     = mean SO(3) geodesic error(q_pred, q_true)
E_omega = mean ||omega_pred - omega_true||_2
```

默认 `lambda_p=lambda_v=lambda_q=lambda_omega=1.0`。后来为了用户希望“先汇报 v/omega 指标，只要 v/omega 变好也有意义”，引入了 v/omega 优先训练协议，例如：

```text
lambda_p = 0.6
lambda_v = 1.6
lambda_q = 0.9
lambda_omega = 1.6
```

这会让 `best_valid_loss` 变成加权值，因此不能和未改权重实验直接比较。对这类实验，gate 只看日志里额外记录的 unweighted `valid_v_loss_epoch`、`valid_omega_loss_epoch`、`valid_q_loss_epoch`、`valid_state_mse_epoch`。

### 3.4 新增 physics loss

受用户给的论文思路启发，`lighting.py` 增加了 train-only physics-informed regularization。当前实现不是完整 nominal dynamics，而是更稳的 kinematic consistency：

```text
pred_step      = p_pred_next - p_t
pred_trap_step = 0.5 * dt * (v_t + v_pred_next)
target_step    = p_true_next - p_t
target_trap    = 0.5 * dt * (v_t + v_true_next)
```

用 target 自身的运动学残差估计物理关系是否可靠：

```text
target_kinematic = ||target_step - target_trap||^2
reliability = exp(-physics_reliability_scale * target_kinematic)
slack = target_kinematic + physics_slack_margin
kinematic_loss = mean(reliability * relu(pred_kinematic - slack))
```

这相当于“物理信息可靠时惩罚模型，不可靠时自动减弱”，和用户提到的 slack/reliability 思路一致。实际实验里它能压低 validation 和 `MSE_1_to_F` 一点点，但没有显著突破 h50。

### 3.5 评估指标

`scripts/eval.py` 在 test set 上跑 full-state open-loop rollout，输出：

```text
E_p(h)     = mean_i ||p_true[i,h] - p_pred[i,h]||_2
E_v(h)     = mean_i ||v_true[i,h] - v_pred[i,h]||_2
E_q(h)     = mean_i d_SO3(q_pred[i,h], q_true[i,h])
E_omega(h) = mean_i ||omega_true[i,h] - omega_pred[i,h]||_2
MSE_x(h)   = mean_i ||x_true[i,h] - x_pred[i,h]||_2^2
```

`h50_E_v` 这样的指标就是第 50 步速度误差：

```text
h50_E_v = (1/N) * sum_i ||v_true[i,50] - v_pred[i,50]||_2
```

后来按用户要求新增了类似截图的多步汇总指标：

```text
MSE_1_to_F = (1/F) * sum_{h=1}^F MSE_x(h)
           = (1/(N F)) * sum_i sum_h ||x_true[i,h] - x_pred[i,h]||_2^2
```

注意：这里 `MSE_x` 对整条 state 向量直接做平方和，包含 quaternion 四个分量的欧氏项；而 `E_q` 是 SO(3) geodesic error。二者用途不同，`MSE_1_to_F` 是稳定聚合指标，`E_q` 是姿态几何指标。

## 4. 原始 baseline 和当前目标线

当前记录里最重要的 baseline 包络如下：

| config | best_valid_loss | h50_E_q | h50_E_v | h50_E_omega | mean_E_q | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gru_H10_F50_seed10` | `0.580389` | `0.0800042` | `0.356014` | `0.283807` | `0.0426877` | 当前最佳 h50 quaternion |
| `gru_H20_F50_seed10` | `0.573358` | `0.0802197` | `0.353015` | `0.260392` | `0.0420377` | 当前最佳 h50 velocity、omega、mean quaternion |
| `tcn_H10_F50_seed10` | `0.533292` | `0.0891246` | `0.347036` | `0.327506` | `0.0452568` | validation 强，h50 velocity 好，但 q/omega 不如 GRU |

因此真正要赢的不是单一模型，而是 strongest baseline envelope：

```text
h50_E_q     <= 0.0800042
h50_E_v     <= 0.353015   或至少 <= 0.347036 才明显强于 TCN velocity
h50_E_omega <= 0.260392
mean_E_q    <= 0.0420377
```

当前 GRUTCN 开发期最优仍为：

```text
MSE_1_to_F = 0.3993478401
h50_E_q    = 0.0852212122
h50_E_v    = 0.3800343019
h50_E_omega= 0.3008244657
```

与最强 GRU 包络相比，差距大致是：

```text
h50_E_q     差约 0.00522
h50_E_v     差约 0.02702
h50_E_omega 差约 0.04043
```

这说明当前 GRUTCN 后续改进虽然能稳定变好，但离真正 baseline win 还很远。

## 5. 当前两个模型的代码状态

### 5.1 GRUTCN 当前结构

`grutcn.py` 现在已经不是原始简单 GRU+TCN。它包含很多为了 checkpoint-safe ablation 保留下来的模块。

主路径：

1. `anchor_history_len = min(history_len, 20)`。即使未来实验用 H30/H50，校准 anchor 只看最近 H20，避免历史长度迁移直接破坏已有 checkpoint。
2. `TemporalConvNet` 编码最近 H20。
3. `encoder_norm` 后接 latent SE residual：`enc_seq = enc_seq + scale * enc_seq * zero_init_delta`。
4. temporal refiner attention：`LayerNorm -> MultiheadAttention -> FFN -> zero-init residual`。
5. `context_gru` 提取 history context。
6. attention over TCN features + null context gate。
7. base decoder 输出 `base_delta`。
8. raw-history token Transformer side branch 从原始 `[x,u]` 和 `dx` tokens 中提取 raw context。
9. raw-token head residual 注入 `head_input`。
10. velocity/attitude observers 和 raw-token velocity/attitude residual 只写输出局部通道。

当前新增的 `multi_step_delta_vomega` 分支：

```text
input = [raw_token_context,
         base_feature,
         context,
         history_context,
         decoder_feature,
         projected_x_last,
         raw_token_dx_last,
         y[:, 3:6],
         y[:, 9:12]]

output = 6 dims = correction to [delta_v, delta_omega]
```

它会把当前 one-step `y` 的 `delta_v/delta_omega` 再修正：

```text
delta_v     = y[:, 3:6]  + multi_step_delta[:, 0:3]
delta_omega = y[:, 9:12] + multi_step_delta[:, 3:6]
```

如果 `multi_step_kinematic_update=true`，代码还会用 `v/omega` 的梯形积分倾向去拉 `delta_p/dtheta`：

```text
kin_delta_p = dt * (v_t + 0.5 * delta_v)
kin_dtheta  = dt * (omega_t + 0.5 * delta_omega)

y[:,0:3] = y[:,0:3] + scale * (kin_delta_p - y[:,0:3])
y[:,6:9] = y[:,6:9] + scale * (kin_dtheta  - y[:,6:9])
```

但当前 `multi_step_kinematic_scale` 初始化为 `0.0`，如果只训练 `multi_step_*`，它可以学习打开或保持关闭。这比用户截图里的“严格只输出整段 Δv/Δω，然后 p/q 完全运动学恢复”更保守。它仍然是 rolling one-step 分支，不是一次性 seq2seq 输出 `F` 步 `delta_v/delta_omega`。

### 5.2 TCNLSTM 当前结构

`tcnlstm.py` 也已经做过多轮 checkpoint-safe 改造。

主设计目标是保留 TCN checkpoint 的 anchor 行为：

1. `encoder` 和 `decoder` 名字保持与 TCN baseline 兼容，使旧 TCN checkpoint 可以直接初始化 anchor。
2. `anchor_history_len = min(history_len, 10)`，H20 试验时 anchor 仍看 H10，长历史只进入额外分支。
3. `history_lstm`、attention decoder 和 latent context 用于小尺度 residual。
4. 多个失败候选（state/cell residual、lag observer、long-history branch、velocity residual 等）部分保留 key 兼容，但 forward 已回到较稳的 H10 attitude anchor context + attitude correction 路径。

TCNLSTM 目前最有价值的结果是 velocity partial win，但姿态/角速度没有赢。

## 6. 实验总览：哪些方向试过，具体改了什么，结果怎样

### 6.1 早期 GRUTCN anchor 系列

#### `modeldev_20260509_grutcn_anchored_context_residual_H20_p1`

结构：在 GRUTCN 中加入 anchored context residual，试图利用 GRU/TCN latent context 改善 long-horizon。

结果：

```text
best_valid_loss = 0.6142662
average rollout loss = 0.5747820139
h50_E_q     = 0.08536557
h50_E_v     = 0.40000114
h50_E_omega = 0.29806767
```

结论：比部分 TCN quaternion 指标好，但速度太差，未击败 GRU strongest targets。

#### `modeldev_20260509_grutcn_anchor_H20_e10_lowlr_swaft_p1`

结构/协议：低学习率 + SWA smoothing，想把 GRUTCN anchor 调稳。

结果：

```text
best_valid_loss = 0.4776078
average rollout loss = 0.5678076148
h50_E_q     = 0.0856633007
h50_E_v     = 0.3885666346
h50_E_omega = 0.3016224673
```

结论：validation 改善明显，但 h50 仍没有接近 GRU。

#### `modeldev_20260510_grutcn_anchor_e7_ultralow_p2`

结构：GRUTCN H20 pure stable anchor，ultra-low LR 延续，形成后续很多实验的初始化基准。

结果：

```text
best_valid_loss = 0.4720816
average rollout loss = 0.5632579327
h50_E_q     = 0.0853667619
h50_E_v     = 0.3810590338
h50_E_omega = 0.3012506581
mean_E_q    = 0.0437535938
```

结论：当时最强 validation checkpoint，但 locked audit 没赢。这个实验成为 GRUTCN 后续 frozen-anchor 小分支的起点。

### 6.2 GRUTCN history length 和输出尺度排查

#### H20 checkpoint 直接迁移到 H50/H10

`modeldev_20260510_grutcn_anchor_H50_fromH20_ultralow_p1`：

```text
e0/e1 valid = 0.9261976 -> 0.8817936
```

`modeldev_20260510_grutcn_anchor_H10_fromH20_ultralow_p1`：

```text
e0/e1 valid = 0.6045368 -> 0.5921974
```

结论：直接改变 `history_length` 会破坏 checkpoint 校准，尤其 positional/raw-history/TCN 时序分布都会变。后续才引入 `anchor_history_len`：anchor 固定看短窗，长历史只进入 gated/adaptive branch。

#### 删除 `output_gain`

`modeldev_20260510_grutcn_no_output_gain_H20_from_ultralow_p1`：

```text
e0 valid = 1.3612511
```

结论：`output_gain` 已经成为 checkpoint 输出尺度的一部分，删除它会直接破坏模型，不可行。

#### memory refresh boost

`modeldev_20260510_grutcn_refreshboost_H20_from_ultralow_p1`：

```text
e0/e1/e2 valid = 0.4836859 -> 0.4813507 -> 0.4818282
```

结论：偏向当前窗口 hidden state 也不能恢复到 anchor 水平，停止。

### 6.3 噪声和 tail-weighted rollout trick

#### input noise

`modeldev_20260510_grutcn_anchor_noise_H20_from_ultralow_p1`

改动：

1. `config.py` 增加 `input_noise_std`、`input_noise_loss_weight`。
2. `lighting.py` 增加 train-only input noise objective。
3. 验证、测试、评估保持 clean。

结果：

```text
best_valid_loss = 0.4722003
average rollout loss = 0.5636401176
h50_E_q     = 0.0854911
h50_E_v     = 0.3807892
h50_E_omega = 0.3017333
mean_E_q    = 0.0438164
```

结论：速度有一点变化，但整体 locked audit 失败，说明简单 noise robustness 没有解决 h50 主误差。

#### feedback noise + tail-weighted rollout

`modeldev_20260510_grutcn_feedbacktail_H20_from_noisee5_p1`

改动：

1. `feedback_noise_std`：rollout 时把预测状态加噪再喂回 history。
2. `rollout_loss_tail_weight`：让后段 horizon loss 权重更高。
3. quaternion-safe state noise，避免扰动后 q 不归一。

结果：

```text
e0/e1/e2 valid = 0.4727978 -> 0.4730892 -> 0.4725238
```

结论：e2 虽然下降，但没到 audit band，停止。说明训练技巧带来的 gain 是 `1e-4` 级，远小于 h50 差距。

### 6.4 encoder temporal refiner

`modeldev_20260510_grutcn_temporalrefine_H20_from_ultralow_e4_p1`

改动：在 `grutcn.py` 的 `encoder_norm(enc_seq)` 后、`context_gru` 前加 checkpoint-safe temporal refiner：

```text
LayerNorm -> MultiheadAttention -> LayerNorm -> FFN -> zero-init residual
```

结果：

```text
e0/e1/e2/e3 valid = 0.4727792 -> 0.4732321 -> 0.4731706 -> 0.4729020
best = 0.4727792442
```

结论：没有进入 screening band，无 audit。单纯 encoder-side attention refine 没有带来足够增益。

### 6.5 TCNLSTM true-anchor 和后续分支

#### TCNLSTM true-anchor H10

`modeldev_20260510_tcnlstm_trueanchor_refine_H10_from_tcnH10_p1`

关键结构：让 TCNLSTM 保持 TCN-compatible anchor，旧 TCN checkpoint 的 `encoder/decoder` 能直接加载；新增 latent context decoder 但 final zero-init。目的是先保住 TCN anchor，再让 LSTM/history context 做小修正。

validation：

```text
e0 valid_loss = 0.4623863
best_valid_loss = 0.4623865
```

locked audit：

```text
average rollout loss = 0.5286130309
h50_E_q     = 0.0909455538
h50_E_v     = 0.3444223371
h50_E_omega = 0.3350359351
mean_E_q    = 0.0458820136
mean_E_v    = 0.1851966663
mean_E_omega= 0.2572232571
```

结论：这是一个重要 partial win。`h50_E_v=0.3444223` 同时优于 `gru_H20=0.353015` 和 `tcn_H10=0.347036`，但 `E_q/E_omega/mean_E_q` 明显不够，不是全面 winner。

#### TCNLSTM attitude H10

`modeldev_20260510_tcnlstm_attitude_H10_from_trueanchor_e0_p1`

改动：新增 attitude attention + attitude correction branch，只写 `dtheta/delta_omega` 对应输出通道 `[:, 6:12]`，final zero-init，anchor 不变。

validation：

```text
best_valid_loss = 0.4615005
e4 valid_q     = 0.0412516
e4 valid_v     = 0.1503205
e4 valid_omega = 0.2379201
```

locked audit：

```text
average rollout loss = 0.5288807750
h50_E_q     = 0.0907420070
h50_E_v     = 0.3453549326
h50_E_omega = 0.3335756319
mean_E_q    = 0.0458699548
mean_E_v    = 0.1857792108
mean_E_omega= 0.2566082108
```

结论：h50 q/omega 比 true-anchor 稍好，但 velocity 回退；仍然只算 partial win，不是全面 winner。

#### TCNLSTM H20 history extension

`modeldev_20260511_tcnlstm_attitude_H20_from_H10e3_p1`

目标：尝试给 TCNLSTM 更多历史，看能否保持 H10 anchor 同时改善 hidden dynamics。

结果：e0 超过 stop line，判定跨 history mismatch 破坏校准；停止不 audit。

结论：TCNLSTM 和 GRUTCN 一样，直接扩长历史会引入分布迁移问题，不能直接 H10->H20/H30。

#### TCNLSTM stateinit/cellinit/attfine/dualview/lagobserver

这些都是围绕 H10 attitude best 的 checkpoint-safe 小结构。

`stateinit`：在原 state initializer 后加 residual 调 LSTM `h0/c0`。结果 e2 `valid_loss=0.4617738`，q/omega 从 e0 到 e2 连续回退，validation-only failure。

`cellinit`：只调 LSTM `c0`，保持 `h0`。结果 e2 `valid_loss=0.4617757`，同样 q/omega 回退，停止。

`attfine`：停用 state residual，只在输出端增加小姿态 fine residual。没有达到冻结线，方向收缩。

`dualview longtrans H20`：anchor 看 H10，长历史只写 translational residual `[:, :6]`。e0 `valid_loss=0.4628875 > 0.4622`，停止。

`lagobserver H10`：用 `[x_anchor, dx_anchor]` 提取 actuator/aero/filter lag context，只注入 head_input，不直接写 y。e2 刷到 `best_valid_loss=0.4614547`，但仍没过冻结线，且 q/omega 不健康，停止。

`velres H10`：后续 eval 读取到：

```text
average rollout loss = 0.5291858315
h50_E_q     = 0.0908154197
h50_E_v     = 0.3454937450
h50_E_omega = 0.3339743749
mean_E_q    = 0.0459003559
mean_E_v    = 0.1858658490
mean_E_omega= 0.2568162155
```

结论：相对 TCNLSTM attitude 没有改善，关闭 velocity-only residual 分支。

TCNLSTM 总结：它证明了“保住强 anchor + 少量 latent context”可让 velocity 赢，但每次动 LSTM hidden/cell 或姿态分支都会让 q/omega 卡住。它的速度优势值得保留，但要全面赢需要更结构化地约束姿态/角速度，而不是继续堆小 residual。

### 6.6 GRUTCN output observer 系列

#### velocity observer

`modeldev_20260511_grutcn_velobserver_H20_from_anchor_e4_p1`

结构：从 GRUTCN anchor e4 初始化，冻结 anchor，只训练 `velocity_observer_*`，只写 `y[:,3:6]`。final zero-init，scale `0.003`。

结果：

```text
average rollout loss = 0.5632427931
h50_E_q     = 0.0853668251
h50_E_v     = 0.3810359016
h50_E_omega = 0.3012500490
mean_E_q    = 0.0437536468
```

结论：只有 velocity 极微小变化，表达力不足。

#### attitude observer

`modeldev_20260511_grutcn_attobserver_H20_from_anchor_e4_p1`

结构：冻结 anchor，只训练 `attitude_observer_*`，只写 `y[:,6:12]`，试图改善 q/omega。

结果：

```text
best_valid_loss = 0.4720434
average rollout loss = 0.5632079840
h50_E_q     = 0.0853396507
h50_E_v     = 0.3809912724
h50_E_omega = 0.3012224156
mean_E_q    = 0.0437367480
```

结论：比 anchor/velocity observer 略好，有信号但很小。

#### joint observer

`modeldev_20260512_grutcn_jointobserver_H20_from_anchor_e4_p1`

结构：同时训练 `velocity_observer` 和 `attitude_observer`，只写 `y[:,3:6]` 和 `y[:,6:12]`。

结果：

```text
best_valid_loss = 0.4719877
average rollout loss = 0.5631942749
h50_E_q     = 0.0853401803
h50_E_v     = 0.3809713071
h50_E_omega = 0.3012216526
MSE_1_to_F  = 0.3996392623
```

结论：相对 attitude observer，v/omega 微小改善，q 微小回退。output observer 方向已经接近收益上限，停止继续堆这类小分支。

### 6.7 latent SE

`modeldev_20260512_grutcn_latentse_H20_from_anchor_e4_p1`

结构：在 TCN encoder 输出 `enc_seq` 后加 latent channel SE residual：

```text
enc_seq = enc_seq + latent_se_scale * enc_seq * zero_init_se_delta
```

它接在 latent channel 上，不直接缩放原始 q/omega/control，因此比原始输入前 SENet 更安全。

结果：

```text
best_valid_loss = 0.4721142
average rollout loss = 0.5632528663
h50_E_q     = 0.0853664371
h50_E_v     = 0.3810531020
h50_E_omega = 0.3012494289
MSE_1_to_F  = 0.3996510302
```

结论：单独 latent SE 不如 jointobserver。SE 本身后续作为 raw-token main 的一部分保留，但不再作为单独小 adapter 方向微调。

### 6.8 raw-history token Transformer side branch

`modeldev_20260512_grutcn_rawtoktf_H20_from_joint_e4_p1`

结构：

1. 输入 raw `[x,u]` 与 `dx` token。
2. `raw_token_proj` 后加 positional embedding。
3. 2-layer `TransformerEncoder`。
4. 使用 query pooling 得到 `raw_token_context`。
5. zero-init `raw_token_head_delta` 注入 head_input。
6. zero-init `raw_token_velocity/attitude` residual 只写 v/attitude 通道。
7. 不替换 encoder/base_decoder/state_initializer/GRU memory。

训练：

```text
e0-e4 valid = 0.4719697 -> 0.4719442 -> 0.4719169 -> 0.4718953 -> 0.4718790
best_valid_loss = 0.4718790
```

eval：

```text
average rollout loss = 0.5631443858
h50_E_q     = 0.0853260663
h50_E_v     = 0.3809074430
h50_E_omega = 0.3011899105
MSE_1_to_F  = 0.3996152170
```

结论：相对 jointobserver 小幅全面改善，是当时第一个比较稳定的 GRUTCN 结构正信号。但距离 strongest GRU targets 仍很远。

#### rawtoktf continuation

`modeldev_20260512_grutcn_rawtoktf_cont_H20_from_rawtok_e4_p1`

结构不变，从 rawtoktf e4 继续训练。

结果：

```text
best_valid_loss = 0.4716687
average rollout loss = 0.5630676746
h50_E_q     = 0.0853119630
h50_E_v     = 0.3808232145
h50_E_omega = 0.3011252939
MSE_1_to_F  = 0.3995643737
```

结论：继续训练有效，但斜率很小。说明 raw-token branch 有用，但只是把 GRUTCN 从 `0.39964` 推到 `0.39956`，还没有改变主要误差结构。

### 6.9 physics-informed kinematic regularization

`modeldev_20260513_grutcn_rawtoktf_physkin_H20_from_cont_e7_p1`

结构：不改 forward/model，继续 rawtoktf e7；训练时加 physics regularization。

配置：

```text
physics_loss_weight = 1000.0
physics_kinematic_weight = 1.0
physics_quat_norm_weight = 0.01
physics_v_smooth_weight = 0.0
physics_omega_smooth_weight = 0.0
physics_reliability_scale = 10.0
physics_slack_margin = 0.0
```

validation：

```text
0.4716556 -> 0.4716390 -> 0.4716238 -> 0.4716125 -> 0.4716022 -> 0.4715962
best_valid_loss = 0.4715962
train_physics_loss ≈ 7.024e-06
physics_reliability ≈ 0.9758
```

eval：

```text
average rollout loss = 0.5630497932
h50_E_q     = 0.0853132015
h50_E_v     = 0.3808130120
h50_E_omega = 0.3010985338
MSE_1_to_F  = 0.3995425400
```

结论：相对 rawtoktf cont e7，`h50_v/h50_omega/MSE_1_to_F` 极小改善，但 `h50_q/mean_E_q` 极小回退。physics kinematic loss 有帮助，但不是突破点。

### 6.10 H30 adaptive history

`modeldev_20260513_grutcn_rawtoktf_adapthist_H30_from_cont_e7_p1` 和 `adapthistfix_H30`

目标：用户提出历史长度可自适应选择。实现思路是 anchor 只看最近 H20，而 full H30 进入 `raw_token_adaptive_*` multi-scale context。multi-scale 包含 short/mid/full context，并用 gate 选择。

第一版失败原因：旧 raw_history residual 仍看到 full H30，导致非 zero-init 的旧路径吃到分布迁移，e0 valid 直接崩到 `0.6041508`。

修复版：

1. anchor 只看 recent H20。
2. old raw_history residual 也只看 recent H20。
3. full H30 只进入新增 `raw_token_adaptive_*`。
4. 冻结旧 H20 raw-token main branch，只训练 `raw_token_adaptive_*`。

结果：p2 仍然 e0 崩坏，说明“只训练新增长历史 adaptive branch”仍会快速破坏 validation，或长历史信号/positional/context gate 还不够稳。

结论：不能继续同形态 H30 adaptive-only。长历史方向如果继续，必须更保守：强 null context、极小 residual scale、严格 pretrain sanity，或者先做 H20 内结构升级。

### 6.11 latentsefix + shufflefix + continuation

背景：曾出现 latentsefix e0 valid `~0.600`，后来确认是 validation shuffle bug。修复后重新跑。

`modeldev_20260513_grutcn_rawtoktf_latentsefix_H20_from_physkin_e5_shufflefix_p1`

结构：raw-token main + latent SE，继续 physkin e5，physics OFF。

结果：

```text
best_valid_loss = 0.4711425900
average rollout loss = 0.5627378821
h50_E_q     = 0.0852424028
h50_E_v     = 0.3803976436
h50_E_omega = 0.3009922140
MSE_1_to_F  = 0.3994537865
```

结论：相对 physkin/rawtoktf 明显有改善，是当前 GRUTCN 主线继续下降的一段。

#### continuation e5

`modeldev_20260513_grutcn_rawtoktf_latentsefix_cont_H20_from_latentse_e4_p1`

结果：

```text
best_valid_loss = 0.4708843529
average rollout loss = 0.5625592470
h50_E_q     = 0.0852242063
h50_E_v     = 0.3801910379
h50_E_omega = 0.3009120027
MSE_1_to_F  = 0.3993963243
```

#### continuation cont2

`modeldev_20260514_grutcn_rawtoktf_latentsefix_cont2_H20_from_cont_e5_p1`

结果：

```text
best_valid_loss = 0.4708260596
average rollout loss = 0.5625137091
h50_E_q     = 0.0852223037
h50_E_v     = 0.3801424071
h50_E_omega = 0.3008873714
MSE_1_to_F  = 0.3993805727
```

结论：这条线确实持续改善，但改善幅度变成 `1e-5` 到 `5e-5` 量级。继续低 LR 微调已经不太可能追回 GRU h50 差距。

### 6.12 v/omega 优先训练协议

用户提出“可以重点报告 v/omega 指标，p/q 可作为辅助”。于是做了低风险前置验证：不改输出语义，只提高 `lambda_v/lambda_omega`，降低但不归零 `lambda_p/lambda_q`。

#### vomegaweight

`modeldev_20260514_grutcn_vomegaweight_H20_from_cont2_e3_p1`

配置：

```text
lambda_p = 0.6
lambda_v = 1.6
lambda_q = 0.9
lambda_omega = 1.6
```

结果：

```text
weighted best_valid_loss = 0.6828041077
unweighted valid_v       = 0.1596023738
unweighted valid_q       = 0.0388069339
unweighted valid_omega   = 0.2290926725
average rollout loss     = 0.8087291121  # weighted, 不和原 loss 直接比较
h50_E_q                  = 0.0852212047
h50_E_v                  = 0.3800689345
h50_E_omega              = 0.3008467932
MSE_1_to_F               = 0.3993592611
```

结论：相对 cont2，h50 q/v/omega 和 MSE 聚合均小幅改善。v/omega 提权是有效的，但幅度仍小。

#### vomegaweight continuation

`modeldev_20260514_grutcn_vomegaweight_cont_H20_from_vomega_e3_p1`

结果：

```text
weighted best_valid_loss = 0.6827782393
unweighted valid_p       = 0.0432733931
unweighted valid_v       = 0.1595940590
unweighted valid_q       = 0.0388018563
unweighted valid_omega   = 0.2290888280
unweighted state_mse     = 0.2502373457
average rollout loss     = 0.8086838126
h50_E_q                  = 0.0852212122
h50_E_v                  = 0.3800343019
h50_E_omega              = 0.3008244657
MSE_1_to_F               = 0.3993478401
```

结论：这是当前 GRUTCN 开发期 best，但相对上一个 best 只提高约 `1e-5`。它说明 v/omega reweight 可以沿着正确方向微调，但不可能靠同类训练协议追上 strongest GRU。

### 6.13 multi-step delta_v/delta_omega predictor

这是用户截图和 `idea.md` 里最相关的新方向。

用户给的理想结构是：

```text
主模型输出:
  delta_v_{t:t+F-1}
  delta_omega_{t:t+F-1}

p/q 通过运动学恢复:
  p_{t+1} = p_t + (v_t + v_hat_{t+1})/2 * dt
  q_{t+1} = q_t ⊗ Exp((omega_t + omega_hat_{t+1})/2 * dt)

可选:
  small delta_p_res / delta_theta_res
```

当前已实现的版本是更保守的 rolling one-step 分支：

```text
multi_step_delta_vomega=true
multi_step_kinematic_update=true
trainable_parameter_patterns=multi_step
```

它从当前 raw-token/latent/context 特征中输出当前 step 的 `delta_v/delta_omega` 修正，并可学习地把 `delta_p/dtheta` 往运动学积分结果拉。

#### 第一次正式训练但 unroll_length 配置错误

`modeldev_20260514_grutcn_multistepdelta_H20_from_vomegaweight_e3_p1`

结果：

```text
best_valid_loss = 0.1660599709
valid_v         = 0.0156964753
valid_omega     = 0.0856593177
valid_q         = 0.0021088682
valid_state_mse = 0.0295935068
```

这些数值看起来非常好，但原因是 `unroll_length=2`，所以训练和 eval 只覆盖两步。eval 也只产出：

```text
average rollout loss = 0.1769678444
h=1 E_v              = 0.0134200027
h=1 E_q              = 0.0015253809
h=1 E_omega          = 0.0652377915
MSE_1_to_F           = 0.0295301919
h10/h25/h50          = skipped, 因为 unroll_length=2
```

结论：不能作为 long-horizon locked audit。它只说明这个分支在 short horizon 下能训练。

#### 当前 active：修正到 unroll_length=50

`modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1`

配置：

```text
history_length = 20
unroll_length = 50
epochs = 4
batch_size = 16
accumulate_grad_batches = 32
effective batch = 512
limit_train_batches = 0.25
limit_val_batches = 0.5
warmup_lr = 8e-8
cosine_lr = 3e-8
trainable_parameter_patterns = multi_step
multi_step_delta_vomega = true
multi_step_kinematic_update = true
init checkpoint = vomegaweight_cont e3
```

启动检查：

```text
checkpoint load ok
missing only multi_step_* keys
no old main-path shape mismatch
trainable only model.multi_step_*
GPU about 1795/8188 MiB at startup
entered epoch 0
no validation row yet at recorded time
```

下一步应等 e0/e1，看 unweighted `valid_v/valid_omega/valid_q` 是否比 vomegaweight_cont e3 更好。只有 natural finish 或 validation-selected checkpoint 后才跑 horizon/test。

## 7. 用户给的 `idea.md` 和截图思路如何映射到当前代码

`/Users/lixiang/Documents/Obsidian Vault/科研文献分析/文献分析/idea.md` 里最适合当前项目的思路有三类。

### 7.1 只把复杂动力学交给网络，p/q 用运动学恢复

核心建议：

```text
网络重点学 delta_v 和 delta_omega
p 用速度积分恢复
q 用 omega 的 SO(3) 积分恢复
delta_p_res / delta_theta_res 只作为 ablation
```

这和用户截图一致，也是当前最值得 GPT Pro 深入判断的方向。当前代码还没有完全做到“主模型一次性输出整段 future delta_v/delta_omega”，只是在 one-step rolling 模型里加了 `multi_step_vomega` correction。它也没有强制 p/q 完全由运动学恢复，只是用可学习 `multi_step_kinematic_scale` 去靠近运动学结果。

建议 GPT Pro 重点分析：

1. 是否应该从当前保守实现升级成真正 seq2seq `delta_v/delta_omega` head。
2. 是否应当让 `delta_p_res/delta_theta_res` 默认关闭，仅作为 ablation。
3. 是否应该把 v/omega loss 作为主 objective，而 p/q 由运动学构造后只做辅助 regularization。

### 7.2 physics-informed weak loss

`idea.md` 建议不要一上来使用完整 nominal dynamics，因为 motor model、气动、电池、载荷等不一定可靠。当前我们采用的是更稳的 kinematic physics loss + slack/reliability。实验表明它可以带来微小改善，但不是突破。

GPT Pro 可分析：

1. 当前 physics loss 是否权重太弱/太强，为什么 `train_physics_loss` 非常小。
2. 是否应该对 q 的运动学一致性也显式加入 SO(3) loss，而不只是 quaternion norm。
3. 是否应该从 `p` kinematic consistency 转向 `v/omega` dynamics residual consistency。

### 7.3 PLE/MMoE translation-rotation soft decoupling

`idea.md` 推荐不要四任务平铺，而是两组动力学任务：

```text
translation group: p, v / delta_v, delta_p_res
rotation group: q, omega / delta_omega, delta_theta_res
shared experts: common actuator/aero/context
task-specific experts: translation-specific / rotation-specific
```

当前代码尝试过 velocity observer、attitude observer、joint observer，但它们只是 output residual，不是真正 PLE/MMoE soft decoupling。GPT Pro 可以考虑是否在 raw-token/context 特征上加 PLE-style expert routing，比继续 tiny observer 更有结构意义。

## 8. 当前最可能的瓶颈

基于实验记录，我认为瓶颈不是“训练没跑够”这么简单。

### 8.1 validation gain 与 h50 gain 强相关但斜率很小

从 rawtoktf -> continuation -> physkin -> latentsefix -> cont2 -> vomegaweight，validation 和 `MSE_1_to_F` 都能持续下降，但每轮 h50 只改善 `1e-5` 到 `1e-4`。这说明当前结构只是在同一局部 basin 里磨平误差，没有改变长时域误差传播方式。

### 8.2 GRUTCN 对 q/v/omega 的 h50 均落后 GRU，说明主误差不是单通道

很多 output observer 只改善某个通道一点点，但 h50 q/v/omega 一起离 GRU 很远。单独优化 velocity observer 或 attitude observer 不够；需要在动力学增量层面同时控制平移和旋转。

### 8.3 直接扩大 history_length 会破坏 checkpoint 校准

H20->H50/H10、TCNLSTM H10->H20、GRUTCN H30 adaptive 都暴露类似问题：历史长度变了，原 checkpoint 的 feature distribution 和 positional/context 路径不稳。后续如果继续长历史，必须做到 anchor 短窗固定、长历史 gated、null context、低尺度、先 no-op/smoke。

### 8.4 过多 checkpoint-safe tiny residual 保护了旧模型，但表达力不足

zero-init 小 adapter 能保证不炸，但也可能让模型只能做微调，无法跳到更好的动力学表示。实验上 output observer、latent SE、physics loss 都是稳定小增益。

### 8.5 当前 `multi_step_delta_vomega` 命名比实际实现更乐观

当前实现不是一次性 seq2seq 输出 future `delta_v/delta_omega`，仍然是 rolling one-step correction。它可能比之前更贴近用户想法，但未必真正解决“未来多步速度/角速度轨迹由未来控制驱动”的建模问题。

### 8.6 TCNLSTM 的 velocity partial win 很重要，但被 q/omega 拖住

TCNLSTM true-anchor 证明某些结构能明显赢 h50 velocity。也许速度预测和姿态/角速度预测需要不同的 inductive bias。把所有任务塞进同一个 residual head 可能不如 translation/rotation soft decoupling。

## 9. 给 GPT Pro 的重点问题

请 GPT Pro 重点分析以下问题：

1. 当前 `GRUTCN raw-token + latent SE + vomega weight` 已经稳定小幅改善，但离 GRU 差距很大。这个现象更像是容量不足、训练目标错位、还是结构 inductive bias 错？
2. 用户提出的“主模型只预测 `delta_v/delta_omega`，用运动学恢复 p/q，可选小 residual”是否应该成为主线？如果是，应该如何在当前代码里最小风险实现？
3. 当前 `multi_step_delta_vomega` 是 rolling one-step correction。是否需要改成真正 seq2seq head，输入 future controls，一次性输出 `F` 步 `delta_v/delta_omega`？
4. `p/q` 完全由运动学恢复会不会导致 `E_q` 更差？是否应该保留 `delta_theta_res`，但用 ablation 决定是否保存？
5. 当前 loss reweight 让 v/omega 改善一点点。如果最终只汇报 `E_v/E_omega`，是否应更激进地降低 `lambda_p/lambda_q`，还是会破坏 open-loop coupling？
6. physics regularization 当前只带来微小改善。是 loss 形式不够强，还是 nominal/kinematic 约束本身对数据集帮助有限？
7. 是否值得实现 PLE/MMoE translation-rotation soft decoupling，替代现在的 output observer 堆叠？
8. TCNLSTM true-anchor 的 velocity partial win 是否提示我们应该回到 TCNLSTM，把 GRUTCN raw-token 的优点迁移过去？
9. 当前训练普遍只用 `limit_train_batches=0.25`、`limit_val_batches=0.5`、epochs 4-8。对稳定方向是否可能欠拟合？还是 GPU/时间限制下这个 screening 足够？
10. 当前最该排查的协议 bug 是什么？例如 eval horizon、loss 权重可比性、history window/control 对齐、shuffle、checkpoint selection、unroll_length 等。

## 10. 我建议的下一步实验优先级

### 优先级 1：把当前 active `u50` multi-step 训练跑完并严格 eval

原因：这是目前最接近用户截图主线的在跑实验。必须先看 `unroll_length=50` 后是否还能保持 short-horizon 的巨大 validation 优势。如果 e0/e1 明显坏，说明当前 rolling correction 不是答案。

判据：

```text
对比 vomegaweight_cont e3:
valid_v       <= 0.1595940590
valid_omega   <= 0.2290888280
valid_q       不明显回退
h50_E_v       < 0.3800343019
h50_E_omega   < 0.3008244657
MSE_1_to_F    < 0.3993478401
```

### 优先级 2：如果 current u50 不够，改成真正 seq2seq `delta_v/delta_omega` predictor

建议结构：

```text
history encoder:
  H20 raw-token / latent context anchor

future control encoder:
  encode u_{t:t+F-1}

decoder:
  output delta_v[1:F], delta_omega[1:F]

state recovery:
  v_{h+1} = v_h + delta_v_h
  omega_{h+1} = omega_h + delta_omega_h
  p_{h+1} = p_h + 0.5*(v_h+v_{h+1})*dt
  q_{h+1} = q_h ⊗ Exp(0.5*(omega_h+omega_{h+1})*dt)

optional ablation:
  delta_p_res, delta_theta_res
```

这条线要非常小心和当前 `DynamicsLearning.full_state_rollout()` contract 对齐。也许需要让模型返回一个 one-step delta 仍可被现有 rollout 调用，或者增加新的 model mode/wiring。若大改 wiring，必须 smoke 保证 eval/train 一致。

### 优先级 3：PLE/MMoE translation-rotation experts

如果 seq2seq 改动太大，可先在当前 raw-token context 上加 PLE-style expert routing：

```text
shared experts: common hidden dynamics
translation experts: delta_v / delta_p_res
rotation experts: delta_omega / delta_theta
gates: translation gate, rotation gate
```

这比继续 `velocity_observer + attitude_observer` 更有结构含义，因为它在 latent dynamics 层共享/分离，而不是最后输出端修补。

### 优先级 4：回看 TCNLSTM velocity partial win

如果用户最终主要汇报 v/omega，TCNLSTM true-anchor 的 `h50_E_v=0.3444223` 已经强于 GRU H20 velocity。问题是 omega/q 不够。可以考虑：

1. 保留 TCNLSTM velocity path。
2. 单独为 omega 设计 physically integrated rotation head。
3. 不再碰 LSTM hidden/cell initializer，因为这些已经多次使 q/omega 回退。

## 11. 重要 caveats

1. 最近实验大多记录在远程，本文使用的是长期记录中的聚合结果；若要复核，需要登录远程读取对应 `train_summary.json`、`horizon_summary.json`、`metrics.csv`。
2. 加权 loss 实验的 `average rollout loss` 和 `best_valid_loss` 不可与未加权实验直接比，只能比 unweighted submetrics 和 horizon metrics。
3. `MSE_1_to_F` 是 full-state 欧氏 MSE 聚合，不等价于几何姿态误差；姿态应同时看 `E_q`。
4. 当前代码有不少失败候选模块保留 key compatibility，读 `grutcn.py/tcnlstm.py` 时不要把所有模块都误认为当前 active forward 的主要贡献。
5. current active `modeldev_20260514_grutcn_multistepdelta_u50_H20_from_vomegaweight_e3_p1` 在本文生成时还未产出 validation row，结论需要后续更新。

## 12. 一句话总结

目前我们已经证明：raw-history token Transformer、latent SE、physics weak loss、v/omega reweight 都能让 GRUTCN 稳定小幅变好，但所有 improvement 仍是同一个 basin 里的微调，远未缩小到 strongest GRU baseline 的 h50 差距。最值得 GPT Pro 判断的是：是否应彻底转向 `delta_v/delta_omega` 为主、`p/q` 由运动学恢复、translation/rotation soft decoupling 的结构，而不是继续在现有 full-state residual head 后面堆小 adapter。
