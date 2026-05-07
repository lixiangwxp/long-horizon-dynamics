# Long Horizon Dynamics 交接文档

更新时间：2026-05-06

这份文档用于在换账号、换聊天框、Codex 上下文丢失、或者需要人工接手时，快速恢复当前 NeuroBEM full-state 实验的状态。

敏感信息没有写进这个 Git 交接文档。若在当前 Mac 本机接手，可看同目录下的 `PROJECT_HANDOFF_SECRETS.local.md`。该文件已经加入 `.git/info/exclude`，不会进入 Git。

## 1. 当前目标

在远端 RTX 4060 机器上跑 NeuroBEM full-state 超参遍历：

- model: `mlp`, `gru`, `tcn`, `tcnlstm`, `grutcn`
- history_length: `1`, `10`, `20`, `50`
- unroll_length: `50`
- 当前采用较快实验设置：`epochs=200`, `patience=20`, `min_delta=1e-4`
- W&B 优先 online，失败时 offline
- 已完成的配置不重新跑，继续补未完成配置

## 2. 关键路径

Mac 本地仓库：

```bash
/Users/lixiang/Developer/long-horizon-dynamics
```

远端 SSH alias：

```bash
gpu4060
```

远端仓库：

```bash
/home/ubuntu/Developer/long-horizon-dynamics
```

远端 conda 环境：

```bash
dynamics_learning
```

远端 tmux session：

```bash
neurobem_sweep
```

当前 sweep root：

```bash
/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online
```

Mac 本地对应结果目录：

```bash
/Users/lixiang/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online
```

## 3. 最近已知状态

最近一次检查时，远端 `neurobem_sweep` 正在运行。

- 当前训练配置：`tcn_H10_F50_seed10`
- 当前训练从 checkpoint 恢复：

```bash
/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/tcn_H10_F50_seed10/checkpoints/last_model.pth
```

- 当前训练日志：

```bash
/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/tcn_H10_F50_seed10/logs/train_attempt_1_b128_a4_online_rerun2.log
```

最近一次结果汇总里已经成功的配置：

- `mlp_H1`, `mlp_H10`, `mlp_H20`, `mlp_H50`
- `gru_H1`, `gru_H10`, `gru_H20`, `gru_H50`
- `tcn_H1`

`tcn_H10` 之前有旧的 missing/failed 记录，现在正在重新训练，等 train/eval 完成后再更新汇总。

已经成功的 H1 不需要重新跑。sweep 脚本会根据每个实验目录下的 `status.json` 跳过已成功配置。

## 4. 我改过的代码

本轮关键改动有两个文件：

```bash
scripts/dynamics_learning/lighting.py
scripts/run_neurobem_sweep.sh
```

### `scripts/dynamics_learning/lighting.py`

修改点：如果验证集 loss 出现 `NaN` 或 `inf`，把 `best_valid_loss` 也置为该非有限值。

原因：Lightning 的 EarlyStopping 监控的是 `best_valid_loss`。如果当前 `valid_loss` 已经 NaN，但 `best_valid_loss` 保持旧的正常数值，早停可能无法及时触发，训练会继续写坏日志。

### `scripts/run_neurobem_sweep.sh`

修改点：

- 增加检测训练日志中非有限 loss 的逻辑。
- 如果已有 `last_model.pth`，但最近训练日志出现 NaN/inf，则不从这个 checkpoint 恢复。
- rerun 日志使用不覆盖旧日志的文件名。

原因：避免从已经数值坏掉的 checkpoint 继续训练，也保留失败现场，方便复盘。

语法检查已经通过：

```bash
python -m py_compile scripts/dynamics_learning/lighting.py
bash -n scripts/run_neurobem_sweep.sh
```

远端同步后做过 hash 对比，两个文件在 Mac 和 4060 上一致。

## 5. 当前已知风险

### 4060 曾经卡死

卡死前 GPU 显存约 `7850MiB / 8188MiB`，GPU 利用率 99%。这说明训练很可能把 8GB 显存压得比较满，再叠加 WSL/桌面/远程连接，机器可能变得很难响应。

当前策略：

- 降低 micro batch size。
- 用 `accumulate_grad_batches` 保持 effective batch size。
- 避免同时跑多个训练。
- tmux 只负责保活训练，不保证机器重启后还能继续。

### 聚合脚本曾出现 `_csv.Error: line contains NUL`

这通常是机器卡死或强制重启时，Lightning CSV log 被写坏，文件中出现 NUL 字符。

处理原则：

- 不删除原始日志。
- 先定位是哪一个 CSV 文件坏了。
- 如果再次阻塞 sweep，可以小改 `aggregate_horizon_results.py`，让它读取 CSV 时跳过 NUL 字符或跳过坏行，并在报告中记录。

## 6. 怎么检查远端是否活着

先在 Mac 上运行：

```bash
ssh gpu4060
```

解释：

- `ssh` 是 secure shell，用来登录远端机器。
- `gpu4060` 是 `~/.ssh/config` 里配置的主机别名。
- 如果远端 IP 变了，先修改 Mac 的 `~/.ssh/config` 里的 `HostName`。

进入远端后：

```bash
tmux ls
```

解释：

- `tmux` 是终端复用工具。
- `tmux ls` 列出当前后台还活着的 tmux 会话。
- 如果看到 `neurobem_sweep`，说明 sweep 终端会话还在。

查看训练画面：

```bash
tmux attach -t neurobem_sweep
```

解释：

- `attach` 是重新连回 tmux 会话。
- `-t neurobem_sweep` 指定连接名为 `neurobem_sweep` 的会话。
- 退出观察但不停止训练：先按 `Ctrl-b`，松开后按 `d`。

查看 GPU：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

解释：

- `nvidia-smi` 是 NVIDIA 的显卡状态工具。
- 在 WSL 里常见路径是 `/usr/lib/wsl/lib/nvidia-smi`。
- 重点看 `Memory-Usage`、`GPU-Util`、`Processes`。

查看 Python 训练进程：

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E "python|train.py|run_neurobem" | grep -v grep
```

解释：

- `ps` 查看进程。
- `-e` 表示列出所有进程。
- `-o pid,ppid,stat,etime,cmd` 指定输出列：
  - `pid`: 进程号
  - `ppid`: 父进程号
  - `stat`: 进程状态
  - `etime`: 已运行时间
  - `cmd`: 启动命令
- `grep -E` 用正则筛选训练相关进程。
- `grep -v grep` 去掉 grep 自己这一行。

## 7. 怎么看最新日志

在远端运行：

```bash
cd /home/ubuntu/Developer/long-horizon-dynamics
```

解释：进入远端项目目录。

```bash
latest_log=$(find resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online -path "*/logs/*.log" -type f -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d" " -f2-)
```

解释：

- `find` 找所有日志文件。
- `-path "*/logs/*.log"` 只匹配 logs 目录下的 `.log`。
- `-printf "%T@ %p\n"` 输出修改时间戳和路径。
- `sort -n` 按时间排序。
- `tail -n 1` 取最新一个。
- `cut -d" " -f2-` 去掉时间戳，只保留路径。
- `latest_log=$(...)` 把结果保存到变量里。

```bash
echo "$latest_log"
tail -n 80 "$latest_log"
```

解释：

- `echo` 打印变量，确认正在看哪个日志。
- `tail -n 80` 看文件最后 80 行，通常最新错误或当前 epoch 都在最后。

## 8. 怎么重新汇总结果

在远端项目目录运行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynamics_learning
```

解释：

- `source` 加载 conda 的 shell 配置。
- `conda activate dynamics_learning` 进入实验环境。

然后运行：

```bash
python scripts/aggregate_horizon_results.py \
  --experiments-root resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online \
  --output resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/horizon_results.csv \
  --plots-dir resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/horizon_curves \
  --include-missing
```

解释：

- `python scripts/aggregate_horizon_results.py` 运行结果汇总脚本。
- `--experiments-root` 指定所有实验子目录所在位置。
- `--output` 指定总 CSV 输出路径。
- `--plots-dir` 指定曲线图输出目录。
- `--include-missing` 把未完成或失败配置也写入汇总，方便看全局进度。

## 9. 怎么从 Mac 同步代码到 4060

在 Mac 本地运行：

```bash
cd /Users/lixiang/Developer/long-horizon-dynamics
scripts/sync_to_gpu.sh
```

解释：

- `cd` 切到本地仓库。
- `scripts/sync_to_gpu.sh` 是本项目的同步脚本。
- 当前约定：Mac 是代码源头，代码修改先在 Mac 做，再同步到 4060。

## 10. 怎么把远端结果同步回 Mac

在 Mac 本地仓库运行：

```bash
rsync -a --human-readable --stats --prune-empty-dirs \
  --include='*/' \
  --include='RUN_REPORT.md' \
  --include='horizon_results.csv' \
  --include='horizon_curves/***' \
  --include='*/args.txt' \
  --include='*/status.json' \
  --include='*/train_summary.json' \
  --include='*/horizon_metrics.csv' \
  --include='*/horizon_summary.json' \
  --include='*/checkpoints/*.pth' \
  --include='*/logs/*.log' \
  --include='*/csv_logs/***' \
  --exclude='*' \
  gpu4060:/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/ \
  resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online/
```

解释：

- `rsync` 用来同步文件。
- `-a` 是 archive 模式，保留目录结构和时间戳。
- `--human-readable` 让大小显示更容易读。
- `--stats` 最后打印同步统计。
- `--prune-empty-dirs` 不保留空目录。
- `--include` 是允许同步的文件类型。
- `--exclude='*'` 表示其他没列出的文件都不同步。
- 最后一行前半段是远端来源，最后一行后半段是 Mac 本地目标。

## 11. 如果需要继续启动 sweep

只在确认没有正在运行的 sweep，或者确定要重启当前 sweep 时使用。

远端命令：

```bash
cd /home/ubuntu/Developer/long-horizon-dynamics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynamics_learning
tmux new-session -d -s neurobem_sweep "cd /home/ubuntu/Developer/long-horizon-dynamics && source ~/miniconda3/etc/profile.d/conda.sh && conda activate dynamics_learning && scripts/run_neurobem_sweep.sh --sweep-root /home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online --wandb-mode auto --epochs 200 --patience 20 --min-delta 1e-4 --batch-sizes 128,64,32 --accum-steps 4,8,16 --limit-train-batches 0.25 --limit-val-batches 0.5 --limit-predict-batches 0 --shuffle True"
```

解释：

- `tmux new-session -d` 新建一个后台 tmux 会话。
- `-s neurobem_sweep` 指定会话名。
- 引号里的内容是这个 tmux 会话启动后要执行的完整训练命令。
- `--batch-sizes 128,64,32` 是按顺序尝试的 micro batch size。
- `--accum-steps 4,8,16` 是对应的梯度累积步数。
- 这样可以在降低显存压力时尽量保持 effective batch size。

如果已有同名 tmux，会提示 session already exists。不要直接杀，先进去确认：

```bash
tmux attach -t neurobem_sweep
```

## 12. 自动巡检说明

当前聊天里曾设置过 heartbeat 自动巡检，每 30 分钟检查远端：

- SSH 是否能连
- `tmux ls`
- 最新 tmux/log 输出
- `horizon_results.csv`
- 各实验的 `status.json`, `horizon_summary.json`, `horizon_metrics.csv`
- GPU 状态和训练进程

重要：heartbeat、子 agent、聊天上下文通常绑定当前账号和当前线程。换账号后，远端训练和本地文件还在，但自动巡检和聊天里的连续记忆可能不在。

换账号后，新的 Codex 可以靠这份文档继续接手。

## 13. 换账号后哪些东西还在

还在本地或远端的东西：

- Mac 本地代码仓库
- Mac 本地实验结果副本
- 远端 4060 上的代码、checkpoint、日志、结果
- Git commit 历史
- 这份 `PROJECT_HANDOFF.md`

不一定还在或不一定能直接看到的东西：

- 当前聊天窗口
- 当前聊天上下文
- heartbeat 自动巡检
- 子 agent 状态
- Codex UI 里的项目列表显示

所以重要信息要落到仓库文件里，尤其是：

- `PROJECT_HANDOFF.md`
- `RUN_REPORT.md`
- `horizon_results.csv`
- 每个实验的 `status.json`
- 关键日志

## 14. 接手后的建议顺序

1. 在 Mac 上确认代码状态：

```bash
cd /Users/lixiang/Developer/long-horizon-dynamics
git status --short
```

2. SSH 到远端：

```bash
ssh gpu4060
```

3. 看 tmux 是否活着：

```bash
tmux ls
```

4. 看 GPU 和训练进程：

```bash
/usr/lib/wsl/lib/nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | grep -E "python|train.py|run_neurobem" | grep -v grep
```

5. 看最新日志和 `RUN_REPORT.md`。

6. 如果训练完成，先同步结果回 Mac，再分析 `horizon_results.csv`。

7. 如果训练失败，先保留失败日志，不删除文件，再根据错误做最小修改。

## 15. 不要做的事

- 不要删除实验目录。
- 不要 `git reset --hard`。
- 不要直接清空日志或 checkpoint。
- 不要把 W&B key 写进代码或文档。
- 不要在远端手改代码后忘记同步回 Mac。
- 不要重新跑已经 `success` 的配置，除非明确要做复现实验。

## 16. 2026-05-07 当前状态快照

更新时间：2026-05-07 10:59 CST。

这个聊天线程已经很长，Codex App 多次触发“上下文已自动压缩”，并出现过：

```text
Error running remote compact task: stream disconnected before completion
```

这通常说明当前聊天上下文太重，或者压缩请求在网络/服务端中途断流。它不代表远端 4060 训练挂了。远端训练跑在 `tmux` 里，只要 `tmux neurobem_sweep`、`run_neurobem_sweep.sh` 和对应的 `python train.py` 进程还在，训练就还在继续。

建议后续开一个新的 Codex 聊天框接手。新聊天第一句话可以直接写：

```text
请先阅读 long-horizon-dynamics 仓库里的 PROJECT_HANDOFF.md，然后继续监控 neurobem_sweep。不要删除文件，优先只读巡检，必要时按文档恢复。
```

当前远端状态：

- 远端主机：`gpu4060`，机器名 `DESKTOP-0R0T1BO`。
- 远端仓库：`/home/ubuntu/Developer/long-horizon-dynamics`。
- sweep root：`/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online`。
- tmux session：`neurobem_sweep`，仍然存在。
- 当前运行配置：`tcn_H20_F50_seed10`。
- 当前训练命令核心参数：`batch_size=128`，`accumulate_grad_batches=4`，`epochs=200`，`patience=20`，`limit_train_batches=0.25`，`limit_val_batches=0.5`，`wandb_mode=online`。
- 最新日志：`tcn_H20_F50_seed10/logs/train_attempt_1_b128_a4_online.log`。
- 最新日志进度：epoch 106，约 60%，`valid_loss_epoch=0.606`，`best_valid_loss=0.586`。
- GPU 状态：RTX 4060，约 50C，1294/8188 MiB，利用率约 38%。
- 最新日志没有发现 `Traceback`、`CUDA out of memory`、`RuntimeError`、`nan`、`inf`。

当前结果状态：

- `horizon_results.csv` 当前有 10 行成功结果。
- `success=10`，`bad_or_missing=0`。
- 目前已有 10 个 `horizon_summary.json`、10 个 `horizon_metrics.csv`、10 个 `status.json`。
- 已成功配置：
  - `mlp_H1_F50_seed10`
  - `mlp_H10_F50_seed10`
  - `mlp_H20_F50_seed10`
  - `mlp_H50_F50_seed10`
  - `gru_H1_F50_seed10`
  - `gru_H10_F50_seed10`
  - `gru_H20_F50_seed10`
  - `gru_H50_F50_seed10`
  - `tcn_H1_F50_seed10`
  - `tcn_H10_F50_seed10`

已经成功的 H1 配置不需要重跑。当前脚本本身会跳过 `status.json` 里已经标记为 `success` 的配置；除非明确要做复现实验，不要重新跑已成功配置。

接手后建议第一轮只读巡检：

```bash
ssh gpu4060
tmux ls
/usr/lib/wsl/lib/nvidia-smi
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "python train.py|python eval.py|run_neurobem_sweep|tmux" | grep -v grep
```

如果要看当前训练画面：

```bash
ssh gpu4060
tmux attach -t neurobem_sweep
```

退出 tmux 画面但不停止训练：先按 `Ctrl-b`，松开后按 `d`。

如果远端 IP 变了，只需要在 Mac 上改 `~/.ssh/config` 里 `Host gpu4060` 对应的 `HostName`，例如：

```sshconfig
Host gpu4060
  HostName 192.168.1.xxx
  User ubuntu
```

然后继续用：

```bash
ssh gpu4060
```

不要因为换 IP 重建实验目录，也不要重新跑已经成功的配置。
