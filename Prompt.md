# Long-Horizon Dynamics Durable Memory

This file is the project-level durable memory for Codex. Read this together with
`AGENTS.md` before running experiments or changing code. Do not store secrets here.

Last updated: 2026-05-06 CST.

## Operating Rules

- Mac local repo is the source of truth for code.
- Remote GPU host alias: `gpu4060`.
- Remote repo: `/home/ubuntu/Developer/long-horizon-dynamics`.
- Remote conda env: `dynamics_learning`.
- Active tmux session: `neurobem_sweep`.
- Active sweep root:
  `/home/ubuntu/Developer/long-horizon-dynamics/resources/experiments/neurobem_fullstate_fast_20260504-154358_wandb_online`.
- Do not delete user files.
- For Linux/SSH commands, explain what will run, why, and the important syntax in Chinese.
- After code changes: run syntax checks, sync to `gpu4060`, and verify local/remote hashes for touched code.
- After each completed training/eval config: update this file with the config id, result summary, analysis, and any failure/fix notes.

## Current Sweep Protocol

- Dataset: `neurobemfullstate`.
- Predictor: `full_state`.
- Models: `mlp`, `gru`, `tcn`, `tcnlstm`, `grutcn`.
- History lengths: `1, 10, 20, 50`.
- Unroll length: `50`.
- Seed: `10`.
- Current fast-sweep run settings:
  - `epochs=200`
  - `early_stopping=True`
  - `patience=20`
  - `min_delta=1e-4`
  - `limit_train_batches=0.25`
  - `limit_val_batches=0.5`
  - `limit_predict_batches=0`
  - `shuffle=True`
  - OOM/VRAM-safe batch ladder: `128/4`, `64/8`, `32/16`
  - Effective batch size stays `512`
  - W&B mode: prefer online; fallback offline if online fails.

## Code Changes Already Made

- Added early stopping and gradient accumulation support to training.
- Added `--experiment_path` support so train/eval target exact experiment directories.
- Added `--resume_from_checkpoint` support.
- Added `scripts/run_neurobem_sweep.sh` for sweep execution, OOM retry, W&B fallback, per-config status, and aggregation.
- Added `scripts/aggregate_horizon_results.py` for combined CSV and horizon plots.
- Fixed aggregation to tolerate NUL bytes in interrupted CSV logs.
- Fixed non-finite validation handling:
  - `scripts/dynamics_learning/lighting.py` now propagates non-finite validation loss into `best_valid_loss`, so early stopping can stop NaN runs.
  - `scripts/run_neurobem_sweep.sh` avoids resuming from `last_model.pth` if the latest train log contains non-finite loss, and writes rerun logs without overwriting old logs.

## Completed Results Snapshot

Source: remote `horizon_results.csv`, read on 2026-05-06.

| config | status | stopped_epoch | micro/accum/eff | best_valid_loss | h1_E_q | h50_E_q | h50_E_v | h50_E_omega | mean_E_q | sum_E_q | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gru_H1_F50_seed10 | success | 85 | 512/1/512 | 0.577148 | 0.004452 | 0.0864979 | 0.362896 | 0.280261 | 0.0466376 | 2.33188 | GRU strong long-horizon baseline. |
| gru_H10_F50_seed10 | success | 84 | 512/1/512 | 0.580389 | 0.00266536 | 0.0800042 | 0.356014 | 0.283807 | 0.0426877 | 2.13439 | Best current `h50_E_q`. |
| gru_H20_F50_seed10 | success | 109 | 256/2/512 | 0.573358 | 0.00264079 | 0.0802197 | 0.353015 | 0.260392 | 0.0420377 | 2.10188 | Best current `h50_E_v`, `h50_E_omega`, `mean_E_q`, `sum_E_q`. |
| gru_H50_F50_seed10 | success | 70 | 128/4/512 | 0.595215 | 0.00257132 | 0.0817253 | 0.380075 | 0.278438 | 0.0435237 | 2.17619 | Completed after lowering micro batch to 128 and using accum 4. |
| mlp_H1_F50_seed10 | success | 166 | 512/1/512 | 0.704774 | 0.00188335 | 0.174414 | 0.588885 | 0.579955 | 0.0847114 | 4.23557 | Best current `h1_E_q` and strong immediate metrics, but weak long horizon. |
| mlp_H10_F50_seed10 | success | 200 | 512/1/512 | 0.79586 | 0.58337 | 0.183214 | 0.576795 | 0.557843 | 0.124004 | 6.2002 | `h1_E_q` anomaly/high. |
| mlp_H20_F50_seed10 | success | 200 | 512/1/512 | 0.808862 | 1.16404 | 0.171745 | 0.592367 | 0.524423 | 0.141003 | 7.05017 | `h1_E_q` anomaly/high. |
| mlp_H50_F50_seed10 | success | 200 | 512/1/512 | 0.713171 | 0.00245296 | 0.158459 | 0.592678 | 0.533048 | 0.0759609 | 3.79805 | Best MLP long-horizon quaternion, still weaker than GRU. |
| tcn_H1_F50_seed10 | success | 37 | 128/4/512 | 1.20341 | 1.01797 | 0.26683 | 0.902589 | 0.690581 | 0.25628 | 12.814 | Completed after NaN intervention; weak long-horizon result. |

## Current In-Progress / Pending

- `tcn_H10_F50_seed10`: currently in progress on remote.
  - Latest CSV row at last read: `run_status` empty, `eval_status=missing_eval`.
  - Current best_valid_loss in CSV snapshot: `0.614203`.
  - Current tmux process was training `tcn_H10` from a resume checkpoint.
- Remaining expected configs after TCN:
  - `tcn_H20`, `tcn_H50`
  - `tcnlstm_H1`, `tcnlstm_H10`, `tcnlstm_H20`, `tcnlstm_H50`
  - `grutcn_H1`, `grutcn_H10`, `grutcn_H20`, `grutcn_H50`

## Analysis So Far

- GRU is the current strongest family for long-horizon full-state prediction.
- Best current long-horizon quaternion (`h50_E_q`) is `gru_H10`.
- Best current velocity and angular velocity at horizon 50 are `gru_H20`.
- `gru_H50` is close to `gru_H10/H20` on quaternion and angular velocity but worse on `h50_E_v`.
- MLP can be strong at `h1`, especially `mlp_H1`, but long-horizon errors are much worse than GRU.
- MLP `H10/H20` have obvious `h1_E_q` anomalies and should not be trusted for immediate quaternion behavior without investigation.
- Plain TCN `H1` is much weaker than GRU and MLP on the tracked metrics. Continue sweep for TCN history lengths before drawing final conclusions about the whole TCN family.

## Failure / Intervention Log

- Remote machine previously froze under high VRAM pressure with `gru_H50` at batch 256 / accum 2.
  - Action: restarted sweep with batch ladder `128,64,32` and accum ladder `4,8,16`.
  - Result: `gru_H50` completed successfully with micro batch 128, accum 4.
- `tcn_H1` showed non-finite (`nan`) train/validation loss during training.
  - Observation: because `best_valid_loss` retained the previous finite best value, early stopping could miss NaN drift.
  - Action: changed validation epoch handling to propagate non-finite validation loss to `best_valid_loss`.
  - Action: changed sweep script to avoid resuming from a latest log containing non-finite loss and to preserve old logs with rerun suffixes.
  - Result: `tcn_H1` now has `status=success`; no need to rerun it.

## Commands To Remember

```bash
ssh gpu4060
cd /home/ubuntu/Developer/long-horizon-dynamics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynamics_learning
tmux attach -t neurobem_sweep
```

- Detach from tmux without stopping training: press `Ctrl-b`, then `d`.
- Do not press `Ctrl-c` inside tmux unless intentionally stopping training.

Useful checks:

```bash
tmux ls
/usr/lib/wsl/lib/nvidia-smi
ps -eo pid,ppid,stat,etime,cmd | grep -E "train.py|run_neurobem" | grep -v grep
tail -n 50 resources/experiments/<sweep_id>/<config>/logs/<log_file>.log
```

## Per-Config Update Template

Append one block like this whenever a training/eval config finishes:

```markdown
### <config_id> - <YYYY-MM-DD HH:mm CST>

- Status:
- Train log:
- Eval log:
- Batch / accumulation / effective batch:
- Stopped epoch:
- W&B mode:
- Best valid loss:
- Key metrics:
  - h1:
  - h10:
  - h25:
  - h50:
  - mean_1_to_F:
  - sum_1_to_F:
- Analysis:
- Failure/fix notes:
- Files produced:
```

## Documentation Suggestions

- Keep `AGENTS.md` for durable project rules and behavioral constraints.
- Keep this `Prompt.md` for evolving memory: active status, completed results, decisions, and intervention history.
- Add `Plan.md` if the experiment roadmap becomes multi-phase; put future experiment design and acceptance criteria there.
- Add `Documentation.md` or `docs/experiment_workflow.md` if you want a stable human-facing guide for sync, tmux, W&B, and result interpretation.
- Avoid spreading the same rule across many files. Put rules in `AGENTS.md`; put evolving state here.
