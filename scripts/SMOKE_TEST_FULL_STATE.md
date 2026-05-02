# Full-State Smoke Tests

Run these commands from the repository root:

```bash
cd scripts
```

Generate NanoDrone HDF5:

```bash
python hdf5.py \
  --dataset nanodronefullstate \
  --nanodrone_raw_path /Users/lixiang/Developer/nanodroneclone \
  --history_length 1 \
  --unroll_length 50
```

`--nanodrone_raw_path` should point to the local NanoDrone raw CSV directory.
`hdf5.py` does not clone GitHub repositories and does not run `git lfs pull`.
Generated HDF5 files are written under `resources/data/nanodronefullstate/`.
`resources/data/` should not be committed.

Run an MLP smoke test:

```bash
WANDB_MODE=offline python train.py \
  --dataset nanodronefullstate \
  --model_type mlp \
  --predictor_type full_state \
  --history_length 1 \
  --unroll_length 50 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 0 \
  --accelerator cpu
```

Run a TCN smoke test:

```bash
WANDB_MODE=offline python train.py \
  --dataset nanodronefullstate \
  --model_type tcn \
  --predictor_type full_state \
  --history_length 20 \
  --unroll_length 50 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 0 \
  --accelerator cpu
```

Evaluate the latest checkpoint for the requested dataset:

```bash
WANDB_MODE=offline python eval.py \
  --dataset nanodronefullstate \
  --accelerator cpu
```

`history_length=1` matches the NanoDrone official current-state initialized
setting. `history_length=20` or `history_length=50` are for later
hidden-history methods. This smoke test only covers data integration,
full-state baselines, and horizon-wise evaluation.
