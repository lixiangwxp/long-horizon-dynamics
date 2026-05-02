# Full-State Smoke Tests

Run commands from the repository root unless a block starts with `cd scripts`.

## Compile Check

```bash
python -m compileall scripts
```

## NeuroBEM Full-State

```bash
cd scripts

python hdf5.py \
  --dataset neurobemfullstate \
  --history_length 20 \
  --unroll_length 10
```

```bash
python check_dataset_interface.py \
  --dataset neurobemfullstate \
  --history_length 20 \
  --unroll_length 10 \
  --batch_size 8 \
  --num_workers 0
```

```bash
WANDB_MODE=offline python train.py \
  --dataset neurobemfullstate \
  --model_type mlp \
  --predictor_type full_state \
  --history_length 20 \
  --unroll_length 10 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 0 \
  --accelerator cpu
```

```bash
WANDB_MODE=offline python eval.py \
  --dataset neurobemfullstate \
  --accelerator cpu
```

## PI-TCN Full-State

```bash
python hdf5.py \
  --dataset pitcnfullstate \
  --history_length 20 \
  --unroll_length 10
```

```bash
python check_dataset_interface.py \
  --dataset pitcnfullstate \
  --history_length 20 \
  --unroll_length 10 \
  --batch_size 8 \
  --num_workers 0
```

`pitcnfullstate` requires `p_x,p_y,p_z` full-state trajectory CSV columns. If
the current PI-TCN CSV is the old derivative-learning export, `hdf5.py` should
raise a clear `ValueError`. Do not fake position.

## NanoDrone Full-State

```bash
python hdf5.py \
  --dataset nanodronefullstate \
  --nanodrone_raw_path /Users/lixiang/Developer/nanodroneclone \
  --history_length 1 \
  --unroll_length 50
```

```bash
python check_dataset_interface.py \
  --dataset nanodronefullstate \
  --history_length 1 \
  --unroll_length 50 \
  --batch_size 8 \
  --num_workers 0
```

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

```bash
WANDB_MODE=offline python eval.py \
  --dataset nanodronefullstate \
  --accelerator cpu
```

`--nanodrone_raw_path` should point to the local NanoDrone raw CSV directory.
`hdf5.py` does not clone GitHub repositories and does not run `git lfs pull`.
Generated HDF5 files are written under `resources/data/nanodronefullstate/`.
`resources/data/` should not be committed.
