# Full-State Smoke Tests

Run all commands from `scripts/`.

## 1. Generate NeuroBEM Full-State HDF5

```bash
cd /Users/lixiang/Developer/long-horizon-dynamics/scripts

python hdf5.py \
  --dataset neurobemfullstate \
  --history_length 20 \
  --unroll_length 10
```

## 2. Generate PI-TCN Full-State HDF5

```bash
python hdf5.py \
  --dataset pitcnfullstate \
  --history_length 20 \
  --unroll_length 10
```

`pitcnfullstate` requires full-state trajectory CSV files with `p_x,p_y,p_z`, `v_x,v_y,v_z`, `q_w,q_x,q_y,q_z`, `w_x,w_y,w_z`, and either `u_0..u_3` or `f_0..f_3`.

If the current PI-TCN CSV is the old derivative-learning export, regenerate it from raw bags first. Do not use `appendHistory`, do not save only `data_columns=[v,q,w,f]`, and do not fake position columns. Save per-trajectory time series such as:

```text
t,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z,w_x,w_y,w_z,u_0,u_1,u_2,u_3
```

or, for thrust control:

```text
t,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z,w_x,w_y,w_z,f_0,f_1,f_2,f_3
```

## 3. Check NeuroBEM Dataloader

```bash
python check_dataset_interface.py \
  --dataset neurobemfullstate \
  --history_length 20 \
  --unroll_length 10 \
  --batch_size 8 \
  --num_workers 0
```

## 4. Check PI-TCN Dataloader

```bash
python check_dataset_interface.py \
  --dataset pitcnfullstate \
  --history_length 20 \
  --unroll_length 10 \
  --batch_size 8 \
  --num_workers 0
```

## 5. NeuroBEM Full-State MLP CPU Smoke Test

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

## 6. NeuroBEM Full-State TCN CPU Smoke Test

```bash
WANDB_MODE=offline python train.py \
  --dataset neurobemfullstate \
  --model_type tcn \
  --predictor_type full_state \
  --history_length 20 \
  --unroll_length 10 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 0 \
  --accelerator cpu
```

## 7. PI-TCN Full-State MLP CPU Smoke Test

Run this after `pitcnfullstate` data is ready:

```bash
WANDB_MODE=offline python train.py \
  --dataset pitcnfullstate \
  --model_type mlp \
  --predictor_type full_state \
  --history_length 20 \
  --unroll_length 10 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 0 \
  --accelerator cpu
```

## 8. Eval

```bash
WANDB_MODE=offline python eval.py \
  --dataset neurobemfullstate \
  --accelerator cpu
```
