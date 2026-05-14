import csv
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import pytorch_lightning
import torch

from config import FULL_STATE_DATASETS, MODEL_TYPES, load_args, parse_args
from dynamics_learning.data import load_dataset
from dynamics_learning.device import select_device
from dynamics_learning.lighting import DynamicsLearning
from dynamics_learning.utils import check_folder_paths
from dynamics_learning.wandb_utils import log_experiment_artifact
from train import resolve_output_size

warnings.filterwarnings("ignore")

HORIZON_METRICS = ["E_p", "E_v", "E_q", "E_omega", "MSE_x"]


def parse_eval_horizons(value):
    if isinstance(value, str):
        return [int(item) for item in value.split(",") if item]
    return list(value)


def quat_geodesic_error(q_pred, q_true):
    q_pred = q_pred / q_pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_true = q_true / q_true.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_true_inv = torch.cat([q_true[..., :1], -q_true[..., 1:]], dim=-1)
    q_rel = quat_multiply(q_true_inv, q_pred)
    q_rel = q_rel / q_rel.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    vec_norm = torch.linalg.norm(q_rel[..., 1:], dim=-1)
    scalar = q_rel[..., 0].abs()
    return 2.0 * torch.atan2(vec_norm, scalar.clamp_min(1e-12))


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def compute_horizon_metrics(pred_future, y_future):
    if not torch.is_tensor(pred_future):
        pred_future = torch.as_tensor(pred_future)
    if not torch.is_tensor(y_future):
        y_future = torch.as_tensor(y_future)

    pred_future = pred_future.float()
    y_future = y_future.float()

    p_error = torch.linalg.norm(
        pred_future[:, :, 0:3] - y_future[:, :, 0:3], dim=-1
    ).mean(dim=0)
    v_error = torch.linalg.norm(
        pred_future[:, :, 3:6] - y_future[:, :, 3:6], dim=-1
    ).mean(dim=0)
    q_error = quat_geodesic_error(pred_future[:, :, 6:10], y_future[:, :, 6:10]).mean(
        dim=0
    )
    omega_error = torch.linalg.norm(
        pred_future[:, :, 10:13] - y_future[:, :, 10:13], dim=-1
    ).mean(dim=0)
    state_mse = ((pred_future - y_future) ** 2).sum(dim=-1).mean(dim=0)

    rows = []
    for horizon_idx in range(pred_future.shape[1]):
        rows.append(
            {
                "horizon": horizon_idx + 1,
                "E_p": float(p_error[horizon_idx].cpu()),
                "E_v": float(v_error[horizon_idx].cpu()),
                "E_q": float(q_error[horizon_idx].cpu()),
                "E_omega": float(omega_error[horizon_idx].cpu()),
                "MSE_x": float(state_mse[horizon_idx].cpu()),
            }
        )
    return rows


def summarize_horizon_metrics(rows, requested_horizons):
    summary = {}
    full_horizon = len(rows)

    for horizon in requested_horizons:
        if horizon > full_horizon:
            print(
                f"Warning: requested horizon h={horizon} but unroll_length={full_horizon}; "
                "skipping."
            )
            continue
        row = rows[horizon - 1]
        summary[f"h={horizon}"] = {name: row[name] for name in HORIZON_METRICS}

    summary["mean_1_to_F"] = {
        name: float(sum(row[name] for row in rows) / full_horizon)
        for name in HORIZON_METRICS
    }
    summary["sum_1_to_F"] = {
        name: float(sum(row[name] for row in rows)) for name in HORIZON_METRICS
    }
    summary["MSE_1_to_F"] = summary["mean_1_to_F"]["MSE_x"]
    return summary


def save_horizon_metrics(experiment_path, rows, summary):
    csv_path = os.path.join(experiment_path, "horizon_metrics.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["horizon", *HORIZON_METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(experiment_path, "horizon_summary.json")
    with open(json_path, "w") as file:
        json.dump(summary, file, indent=2)

    print("Saved horizon metrics to", csv_path)
    print("Saved horizon summary to", json_path)
    return csv_path, json_path


def requested_cli_names(argv):
    names = set()
    aliases = {"-N": "model_type", "-r": "run_id", "-d": "gpu_id", "-e": "epochs"}
    for item in argv[1:]:
        if item in aliases:
            names.add(aliases[item])
        elif item.startswith("--"):
            names.add(item.split("=", maxsplit=1)[0].lstrip("-").replace("-", "_"))
    return names


def _matches_if_requested(loaded, requested_args, requested_filters, name):
    if name not in requested_filters:
        return True
    requested_value = getattr(requested_args, name, None)
    if requested_value in (None, ""):
        return True
    return getattr(loaded, name, None) == requested_value


def experiment_matches_request(experiment_path, requested_args, requested_filters):
    args_path = os.path.join(experiment_path, "args.txt")
    if not os.path.isfile(args_path):
        return False
    if not glob.glob(os.path.join(experiment_path, "checkpoints", "*.pth")):
        return False
    try:
        loaded = load_args(args_path)
    except Exception:
        return False

    for name in ("dataset", "model_type", "history_length", "unroll_length"):
        if not _matches_if_requested(loaded, requested_args, requested_filters, name):
            return False
    for name in ("multi_step_delta_vomega", "multi_step_kinematic_update"):
        if not _matches_if_requested(loaded, requested_args, requested_filters, name):
            return False
    return True


def find_latest_experiment(resources_path, requested_args, requested_filters):
    experiments = sorted(
        glob.glob(os.path.join(resources_path, "experiments", "*/")),
        key=os.path.getctime,
        reverse=True,
    )
    matching = [
        experiment_path
        for experiment_path in experiments
        if experiment_matches_request(experiment_path, requested_args, requested_filters)
    ]
    if matching:
        return matching[0]
    raise FileNotFoundError(
        "No matching experiment with checkpoints found for "
        f"dataset={requested_args.dataset}, model_type={requested_args.model_type}, "
        f"history_length={requested_args.history_length}, "
        f"unroll_length={requested_args.unroll_length}. Train a matching model first."
    )


def find_checkpoint(experiment_path):
    experiment_dir = Path(experiment_path).resolve()
    checkpoint_dir = experiment_dir / "checkpoints"
    summary_path = experiment_dir / "train_summary.json"

    if summary_path.is_file():
        with open(summary_path, "r") as file:
            summary = json.load(file)
        best_model_path = summary.get("best_model_path")
        if best_model_path:
            best_path = Path(best_model_path).resolve()
            if best_path.is_file() and checkpoint_dir in best_path.parents:
                return str(best_path)

    checkpoint_paths = sorted(checkpoint_dir.glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {experiment_path}")

    model_paths = [path for path in checkpoint_paths if path.name != "last_model.pth"]

    def best_epoch_from_csv_logs():
        def version_key(item):
            match = re.search(r"version_(\\d+)", item.parent.name)
            return int(match.group(1)) if match else -1

        versions = sorted(
            experiment_dir.glob("csv_logs/version_*/metrics.csv"),
            key=version_key,
        )
        if not versions:
            return None

        metrics_path = versions[-1]
        best_epoch = None
        best_loss = None
        with metrics_path.open() as file:
            reader = csv.DictReader(file)
            for row in reader:
                loss_text = row.get("valid_loss_epoch")
                if not loss_text:
                    continue
                epoch_text = row.get("epoch")
                if epoch_text is None:
                    continue
                loss = float(loss_text)
                epoch = int(float(epoch_text))
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
        return best_epoch

    best_epoch = best_epoch_from_csv_logs()
    if best_epoch is not None:
        epoch_candidates = [
            path
            for path in model_paths
            if f"epoch={best_epoch:02d}" in path.name or f"epoch={best_epoch}" in path.name
        ]
        if epoch_candidates:
            return str(max(epoch_candidates, key=lambda item: item.stat().st_mtime))

    if model_paths:
        return str(max(model_paths, key=lambda item: item.stat().st_mtime))
    return str(max(checkpoint_paths, key=lambda item: item.stat().st_mtime))


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def max_predict_batches(args):
    limit = getattr(args, "limit_predict_batches", 0)
    if isinstance(limit, float):
        if limit <= 0:
            return None
        if limit < 1:
            raise ValueError("--limit_predict_batches must be an integer for eval.")
        return int(limit)
    if limit <= 0:
        return None
    return int(limit)


def run_prediction(model, dataloaders, device, max_batches=None):
    model.to(device)
    model.eval()
    loss_sum = 0.0
    metric_sums = None
    sample_count = 0
    seen_batches = 0

    with torch.no_grad():
        for dataloader in dataloaders:
            for batch_idx, batch in enumerate(dataloader):
                batch = move_batch_to_device(batch, device)
                output = model.predict_step(batch, batch_idx)
                pred_future = output["pred_future"].float()
                y_future = output["y_future"].float()
                batch_rows = compute_horizon_metrics(pred_future, y_future)
                batch_size = pred_future.shape[0]

                if metric_sums is None:
                    metric_sums = {
                        metric: torch.zeros(len(batch_rows), dtype=torch.float64)
                        for metric in HORIZON_METRICS
                    }
                for idx, row in enumerate(batch_rows):
                    for metric in metric_sums:
                        metric_sums[metric][idx] += row[metric] * batch_size

                sample_count += batch_size
                loss_sum += float(output["loss"].detach().cpu()) * batch_size
                seen_batches += 1
                if max_batches is not None and seen_batches >= max_batches:
                    break
            if max_batches is not None and seen_batches >= max_batches:
                break

    avg_loss = loss_sum / sample_count if sample_count else float("nan")
    if metric_sums is None or sample_count == 0:
        raise RuntimeError("No prediction batches were evaluated.")

    rows = []
    for horizon_idx in range(len(next(iter(metric_sums.values())))):
        row = {"horizon": horizon_idx + 1}
        for metric, values in metric_sums.items():
            row[metric] = float(values[horizon_idx] / sample_count)
        rows.append(row)
    return rows, torch.tensor(avg_loss), sample_count, seen_batches


def main(args, resources_path, data_path, experiment_path, model_path):
    args.experiment_path = experiment_path
    args.wandb_mode = os.environ.get("WANDB_MODE", getattr(args, "wandb_mode", "online"))
    os.environ["WANDB_MODE"] = args.wandb_mode

    device_config = select_device(args.accelerator, args.gpu_id, args.num_devices)
    args.device = str(device_config["device"])
    args.resolved_accelerator = device_config["resolved"]
    print("Evaluating model on", args.device, "\n")

    wandb_logger = pytorch_lightning.loggers.WandbLogger(
        name="wandb_logger",
        project="dynamics_learning",
        save_dir=experiment_path,
    )
    csv_logger = pytorch_lightning.loggers.CSVLogger(
        save_dir=experiment_path, name="csv_logs"
    )

    hdf5_files = sorted(glob.glob(os.path.join(data_path, "test", "*.h5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No test HDF5 files found under {data_path}/test")

    test_datasets = []
    test_dataloaders = []
    for hdf5_path in hdf5_files:
        dataset, dataloader = load_dataset(
            "test",
            os.path.dirname(hdf5_path),
            os.path.basename(hdf5_path),
            args,
            num_workers=0,
            pin_memory=device_config["pin_memory"],
        )
        test_datasets.append(dataset)
        test_dataloaders.append(dataloader)

    input_size = test_datasets[0].state_dim + test_datasets[0].control_dim
    output_size = resolve_output_size(args)
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=input_size,
        output_size=output_size,
        max_iterations=1,
    )

    try:
        checkpoint = torch.load(
            model_path, map_location=device_config["device"], weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device_config["device"])
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    rows, avg_loss, sample_count, seen_batches = run_prediction(
        model,
        test_dataloaders,
        device_config["device"],
        max_batches=max_predict_batches(args),
    )
    print("Average rollout loss:", float(avg_loss))
    print("Evaluated samples:", sample_count)
    print("Evaluated batches:", seen_batches)

    requested_horizons = parse_eval_horizons(
        getattr(args, "eval_horizons", "1,10,25,50")
    )
    summary = summarize_horizon_metrics(rows, requested_horizons)
    summary.update(
        {
            "dataset": args.dataset,
            "model_type": args.model_type,
            "history_length": args.history_length,
            "unroll_length": args.unroll_length,
        }
    )
    csv_path, json_path = save_horizon_metrics(experiment_path, rows, summary)

    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config.update({"evaluated_checkpoint": model_path})
    log_experiment_artifact(
        wandb_logger,
        experiment_path,
        artifact_prefix=f"eval-{args.dataset}-{args.model_type}",
        include_checkpoints=False,
        extra_files=[model_path, csv_path, json_path],
    )
    _ = csv_logger


if __name__ == "__main__":
    requested_filters = requested_cli_names(sys.argv)
    requested_args = parse_args()
    requested_accelerator = requested_args.accelerator

    if requested_args.model_type not in MODEL_TYPES:
        raise ValueError(f"Model type must be one of {MODEL_TYPES}")
    if requested_args.dataset not in FULL_STATE_DATASETS:
        raise ValueError(f"Dataset must be one of {FULL_STATE_DATASETS}")

    pytorch_lightning.seed_everything(requested_args.seed)

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    if requested_args.experiment_path:
        experiment_path = os.path.abspath(requested_args.experiment_path)
    else:
        experiment_path = find_latest_experiment(
            resources_path, requested_args, requested_filters
        )
    model_path = find_checkpoint(experiment_path)

    print("Experiment path:", experiment_path)
    print("Evaluating Dynamics model:", model_path)

    args = load_args(os.path.join(experiment_path, "args.txt"))
    args.accelerator = requested_accelerator
    args.eval_horizons = getattr(requested_args, "eval_horizons", "1,10,25,50")
    args.wandb_mode = getattr(requested_args, "wandb_mode", getattr(args, "wandb_mode", "online"))
    args.eval_batch_size = getattr(requested_args, "eval_batch_size", 0)
    if args.eval_batch_size:
        args.batch_size = args.eval_batch_size
    if not hasattr(args, "nanodrone_raw_path"):
        args.nanodrone_raw_path = requested_args.nanodrone_raw_path
    if not hasattr(args, "lambda_p"):
        args.lambda_p = 1.0
        args.lambda_v = 1.0
        args.lambda_q = 1.0
        args.lambda_omega = 1.0
    if not hasattr(args, "multi_step_delta_vomega"):
        args.multi_step_delta_vomega = False
    if not hasattr(args, "multi_step_kinematic_update"):
        args.multi_step_kinematic_update = False

    data_path = os.path.join(resources_path, "data", args.dataset)
    check_folder_paths(
        [
            os.path.join(experiment_path, "plots"),
            os.path.join(experiment_path, "plots", "trajectory"),
            os.path.join(experiment_path, "plots", "testset"),
        ]
    )

    main(args, resources_path, data_path, experiment_path, model_path)
