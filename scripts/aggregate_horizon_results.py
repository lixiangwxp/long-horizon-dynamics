import argparse
import csv
import glob
import json
import math
from pathlib import Path


METRICS = ("E_p", "E_v", "E_q", "E_omega")
FRONT_FIELDS = [
    "sweep_root",
    "experiment_name",
    "experiment_path",
    "run_status",
    "eval_status",
    "failure_reason",
    "retry_count",
    "dataset",
    "model_type",
    "predictor_type",
    "history_length",
    "unroll_length",
    "seed",
    "run_id",
    "epochs",
    "max_epochs",
    "stopped_epoch",
    "early_stopping",
    "early_stopped",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "limit_train_batches",
    "limit_val_batches",
    "limit_predict_batches",
    "micro_batch_size",
    "batch_size",
    "accumulate_grad_batches",
    "effective_batch_size",
    "wandb_mode",
    "learning_rate",
    "lambda_p",
    "lambda_v",
    "lambda_q",
    "lambda_omega",
    "horizon_count",
    "max_horizon",
    "best_valid_loss",
    "best_model_score",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate horizon metrics.")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("resources/experiments"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/experiments/horizon_results.csv"),
    )
    parser.add_argument("--plots-dir", type=Path, default=None)
    parser.add_argument("--include-missing", action="store_true")
    return parser.parse_args()


def coerce_value(type_name, raw_value):
    if type_name == "int":
        return int(raw_value)
    if type_name == "float":
        return float(raw_value)
    if type_name == "bool":
        return raw_value == "True"
    if type_name == "list":
        value = raw_value.strip()
        if value == "[]":
            return []
        return [int(element) for element in value[1:-1].split(", ")]
    if type_name == "tuple":
        value = raw_value.strip()
        if value == "()":
            return ()
        return tuple(int(element) for element in value[1:-1].split(", "))
    return raw_value


def read_args_file(path):
    values = {}
    with open(path, "r") as file:
        for line in file:
            name, type_name, raw_value = line.rstrip("\n").split(";", maxsplit=2)
            values[name] = coerce_value(type_name, raw_value)
    return values


def read_json(path):
    if not path.is_file():
        return {}
    with open(path, "r") as file:
        return json.load(file)


def read_status(path):
    status = read_json(path)
    values = {
        "run_status": status.get("status", ""),
        "failure_reason": status.get("failure_reason", ""),
        "retry_count": status.get("retry_count", ""),
        "wandb_mode": status.get("wandb_mode", ""),
        "micro_batch_size": status.get("batch_size", ""),
        "accumulate_grad_batches": status.get("accumulate_grad_batches", ""),
        "effective_batch_size": status.get("effective_batch_size", ""),
    }
    return {key: value for key, value in values.items() if value not in ("", None)}


def read_horizon_metrics(path):
    if not path.is_file():
        return []
    with open(path, "r", newline="") as file:
        return list(csv.DictReader(file))


def read_latest_metric(experiment_dir, metric_name):
    metrics_paths = sorted(
        glob.glob(str(experiment_dir / "csv_logs" / "version_*" / "metrics.csv"))
    )
    values = []
    for path in metrics_paths:
        with open(path, "r", newline="") as file:
            for row in csv.DictReader(file):
                raw_value = row.get(metric_name)
                if raw_value in (None, ""):
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
    if not values:
        return ""
    if metric_name == "best_valid_loss":
        return min(values)
    return values[-1]


def add_summary_metrics(row, summary):
    for key, value in summary.items():
        if key.startswith("h="):
            horizon = key.split("=", maxsplit=1)[1]
            for metric in METRICS:
                row[f"h{horizon}_{metric}"] = value.get(metric, "")
        elif key in {"mean_1_to_F", "sum_1_to_F"}:
            for metric in METRICS:
                row[f"{key}_{metric}"] = value.get(metric, "")


def experiment_row(args_path, experiments_root):
    experiment_dir = args_path.parent
    args_values = read_args_file(args_path)
    summary = read_json(experiment_dir / "horizon_summary.json")
    train_summary = read_json(experiment_dir / "train_summary.json")
    status_values = read_status(experiment_dir / "status.json")
    horizon_rows = read_horizon_metrics(experiment_dir / "horizon_metrics.csv")

    row = {
        "sweep_root": str(experiments_root),
        "experiment_name": experiment_dir.name,
        "experiment_path": str(experiment_dir),
        "eval_status": "success" if summary else "missing_eval",
        "horizon_count": len(horizon_rows),
        "max_horizon": horizon_rows[-1]["horizon"] if horizon_rows else "",
    }
    row.update(status_values)

    for name in FRONT_FIELDS:
        if name in row:
            continue
        if name in train_summary:
            row[name] = train_summary[name]
        elif name in args_values:
            row[name] = args_values[name]
        else:
            row[name] = ""

    if not row.get("micro_batch_size"):
        row["micro_batch_size"] = row.get("batch_size", "")
    if not row.get("effective_batch_size"):
        batch_size = row.get("batch_size")
        accumulate = row.get("accumulate_grad_batches") or 1
        if batch_size != "":
            row["effective_batch_size"] = int(batch_size) * int(accumulate)

    if not row.get("best_valid_loss"):
        row["best_valid_loss"] = read_latest_metric(experiment_dir, "best_valid_loss")

    add_summary_metrics(row, summary)
    return row


def collect_rows(experiments_root, include_missing):
    args_paths = sorted(experiments_root.rglob("args.txt"))
    rows = []
    for args_path in args_paths:
        row = experiment_row(args_path, experiments_root)
        if include_missing or row["eval_status"] == "success":
            rows.append(row)
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_fields = sorted(
        {field for row in rows for field in row.keys()} - set(FRONT_FIELDS)
    )
    fieldnames = FRONT_FIELDS + [
        field for field in dynamic_fields if field not in FRONT_FIELDS
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_horizon_curves(rows, plots_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib failed to import: {exc}")
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False
        for row in rows:
            metrics_path = Path(row["experiment_path"]) / "horizon_metrics.csv"
            horizon_rows = read_horizon_metrics(metrics_path)
            if not horizon_rows:
                continue
            horizons = [int(item["horizon"]) for item in horizon_rows]
            values = [float(item[metric]) for item in horizon_rows]
            label = f"{row.get('model_type', '')} H={row.get('history_length', '')}"
            ax.plot(horizons, values, label=label, linewidth=1.5)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}(h)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        output_path = plots_dir / f"{metric}_horizon_curve.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        written.append(output_path)
    return written


def main():
    args = parse_args()
    rows = collect_rows(args.experiments_root, args.include_missing)
    output_path = write_csv(rows, args.output)
    plots_dir = args.plots_dir or output_path.parent / "horizon_curves"
    plot_paths = plot_horizon_curves(rows, plots_dir)
    print(f"Wrote {len(rows)} rows to {output_path}")
    for path in plot_paths:
        print(f"Wrote plot {path}")


if __name__ == "__main__":
    main()
