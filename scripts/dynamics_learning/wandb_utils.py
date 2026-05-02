import re
from pathlib import Path

import wandb


def _safe_artifact_name(name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name)
    return safe_name.strip("-.") or "dynamics-learning-run"


def _has_files(path):
    return path.exists() and any(item.is_file() for item in path.rglob("*"))


def log_experiment_artifact(
    wandb_logger,
    experiment_path,
    *,
    artifact_prefix,
    include_checkpoints=True,
    extra_files=None,
):
    run = wandb_logger.experiment
    run_id = getattr(run, "id", None) or "run"
    artifact_name = _safe_artifact_name(f"{artifact_prefix}-{run_id}")
    artifact = wandb.Artifact(artifact_name, type="experiment")

    experiment_dir = Path(experiment_path)
    added = False

    args_path = experiment_dir / "args.txt"
    if args_path.exists():
        artifact.add_file(str(args_path), name="args.txt")
        added = True

    if include_checkpoints:
        checkpoint_dir = experiment_dir / "checkpoints"
        if _has_files(checkpoint_dir):
            artifact.add_dir(str(checkpoint_dir), name="checkpoints")
            added = True

    for folder_name in ("csv_logs", "plots", "plotting_data"):
        folder_path = experiment_dir / folder_name
        if _has_files(folder_path):
            artifact.add_dir(str(folder_path), name=folder_name)
            added = True

    for file_path in extra_files or []:
        file_path = Path(file_path)
        if file_path.exists():
            try:
                artifact_path = file_path.relative_to(experiment_dir).as_posix()
            except ValueError:
                artifact_path = f"extra/{file_path.name}"
            artifact.add_file(str(file_path), name=artifact_path)
            added = True

    if added:
        run.log_artifact(artifact)
