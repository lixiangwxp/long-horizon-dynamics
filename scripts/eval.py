import glob
import os
import sys
import time
import warnings
from pathlib import Path

import pytorch_lightning
import torch

from config import load_args, parse_args
from dynamics_learning.data import load_dataset
from dynamics_learning.device import select_device
from dynamics_learning.lighting import DynamicsLearning
from dynamics_learning.utils import check_folder_paths
from dynamics_learning.wandb_utils import log_experiment_artifact

warnings.filterwarnings("ignore")


FULL_STATE_DATASETS = ["neurobemfullstate", "pitcnfullstate"]


def find_latest_experiment_with_checkpoint(resources_path):
    candidates = []
    for experiment_dir in glob.glob(resources_path + "experiments/*/"):
        args_path = os.path.join(experiment_dir, "args.txt")
        checkpoints = glob.glob(os.path.join(experiment_dir, "checkpoints", "*.pth"))
        if os.path.exists(args_path) and checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            candidates.append((os.path.getctime(latest_checkpoint), experiment_dir, latest_checkpoint))

    if not candidates:
        raise FileNotFoundError(
            f"No experiment with args.txt and checkpoints/*.pth found in {resources_path + 'experiments/'}"
        )

    _, experiment_dir, latest_checkpoint = max(candidates, key=lambda item: item[0])
    return experiment_dir, latest_checkpoint


def main(args, hdf5_files, model_path):
    if args.predictor_type != "full_state":
        raise ValueError("This branch is for the new full_state task only.")

    device_config = select_device(args.accelerator, args.gpu_id, args.num_devices)
    args.device = str(device_config["device"])
    args.resolved_accelerator = device_config["resolved"]
    print("Testing model on", args.device, "\n")

    wandb_logger = pytorch_lightning.loggers.WandbLogger(
        name="wandb_logger",
        project="dynamics_learning",
        save_dir=experiment_path,
    )
    csv_logger = pytorch_lightning.loggers.CSVLogger(save_dir=experiment_path, name="csv_logs")
    loggers = [wandb_logger, csv_logger]

    dataloaders = {}
    first_dataset = None
    for hdf5_file in hdf5_files:
        hdf5_file = Path(hdf5_file)
        dataset, dataloader = load_dataset(
            "test",
            str(hdf5_file.parent) + "/",
            hdf5_file.name,
            args,
            num_workers=0,
            pin_memory=False,
        )
        if first_dataset is None:
            first_dataset = dataset
        dataloaders[hdf5_file.stem] = dataloader

    input_size = first_dataset.state_dim + first_dataset.control_dim
    output_size = 12

    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=input_size,
        output_size=output_size,
        max_iterations=1,
    )

    try:
        checkpoint = torch.load(model_path, map_location=device_config["device"], weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device_config["device"])
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pytorch_lightning.Trainer(
        accelerator=device_config["lightning_accelerator"],
        devices=device_config["devices"],
        logger=loggers,
        max_epochs=0,
    )

    trainer.test(model, list(dataloaders.values()), verbose=True)
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(vars(args))
        wandb_logger.experiment.config.update({"evaluated_checkpoint": str(model_path)})
        log_experiment_artifact(
            wandb_logger,
            experiment_path,
            artifact_prefix=f"eval-{args.dataset}-{args.model_type}",
            include_checkpoints=False,
            extra_files=[model_path],
        )

    print("Evaluated dataloaders:", dataloaders.keys())


if __name__ == "__main__":
    requested_args = parse_args()
    requested_accelerator = requested_args.accelerator

    if requested_args.model_type not in ["mlp", "lstm", "gru", "tcn"]:
        raise ValueError("Model type must be one of [mlp, lstm, gru, tcn]")

    pytorch_lightning.seed_everything(requested_args.seed)

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    experiment_path, model_path = find_latest_experiment_with_checkpoint(resources_path)

    print("Testing Dynamics model:", model_path)
    args = load_args(experiment_path + "args.txt")
    args.accelerator = requested_accelerator

    if args.dataset not in FULL_STATE_DATASETS:
        raise ValueError("Dataset must be one of [neurobemfullstate, pitcnfullstate]")
    if args.predictor_type != "full_state":
        raise ValueError("This branch is for the new full_state task only.")

    data_path = resources_path + "data/" + args.dataset + "/"
    hdf5_files = sorted(glob.glob(data_path + "test/*.h5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No test HDF5 files found in {data_path + 'test/*.h5'}")

    eval_path = experiment_path + "eval-" + time.strftime("%Y%m%d-%H%M%S") + "/"
    check_folder_paths(
        [
            os.path.join(eval_path, "plots"),
            os.path.join(eval_path, "plots", "trajectory"),
            os.path.join(eval_path, "plots", "testset"),
        ]
    )
    experiment_path = eval_path

    main(args, hdf5_files, model_path)
