import os
import sys
import time
import warnings

import pytorch_lightning
from torch.utils.data import DataLoader

from config import parse_args, save_args
from dynamics_learning.data import load_dataset
from dynamics_learning.device import select_device
from dynamics_learning.lighting import DynamicsLearning
from dynamics_learning.utils import check_folder_paths
from dynamics_learning.wandb_utils import log_experiment_artifact

warnings.filterwarnings("ignore")


FULL_STATE_DATASETS = ["neurobemfullstate", "pitcnfullstate"]


def main(args):
    if args.predictor_type != "full_state":
        raise ValueError("This branch is for the new full_state task only.")

    device_config = select_device(args.accelerator, args.gpu_id, args.num_devices)
    args.device = str(device_config["device"])
    args.resolved_accelerator = device_config["resolved"]
    print("Training model on", args.device, "\n")
    save_args(args, os.path.join(experiment_path, "args.txt"))

    wandb_logger = pytorch_lightning.loggers.WandbLogger(
        name="wandb_logger",
        project="dynamics_learning",
        save_dir=experiment_path,
        log_model="all",
    )
    csv_logger = pytorch_lightning.loggers.CSVLogger(save_dir=experiment_path, name="csv_logs")
    loggers = [wandb_logger, csv_logger]

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor="best_valid_loss",
        dirpath=os.path.join(experiment_path, "checkpoints"),
        filename="model-{epoch:02d}-{best_valid_loss:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last_model"
    checkpoint_callback.FILE_EXTENSION = ".pth"

    train_dataset, train_loader = load_dataset(
        "training",
        data_path + "train/",
        "train.h5",
        args,
        num_workers=args.num_workers,
        pin_memory=device_config["pin_memory"],
    )
    valid_dataset, valid_loader = load_dataset(
        "validation",
        data_path + "valid/",
        "valid.h5",
        args,
        num_workers=args.num_workers,
        pin_memory=device_config["pin_memory"],
    )

    input_size = train_dataset.state_dim + train_dataset.control_dim
    output_size = 12

    print("Loading full_state model ...")
    model = DynamicsLearning(
        args,
        resources_path,
        experiment_path,
        input_size=input_size,
        output_size=output_size,
        max_iterations=train_dataset.num_steps * args.epochs,
    )

    trainer = pytorch_lightning.Trainer(
        accelerator=device_config["lightning_accelerator"],
        devices=device_config["devices"],
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_freq,
        default_root_dir=experiment_path,
        logger=loggers,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
    )
    if trainer.is_global_zero:
        wandb_logger.experiment.config.update(vars(args))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if trainer.is_global_zero:
        log_experiment_artifact(
            wandb_logger,
            experiment_path,
            artifact_prefix=f"train-{args.dataset}-{args.model_type}",
            include_checkpoints=True,
        )


if __name__ == "__main__":
    args = parse_args()

    if args.model_type not in ["mlp", "lstm", "gru", "tcn"]:
        raise ValueError("Model type must be one of [mlp, lstm, gru, tcn]")
    if args.dataset not in FULL_STATE_DATASETS:
        raise ValueError("Dataset must be one of [neurobemfullstate, pitcnfullstate]")
    if args.predictor_type != "full_state":
        raise ValueError("This branch is for the new full_state task only.")

    pytorch_lightning.seed_everything(args.seed)

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + args.dataset + "/"
    experiment_path = resources_path + "experiments/" + time.strftime("%Y%m%d-%H%M%S") + "_" + str(args.run_id) + "/"

    check_folder_paths(
        [
            os.path.join(experiment_path, "checkpoints"),
            os.path.join(experiment_path, "plots"),
            os.path.join(experiment_path, "plots", "trajectory"),
            os.path.join(experiment_path, "plots", "testset"),
        ]
    )

    main(args)
