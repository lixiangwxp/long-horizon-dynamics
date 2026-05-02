import os
import sys

import torch

from config import FULL_STATE_DATASETS, parse_args
from dynamics_learning.data import load_dataset


def assert_finite_batch(batch):
    for key, value in batch.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError(f"Batch tensor {key} contains non-finite values.")


def assert_quaternion_norm(batch, atol=1e-3):
    for key in ("x_hist", "y_future"):
        q = batch[key][..., 6:10]
        norm = torch.linalg.norm(q, dim=-1)
        expected = torch.ones_like(norm)
        if not torch.allclose(norm, expected, atol=atol, rtol=atol):
            max_error = torch.max(torch.abs(norm - expected)).item()
            raise ValueError(
                f"{key} quaternion norm check failed. "
                f"max |norm(q)-1| = {max_error:.6g}"
            )


def check_batch_shapes(dataset, batch):
    if batch["x_hist"].shape[-1] != 13:
        raise ValueError(f"x_hist last dim should be 13, got {batch['x_hist'].shape}")
    if batch["u_hist"].shape[-1] != 4:
        raise ValueError(f"u_hist last dim should be 4, got {batch['u_hist'].shape}")
    if batch["u_roll"].shape[-1] != 4:
        raise ValueError(f"u_roll last dim should be 4, got {batch['u_roll'].shape}")
    if batch["y_future"].shape[-1] != 13:
        raise ValueError(
            f"y_future last dim should be 13, got {batch['y_future'].shape}"
        )
    if dataset.state_dim != 13:
        raise ValueError(f"dataset.state_dim should be 13, got {dataset.state_dim}")
    if dataset.control_dim != 4:
        raise ValueError(f"dataset.control_dim should be 4, got {dataset.control_dim}")


def check_control_alignment(batch, atol=1e-6):
    if not torch.allclose(
        batch["u_hist"][:, -1, :], batch["u_roll"][:, 0, :], atol=atol
    ):
        max_error = torch.max(
            torch.abs(batch["u_hist"][:, -1, :] - batch["u_roll"][:, 0, :])
        ).item()
        raise ValueError(
            "Control alignment check failed: u_hist[:, -1, :] must equal "
            f"u_roll[:, 0, :]. max error = {max_error:.6g}"
        )


def main():
    args = parse_args()
    if args.dataset not in FULL_STATE_DATASETS:
        raise ValueError(f"Dataset must be one of {FULL_STATE_DATASETS}")

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = os.path.join(resources_path, "data", args.dataset, "train")
    hdf5_file = "train.h5"
    hdf5_path = os.path.join(data_path, hdf5_file)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(
            f"Missing training HDF5 for dataset={args.dataset}: {hdf5_path}"
        )

    dataset, loader = load_dataset(
        "training",
        data_path,
        hdf5_file,
        args,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    batch = next(iter(loader))

    check_batch_shapes(dataset, batch)
    assert_finite_batch(batch)
    check_control_alignment(batch)
    assert_quaternion_norm(batch)

    print("Dataset interface check passed.")
    print("  dataset:", args.dataset)
    print("  schema_mode:", getattr(dataset, "schema_mode", "unknown"))
    print("  x_hist:", tuple(batch["x_hist"].shape))
    print("  u_hist:", tuple(batch["u_hist"].shape))
    print("  u_roll:", tuple(batch["u_roll"].shape))
    print("  y_future:", tuple(batch["y_future"].shape))
    print("  context_hist:", tuple(batch["context_hist"].shape))
    print("  context_dim:", dataset.context_dim)


if __name__ == "__main__":
    main()
