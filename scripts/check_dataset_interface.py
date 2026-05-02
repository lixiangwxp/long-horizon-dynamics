import os
import sys

import torch

from config import parse_args
from dynamics_learning.data import load_dataset


def main():
    args = parse_args()

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    data_path = resources_path + "data/" + args.dataset + "/train/"

    dataset, loader = load_dataset(
        "interface_check",
        data_path,
        "train.h5",
        args,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print("dataset.mode:", dataset.mode)
    print("dataset.data_len:", dataset.data_len)
    print("dataset.state_dim:", dataset.state_dim)
    print("dataset.control_dim:", dataset.control_dim)
    print("dataset.context_dim:", dataset.context_dim)
    print("dataset.X_shape:", dataset.X_shape)
    print("dataset.Y_shape:", dataset.Y_shape)

    batch = next(iter(loader))
    print('batch["x_hist"].shape:', batch["x_hist"].shape)
    print('batch["u_hist"].shape:', batch["u_hist"].shape)
    print('batch["u_roll"].shape:', batch["u_roll"].shape)
    print('batch["y_future"].shape:', batch["y_future"].shape)
    print('batch["context_hist"].shape:', batch["context_hist"].shape)

    assert batch["x_hist"].shape[-1] == 13
    assert batch["u_hist"].shape[-1] == 4
    assert batch["u_roll"].shape[-1] == 4
    assert batch["y_future"].shape[-1] == 13
    assert dataset.state_dim == 13
    assert dataset.control_dim == 4

    for key, value in batch.items():
        assert torch.isfinite(value).all(), f"{key} contains non-finite values"

    assert torch.allclose(batch["u_hist"][:, -1, :], batch["u_roll"][:, 0, :]), (
        "Control alignment failed: u_hist[:, -1, :] must equal u_roll[:, 0, :]"
    )

    q = batch["x_hist"][..., 6:10]
    q_norm = torch.linalg.norm(q, dim=-1)
    assert torch.allclose(q_norm, torch.ones_like(q_norm), atol=1e-3), "Quaternion norm is not close to 1"

    print("Dataset interface check passed.")


if __name__ == "__main__":
    main()
