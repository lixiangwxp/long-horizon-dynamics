import json
import os

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


STATE_GROUPS = ("p_W", "v_W", "q", "omega_B")
CONTEXT_GROUPS = ("v_B", "a", "alpha", "dmot", "vbat")


class DynamicsDataset(Dataset):
    def __init__(self, mode, data_path, hdf5_file, args):
        self.mode = mode
        self.history_length = args.history_length
        self.unroll_length = args.unroll_length
        self.batch_size = args.batch_size
        self.model_type = args.model_type
        self.schema_mode = "full_state_trajectory"

        self.hdf5_path = os.path.join(data_path, hdf5_file)
        self._load_trajectory_hdf5()
        self._build_window_index()

        self.data_len = len(self.windows)
        self.num_steps = int(np.ceil(self.data_len / self.batch_size))
        self.X_shape = (
            self.data_len,
            self.history_length,
            self.state_dim + self.control_dim,
        )
        self.Y_shape = (self.data_len, self.unroll_length, self.state_dim)
        self.state_len = self.state_dim

        print("Dataset initialized:")
        print("  mode:", self.mode)
        print("  dataset name:", self.dataset_name)
        print("  number of windows:", self.data_len)
        print("  state_dim:", self.state_dim)
        print("  control_dim:", self.control_dim)
        print("  context_dim:", self.context_dim)
        print("  history_length:", self.history_length)
        print("  unroll_length:", self.unroll_length)

    def _load_trajectory_hdf5(self):
        with h5py.File(self.hdf5_path, "r") as hf:
            required_keys = {"data", "trajectory_starts", "trajectory_lengths"}
            missing_keys = required_keys.difference(hf.keys())
            if missing_keys:
                raise ValueError(
                    f"{self.hdf5_path} is not a trajectory-level full-state HDF5. "
                    f"Missing keys: {sorted(missing_keys)}"
                )

            self.dataset_name = hf.attrs.get("dataset_name", "unknown")
            self.feature_slices = json.loads(hf.attrs["feature_slices"])
            data = hf["data"][:].astype(np.float32, copy=False)
            trajectory_starts = hf["trajectory_starts"][:].astype(np.int64)
            trajectory_lengths = hf["trajectory_lengths"][:].astype(np.int64)

        state_parts = []
        for group_name in STATE_GROUPS:
            if group_name not in self.feature_slices:
                raise ValueError(
                    f"{self.hdf5_path} feature_slices is missing {group_name}."
                )
            start, end = self.feature_slices[group_name]
            state_parts.append(data[:, start:end])

        if "u" not in self.feature_slices:
            raise ValueError(f"{self.hdf5_path} feature_slices is missing u.")

        u_start, u_end = self.feature_slices["u"]
        self.x = np.concatenate(state_parts, axis=1).astype(np.float32, copy=False)
        self.u = data[:, u_start:u_end].astype(np.float32, copy=False)

        if "context" in self.feature_slices:
            context_start, context_end = self.feature_slices["context"]
            self.context = data[:, context_start:context_end].astype(
                np.float32, copy=False
            )
        else:
            context_parts = []
            for group_name in CONTEXT_GROUPS:
                if group_name not in self.feature_slices:
                    continue
                start, end = self.feature_slices[group_name]
                context_parts.append(data[:, start:end])

            if context_parts:
                self.context = np.concatenate(context_parts, axis=1).astype(
                    np.float32, copy=False
                )
            else:
                self.context = np.empty((data.shape[0], 0), dtype=np.float32)

        self.trajectory_starts = trajectory_starts
        self.trajectory_lengths = trajectory_lengths
        self.state_dim = self.x.shape[1]
        self.control_dim = self.u.shape[1]
        self.context_dim = self.context.shape[1]

    def _build_window_index(self):
        self.windows = []
        min_length = self.history_length + self.unroll_length
        for trajectory_start, trajectory_length in zip(
            self.trajectory_starts, self.trajectory_lengths
        ):
            if trajectory_length < min_length:
                continue
            num_windows = trajectory_length - min_length + 1
            for local_start in range(num_windows):
                self.windows.append((int(trajectory_start), int(local_start)))

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        trajectory_start, local_start = self.windows[idx]
        start = trajectory_start + local_start
        history_end = start + self.history_length
        rollout_control_start = history_end - 1
        rollout_control_end = rollout_control_start + self.unroll_length
        future_end = history_end + self.unroll_length

        x_hist = self.x[start:history_end]
        u_hist = self.u[start:history_end]
        u_roll = self.u[rollout_control_start:rollout_control_end]
        y_future = self.x[history_end:future_end]
        context_hist = self.context[start:history_end]

        return {
            "x_hist": x_hist,
            "u_hist": u_hist,
            "u_roll": u_roll,
            "y_future": y_future,
            "context_hist": context_hist,
        }


def load_dataset(mode, data_path, hdf5_file, args, num_workers, pin_memory):
    print("Generating", mode, "data ...")
    dataset = DynamicsDataset(mode, data_path, hdf5_file, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print("... Loaded", dataset.data_len, "windows")
    print("|State|   =", dataset.state_dim)
    print("|Control| =", dataset.control_dim)
    print("|Context| =", dataset.context_dim)
    print("|History| =", dataset.history_length)
    print("|Unroll|  =", dataset.unroll_length)
    return dataset, loader
