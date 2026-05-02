import json
import os

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset


FULL_STATE_HDF5_ERROR = (
    "This code now expects canonical full-state trajectory HDF5. "
    "Please regenerate HDF5 with hdf5.py."
)


class DynamicsDataset(Dataset):
    def __init__(self, data_path, hdf5_file, args):
        self.mode = "full_state_trajectory"
        self.history_length = args.history_length
        self.unroll_length = args.unroll_length
        self.batch_size = args.batch_size
        self.model_type = args.model_type

        self.data, self.feature_slices, self.trajectory_starts, self.trajectory_lengths = self.load_data(data_path, hdf5_file)
        self.dataset_name = self.hdf5_attrs.get("dataset_name", "unknown")

        self.x = self.make_state_array()
        self.u = self.slice_group("u")
        self.context = self.make_context_array()

        self.state_dim = 13
        self.control_dim = 4
        self.context_dim = self.context.shape[1]

        self.window_starts = self.build_window_starts()
        self.data_len = len(self.window_starts)
        self.num_steps = int(np.ceil(self.data_len / self.batch_size))
        self.X_shape = (self.data_len, self.history_length, self.state_dim + self.control_dim)
        self.Y_shape = (self.data_len, self.unroll_length, self.state_dim)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        H = self.history_length
        F = self.unroll_length

        sample = {
            "x_hist": self.x[start:start + H].astype(np.float32, copy=False),
            "u_hist": self.u[start:start + H].astype(np.float32, copy=False),
            "u_roll": self.u[start + H - 1:start + H - 1 + F].astype(np.float32, copy=False),
            "y_future": self.x[start + H:start + H + F].astype(np.float32, copy=False),
            "context_hist": self.context[start:start + H].astype(np.float32, copy=False),
        }
        return sample

    def load_data(self, hdf5_path, hdf5_file):
        hdf5_path = os.path.join(hdf5_path, hdf5_file)
        with h5py.File(hdf5_path, "r") as hf:
            required = ["data", "trajectory_starts", "trajectory_lengths"]
            if any(key not in hf for key in required) or "feature_slices" not in hf.attrs:
                raise ValueError(FULL_STATE_HDF5_ERROR)

            self.hdf5_attrs = {key: hf.attrs[key] for key in hf.attrs.keys()}
            data = hf["data"][:].astype(np.float32, copy=False)
            feature_slices = json.loads(hf.attrs["feature_slices"])
            trajectory_starts = hf["trajectory_starts"][:].astype(np.int64, copy=False)
            trajectory_lengths = hf["trajectory_lengths"][:].astype(np.int64, copy=False)

        return data, feature_slices, trajectory_starts, trajectory_lengths

    def slice_group(self, name):
        start, end = self.feature_slices[name]
        return self.data[:, start:end]

    def make_state_array(self):
        return np.concatenate(
            [
                self.slice_group("p_W"),
                self.slice_group("v_W"),
                self.slice_group("q"),
                self.slice_group("omega_B"),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def make_context_array(self):
        context_groups = []
        for name in ["v_B", "a", "alpha", "dmot", "vbat"]:
            if name in self.feature_slices:
                context_groups.append(self.slice_group(name))

        if context_groups:
            return np.concatenate(context_groups, axis=1).astype(np.float32, copy=False)
        return np.zeros((self.data.shape[0], 0), dtype=np.float32)

    def build_window_starts(self):
        window_starts = []
        H = self.history_length
        F = self.unroll_length

        for trajectory_start, trajectory_length in zip(self.trajectory_starts, self.trajectory_lengths):
            trajectory_end = trajectory_start + trajectory_length
            last_start_exclusive = trajectory_end - H - F + 1
            for start in range(trajectory_start, last_start_exclusive):
                window_starts.append(start)

        if not window_starts:
            raise ValueError(f"No valid full-state windows for H={H}, F={F}.")

        return np.asarray(window_starts, dtype=np.int64)


def load_dataset(mode, data_path, hdf5_file, args, num_workers, pin_memory):
    print("Generating", mode, "data ...")
    dataset = DynamicsDataset(data_path, hdf5_file, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("mode =", dataset.mode)
    print("dataset =", dataset.dataset_name)
    print("Loaded", dataset.data_len, "windows")
    print("|State|   =", dataset.state_dim)
    print("|Control| =", dataset.control_dim)
    print("|Context| =", dataset.context_dim)
    print("|History| =", dataset.history_length)
    print("|Unroll|  =", dataset.unroll_length)

    return dataset, loader
