import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import parse_args


DATA_DTYPE = np.float32
CANONICAL_DT_SECONDS = 0.01

NEUROBEM_FULLSTATE_DATASET = "neurobemfullstate"
PITCN_FULLSTATE_DATASET = "pitcnfullstate"
CANONICAL_DATASET = NEUROBEM_FULLSTATE_DATASET
SOURCE_DATASET_FOR_CANONICAL = "neurobem"
SCHEMA_VERSION = "full_state_v1"

BASE_FEATURE_SLICES = {
    "p_W": [0, 3],
    "v_W": [3, 6],
    "q": [6, 10],
    "omega_B": [10, 13],
    "u": [13, 17],
}

NEUROBEM_FEATURE_NAMES = [
    "p_W_x", "p_W_y", "p_W_z",
    "v_W_x", "v_W_y", "v_W_z",
    "q_WB_w", "q_WB_x", "q_WB_y", "q_WB_z",
    "omega_B_x", "omega_B_y", "omega_B_z",
    "u_1", "u_2", "u_3", "u_4",
    "v_B_x", "v_B_y", "v_B_z",
    "a_x", "a_y", "a_z",
    "alpha_x", "alpha_y", "alpha_z",
    "dmot_1", "dmot_2", "dmot_3", "dmot_4",
    "vbat",
]

NEUROBEM_FEATURE_SLICES = {
    **BASE_FEATURE_SLICES,
    "v_B": [17, 20],
    "a": [20, 23],
    "alpha": [23, 26],
    "dmot": [26, 30],
    "vbat": [30, 31],
}

PITCN_FEATURE_NAMES = [
    "p_W_x", "p_W_y", "p_W_z",
    "v_W_x", "v_W_y", "v_W_z",
    "q_WB_w", "q_WB_x", "q_WB_y", "q_WB_z",
    "omega_B_x", "omega_B_y", "omega_B_z",
    "u_0", "u_1", "u_2", "u_3",
]

PITCN_FEATURE_SLICES = dict(BASE_FEATURE_SLICES)

CANONICAL_FEATURE_NAMES = NEUROBEM_FEATURE_NAMES
CANONICAL_FEATURE_SLICES = NEUROBEM_FEATURE_SLICES

PITCN_MISSING_POSITION_MESSAGE = (
    "PI-TCN full-state requires position columns p_x,p_y,p_z. "
    "The current CSV seems to be the old derivative-learning export. "
    "Please regenerate PI-TCN CSV from raw bags before appendHistory, "
    "keeping p,q,v,w,u per trajectory."
)


def normalize_and_resample_time(data):
    data = data.copy()
    data["t"] = data["t"] - data["t"].values[0]
    data["t"] = pd.to_datetime(data["t"], unit="s")
    data.set_index("t", inplace=True)
    data = data.resample(f"{CANONICAL_DT_SECONDS}s").mean()
    data = data.interpolate(method="linear", limit_direction="both")
    data = data.dropna()
    data.reset_index(inplace=True)
    return data


def normalize_quaternion_wxyz(q):
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.clip(norm, 1e-12, None)


def quaternion_wxyz_to_rotation_matrix(q):
    q = normalize_quaternion_wxyz(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def extract_neurobem_full_state(data):
    p_W = data[["pos x", "pos y", "pos z"]].values
    q = normalize_quaternion_wxyz(data[["quat w", "quat x", "quat y", "quat z"]].values)
    v_B = data[["vel x", "vel y", "vel z"]].values
    R_WB = quaternion_wxyz_to_rotation_matrix(q)
    v_W = np.einsum("nij,nj->ni", R_WB, v_B)
    omega_B = data[["ang vel x", "ang vel y", "ang vel z"]].values
    u = data[["mot 1", "mot 2", "mot 3", "mot 4"]].values * 0.001
    a = data[["acc x", "acc y", "acc z"]].values
    alpha = data[["ang acc x", "ang acc y", "ang acc z"]].values
    dmot = data[["dmot 1", "dmot 2", "dmot 3", "dmot 4"]].values * 0.001
    vbat = data[["vbat"]].values

    data_np = np.hstack((p_W, v_W, q, omega_B, u, v_B, a, alpha, dmot, vbat))
    return data_np.astype(DATA_DTYPE, copy=False)


def extract_pitcn_full_state(data):
    position_columns = ["p_x", "p_y", "p_z"]
    if not all(column in data.columns for column in position_columns):
        raise ValueError(PITCN_MISSING_POSITION_MESSAGE)

    state_columns = [
        "p_x", "p_y", "p_z",
        "v_x", "v_y", "v_z",
        "q_w", "q_x", "q_y", "q_z",
        "w_x", "w_y", "w_z",
    ]
    missing_state_columns = [column for column in state_columns if column not in data.columns]
    if missing_state_columns:
        raise ValueError(f"PI-TCN full-state missing columns: {missing_state_columns}")

    if all(column in data.columns for column in ["u_0", "u_1", "u_2", "u_3"]):
        control = data[["u_0", "u_1", "u_2", "u_3"]].values
        control_kind = "motor_speed"
    elif all(column in data.columns for column in ["f_0", "f_1", "f_2", "f_3"]):
        control = data[["f_0", "f_1", "f_2", "f_3"]].values
        control_kind = "motor_thrust"
    else:
        raise ValueError("PI-TCN full-state requires u_0..u_3 or f_0..f_3")

    p_W = data[["p_x", "p_y", "p_z"]].values
    v_W = data[["v_x", "v_y", "v_z"]].values
    q = normalize_quaternion_wxyz(data[["q_w", "q_x", "q_y", "q_z"]].values)
    omega_B = data[["w_x", "w_y", "w_z"]].values

    data_np = np.hstack((p_W, v_W, q, omega_B, control))
    return data_np.astype(DATA_DTYPE, copy=False), control_kind


def neurobem_csv_to_canonical_trajectory(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = normalize_and_resample_time(data)
    data_np = extract_neurobem_full_state(data)
    return data_np.astype(DATA_DTYPE, copy=False)


def pitcn_csv_to_canonical_trajectory(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = normalize_and_resample_time(data)
    return extract_pitcn_full_state(data)


def dataset_schema(dataset_name):
    if dataset_name == NEUROBEM_FULLSTATE_DATASET:
        return {
            "source_dataset": "neurobem",
            "feature_names": NEUROBEM_FEATURE_NAMES,
            "feature_slices": NEUROBEM_FEATURE_SLICES,
            "control_kind": "motor_speed_scaled",
        }
    if dataset_name == PITCN_FULLSTATE_DATASET:
        return {
            "source_dataset": "pi_tcn",
            "feature_names": PITCN_FEATURE_NAMES,
            "feature_slices": PITCN_FEATURE_SLICES,
            "control_kind": None,
        }
    raise ValueError(f"Unsupported full-state dataset: {dataset_name}")


def csv_to_trajectory(dataset_name, csv_file_path):
    if dataset_name == NEUROBEM_FULLSTATE_DATASET:
        data_np = neurobem_csv_to_canonical_trajectory(csv_file_path)
        return data_np, "motor_speed_scaled"
    if dataset_name == PITCN_FULLSTATE_DATASET:
        return pitcn_csv_to_canonical_trajectory(csv_file_path)
    raise ValueError(f"Unsupported full-state dataset: {dataset_name}")


def write_full_state_split_hdf5(source_split_path, output_split_path, hdf5_file, dataset_name):
    schema = dataset_schema(dataset_name)
    os.makedirs(output_split_path, exist_ok=True)

    trajectories = []
    trajectory_names = []
    source_files = []
    trajectory_lengths = []
    split_control_kind = schema["control_kind"]

    csv_files = [file for file in sorted(os.listdir(source_split_path)) if file.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in {source_split_path}")

    for file in tqdm(csv_files):
        csv_file_path = os.path.join(source_split_path, file)
        data_np, control_kind = csv_to_trajectory(dataset_name, csv_file_path)

        if split_control_kind is None:
            split_control_kind = control_kind
        elif split_control_kind != control_kind:
            raise ValueError(f"Mixed control_kind in {source_split_path}: {split_control_kind} and {control_kind}")

        trajectory_name = os.path.splitext(file)[0]
        trajectories.append(data_np)
        trajectory_names.append(trajectory_name)
        source_files.append(file)
        trajectory_lengths.append(data_np.shape[0])

    all_data = np.concatenate(trajectories, axis=0).astype(DATA_DTYPE, copy=False)
    trajectory_starts = np.cumsum([0] + trajectory_lengths[:-1]).astype(np.int64)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)

    hdf5_path = os.path.join(output_split_path, hdf5_file)
    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["dataset_name"] = dataset_name
        hf.attrs["source_dataset"] = schema["source_dataset"]
        hf.attrs["schema_version"] = SCHEMA_VERSION
        hf.attrs["dt_seconds"] = CANONICAL_DT_SECONDS
        hf.attrs["feature_names"] = json.dumps(schema["feature_names"])
        hf.attrs["feature_slices"] = json.dumps(schema["feature_slices"])
        hf.attrs["trajectory_names"] = json.dumps(trajectory_names)
        hf.attrs["source_files"] = json.dumps(source_files)
        hf.attrs["control_kind"] = split_control_kind

        data_dataset = hf.create_dataset("data", data=all_data)
        data_dataset.dims[0].label = "time_steps"
        data_dataset.dims[1].label = "features"

        hf.create_dataset("trajectory_starts", data=trajectory_starts)
        hf.create_dataset("trajectory_lengths", data=trajectory_lengths)

        trajectories_group = hf.create_group("trajectories")
        for trajectory_name, source_file, data_np in zip(trajectory_names, source_files, trajectories):
            trajectory_group = trajectories_group.create_group(trajectory_name)
            trajectory_group.attrs["source_file"] = source_file
            trajectory_group.attrs["feature_names"] = json.dumps(schema["feature_names"])
            trajectory_group.attrs["feature_slices"] = json.dumps(schema["feature_slices"])
            trajectory_group.attrs["control_kind"] = split_control_kind
            trajectory_data = trajectory_group.create_dataset("data", data=data_np)
            trajectory_data.dims[0].label = "time_steps"
            trajectory_data.dims[1].label = "features"

        hf.flush()

    return hdf5_path


def split_specs(source_data_path, dataset_name):
    specs = [
        ("train/", "train/", "train.h5"),
        ("valid/", "valid/", "valid.h5"),
    ]
    if dataset_name == PITCN_FULLSTATE_DATASET and os.path.isdir(os.path.join(source_data_path, "test_trajectories")):
        specs.append(("test_trajectories/", "test/", "test.h5"))
    else:
        specs.append(("test/", "test/", "test.h5"))
    return specs


def csv_to_full_state_hdf5(source_data_path, output_data_path, dataset_name):
    for source_folder, output_folder, hdf5_file in split_specs(source_data_path, dataset_name):
        source_split_path = os.path.join(source_data_path, source_folder)
        output_split_path = os.path.join(output_data_path, output_folder)
        hdf5_path = write_full_state_split_hdf5(source_split_path, output_split_path, hdf5_file, dataset_name)
        print(f"Saved {dataset_name} split to {hdf5_path}")


def write_canonical_split_hdf5(source_split_path, output_split_path, hdf5_file):
    return write_full_state_split_hdf5(source_split_path, output_split_path, hdf5_file, NEUROBEM_FULLSTATE_DATASET)


def csv_to_canonical_hdf5(source_data_path, output_data_path):
    return csv_to_full_state_hdf5(source_data_path, output_data_path, NEUROBEM_FULLSTATE_DATASET)


if __name__ == "__main__":
    args = parse_args()

    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"

    if args.dataset == NEUROBEM_FULLSTATE_DATASET:
        source_data_path = resources_path + "data/neurobem/"
        output_data_path = resources_path + f"data/{NEUROBEM_FULLSTATE_DATASET}/"
    elif args.dataset == PITCN_FULLSTATE_DATASET:
        source_data_path = resources_path + "data/pi_tcn/"
        output_data_path = resources_path + f"data/{PITCN_FULLSTATE_DATASET}/"
    else:
        raise ValueError("hdf5.py now generates canonical full-state trajectory HDF5 for neurobemfullstate or pitcnfullstate.")

    csv_to_full_state_hdf5(source_data_path, output_data_path, args.dataset)
