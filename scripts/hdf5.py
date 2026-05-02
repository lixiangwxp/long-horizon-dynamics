import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import parse_args


DATA_DTYPE = np.float32
DT_SECONDS = 0.01
FULL_STATE_SCHEMA_VERSION = "full_state_v1"

FULL_STATE_FEATURE_SLICES = {
    "p_W": [0, 3],
    "v_W": [3, 6],
    "q": [6, 10],
    "omega_B": [10, 13],
    "u": [13, 17],
}

FULL_STATE_FEATURE_NAMES = [
    "p_W_x",
    "p_W_y",
    "p_W_z",
    "v_W_x",
    "v_W_y",
    "v_W_z",
    "q_WB_w",
    "q_WB_x",
    "q_WB_y",
    "q_WB_z",
    "omega_B_x",
    "omega_B_y",
    "omega_B_z",
    "u_1",
    "u_2",
    "u_3",
    "u_4",
]

PITCN_POSITION_ERROR = (
    "PI-TCN full-state requires position columns p_x,p_y,p_z. "
    "The current CSV seems to be the old derivative-learning export. "
    "Please regenerate PI-TCN CSV from raw bags before appendHistory, "
    "keeping p,q,v,w,u per trajectory."
)

NEUROBEM_FEATURE_SLICES = {
    **FULL_STATE_FEATURE_SLICES,
    "v_B": [17, 20],
    "a": [20, 23],
    "alpha": [23, 26],
    "dmot": [26, 30],
    "vbat": [30, 31],
}

NEUROBEM_FEATURE_NAMES = [
    *FULL_STATE_FEATURE_NAMES,
    "v_B_x",
    "v_B_y",
    "v_B_z",
    "a_x",
    "a_y",
    "a_z",
    "alpha_x",
    "alpha_y",
    "alpha_z",
    "dmot_1",
    "dmot_2",
    "dmot_3",
    "dmot_4",
    "vbat",
]

DATASET_SPECS = {
    "neurobemfullstate": {
        "source_dataset": "neurobem",
        "source_dir": "neurobem",
        "feature_names": NEUROBEM_FEATURE_NAMES,
        "feature_slices": NEUROBEM_FEATURE_SLICES,
        "raw_quaternion_order": "wxyz",
        "raw_control_unit": "rad/s",
        "control_kind": "motor_speed_rad_per_sec_scaled_0.001",
        "control_scale": 0.001,
        "canonical_velocity_frame": "world",
        "angular_velocity_frame": "body",
    },
    "pitcnfullstate": {
        "source_dataset": "pi_tcn",
        "source_dir": "pi_tcn",
        "feature_names": FULL_STATE_FEATURE_NAMES,
        "feature_slices": FULL_STATE_FEATURE_SLICES,
        "raw_quaternion_order": "wxyz",
        "raw_control_unit": "rad/s",
        "control_kind": "motor_speed_rad_per_sec_scaled_0.001",
        "control_scale": 0.001,
        "canonical_velocity_frame": "world",
        "angular_velocity_frame": "body",
    },
    "nanodronefullstate": {
        "source_dataset": "nanodrone",
        "feature_names": FULL_STATE_FEATURE_NAMES,
        "feature_slices": FULL_STATE_FEATURE_SLICES,
        "raw_quaternion_order": "xyzw",
        "raw_control_unit": "rad/s",
        "control_kind": "motor_speed_rad_per_sec_scaled_0.001",
        "control_scale": 0.001,
        "canonical_velocity_frame": "world",
        "angular_velocity_frame": "body",
    },
}


def normalize_quaternion_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("Cannot normalize quaternion with zero norm.")
    return q / norm


def enforce_quaternion_sign_continuity_wxyz(q):
    q = np.asarray(q, dtype=np.float64).copy()
    for idx in range(1, q.shape[0]):
        if np.dot(q[idx], q[idx - 1]) < 0.0:
            q[idx] = -q[idx]
    return q


def prepare_quaternion_wxyz(q):
    q = normalize_quaternion_wxyz(q)
    return enforce_quaternion_sign_continuity_wxyz(q)


def quaternion_wxyz_to_rotation_matrix(q):
    q = normalize_quaternion_wxyz(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    rotation = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def normalize_and_resample_time(data):
    data = data.copy()
    data["t"] = data["t"] - data["t"].values[0]
    data["t"] = pd.to_timedelta(data["t"], unit="s")
    data = data.set_index("t").resample(f"{DT_SECONDS}s").mean()
    data = data.interpolate(method="linear").ffill().bfill()
    return data.reset_index(drop=True)


def require_columns(data, required_columns, file_path):
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"{file_path} is missing required columns: {missing_columns}")


def require_pitcn_columns(data, required_columns, group_name, file_path):
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        if group_name == "position":
            raise ValueError(
                f"{PITCN_POSITION_ERROR} Source file: {file_path}. "
                f"Missing columns: {missing_columns}"
            )
        raise ValueError(
            "PI-TCN full-state requires "
            f"{group_name} columns {','.join(required_columns)}. "
            f"Source file: {file_path}. Missing columns: {missing_columns}"
        )


def check_finite(data_np, dataset_name, source_file, feature_slices, feature_names):
    finite_mask = np.isfinite(data_np)
    if finite_mask.all():
        return

    bad_mask = ~finite_mask
    bad_columns = np.where(bad_mask.any(axis=0))[0].tolist()
    bad_feature_names = [
        feature_names[idx] if idx < len(feature_names) else f"column_{idx}"
        for idx in bad_columns
    ]
    bad_groups = []
    for group_name, (start, end) in feature_slices.items():
        if bad_mask[:, start:end].any():
            bad_groups.append(group_name)

    raise ValueError(
        "Non-finite values found before HDF5 write: "
        f"dataset_name={dataset_name}, source_file={source_file}, "
        f"bad_entries={int(bad_mask.sum())}, "
        f"bad_columns={bad_columns}, bad_feature_names={bad_feature_names}, "
        f"bad_feature_groups={bad_groups}"
    )


def extract_neurobem_full_state(data):
    required_columns = [
        "pos x",
        "pos y",
        "pos z",
        "vel x",
        "vel y",
        "vel z",
        "quat w",
        "quat x",
        "quat y",
        "quat z",
        "ang vel x",
        "ang vel y",
        "ang vel z",
        "mot 1",
        "mot 2",
        "mot 3",
        "mot 4",
        "acc x",
        "acc y",
        "acc z",
        "ang acc x",
        "ang acc y",
        "ang acc z",
        "dmot 1",
        "dmot 2",
        "dmot 3",
        "dmot 4",
        "vbat",
    ]
    require_columns(data, required_columns, "neurobem csv")

    p_W = data[["pos x", "pos y", "pos z"]].values
    q = prepare_quaternion_wxyz(data[["quat w", "quat x", "quat y", "quat z"]].values)
    v_B = data[["vel x", "vel y", "vel z"]].values
    rotation_WB = quaternion_wxyz_to_rotation_matrix(q)
    v_W = np.einsum("nij,nj->ni", rotation_WB, v_B)
    omega_B = data[["ang vel x", "ang vel y", "ang vel z"]].values
    u = data[["mot 1", "mot 2", "mot 3", "mot 4"]].values * 0.001
    a = data[["acc x", "acc y", "acc z"]].values
    alpha = data[["ang acc x", "ang acc y", "ang acc z"]].values
    dmot = data[["dmot 1", "dmot 2", "dmot 3", "dmot 4"]].values * 0.001
    vbat = data[["vbat"]].values

    data_np = np.hstack((p_W, v_W, q, omega_B, u, v_B, a, alpha, dmot, vbat))
    return data_np.astype(DATA_DTYPE, copy=False)


def extract_pitcn_full_state(data, csv_file_path):
    position_columns = ["p_x", "p_y", "p_z"]
    velocity_columns = ["v_x", "v_y", "v_z"]
    quaternion_columns = ["q_w", "q_x", "q_y", "q_z"]
    angular_velocity_columns = ["w_x", "w_y", "w_z"]
    motor_speed_columns = ["u_0", "u_1", "u_2", "u_3"]
    motor_thrust_columns = ["f_0", "f_1", "f_2", "f_3"]

    require_pitcn_columns(data, position_columns, "position", csv_file_path)
    require_pitcn_columns(data, velocity_columns, "velocity", csv_file_path)
    require_pitcn_columns(data, quaternion_columns, "quaternion", csv_file_path)
    require_pitcn_columns(
        data, angular_velocity_columns, "angular velocity", csv_file_path
    )

    p_W = data[position_columns].values
    v_W = data[velocity_columns].values
    q = prepare_quaternion_wxyz(data[quaternion_columns].values)
    omega_B = data[angular_velocity_columns].values

    if all(column in data.columns for column in motor_speed_columns):
        u = data[motor_speed_columns].values * 0.001
        control_metadata = {
            "control_kind": "motor_speed",
            "raw_control_unit": "rad/s",
            "control_scale": 0.001,
        }
    elif all(column in data.columns for column in motor_thrust_columns):
        u = data[motor_thrust_columns].values
        control_metadata = {
            "control_kind": "motor_thrust",
            "raw_control_unit": "N",
            "control_scale": 1.0,
        }
    else:
        missing_speed = [
            column for column in motor_speed_columns if column not in data.columns
        ]
        missing_thrust = [
            column for column in motor_thrust_columns if column not in data.columns
        ]
        raise ValueError(
            "PI-TCN full-state requires u_0..u_3 or f_0..f_3. "
            f"Source file: {csv_file_path}. "
            f"Missing motor_speed columns: {missing_speed}; "
            f"missing motor_thrust columns: {missing_thrust}"
        )

    data_np = np.hstack((p_W, v_W, q, omega_B, u))
    return data_np.astype(DATA_DTYPE, copy=False), control_metadata


def extract_nanodrone_full_state(data):
    required_columns = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "qx",
        "qy",
        "qz",
        "qw",
        "wx",
        "wy",
        "wz",
        "m1_rads",
        "m2_rads",
        "m3_rads",
        "m4_rads",
    ]
    require_columns(data, required_columns, "nanodrone csv")

    p_W = data[["x", "y", "z"]].values
    v_W = data[["vx", "vy", "vz"]].values
    q_xyzw = data[["qx", "qy", "qz", "qw"]].values
    q = prepare_quaternion_wxyz(q_xyzw[:, [3, 0, 1, 2]])
    omega_B = data[["wx", "wy", "wz"]].values
    u = data[["m1_rads", "m2_rads", "m3_rads", "m4_rads"]].values * 0.001

    data_np = np.hstack((p_W, v_W, q, omega_B, u))
    return data_np.astype(DATA_DTYPE, copy=False)


def csv_to_canonical_trajectory(csv_file_path, dataset_name):
    data = pd.read_csv(csv_file_path)
    if dataset_name in {"neurobemfullstate", "pitcnfullstate"}:
        data = normalize_and_resample_time(data)

    if dataset_name == "neurobemfullstate":
        return extract_neurobem_full_state(data), {}
    if dataset_name == "pitcnfullstate":
        return extract_pitcn_full_state(data, csv_file_path)
    if dataset_name == "nanodronefullstate":
        return extract_nanodrone_full_state(data), {}

    raise ValueError(f"Unsupported full-state dataset: {dataset_name}")


def unique_trajectory_name(csv_file_path, used_names):
    base_name = csv_file_path.stem
    trajectory_name = base_name
    suffix = 2
    while trajectory_name in used_names:
        trajectory_name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(trajectory_name)
    return trajectory_name


def write_canonical_hdf5(
    csv_files, output_split_path, hdf5_file, dataset_name, metadata
):
    os.makedirs(output_split_path, exist_ok=True)

    trajectories = []
    trajectory_names = []
    source_files = []
    trajectory_lengths = []
    used_names = set()
    feature_slices = metadata["feature_slices"]
    feature_names = metadata["feature_names"]
    metadata_overrides = {}

    for csv_file_path in tqdm(csv_files):
        data_np, overrides = csv_to_canonical_trajectory(csv_file_path, dataset_name)
        for key, value in overrides.items():
            if key in metadata_overrides and metadata_overrides[key] != value:
                raise ValueError(
                    f"Inconsistent {key} values for {dataset_name}: "
                    f"{metadata_overrides[key]} vs {value}. "
                    "Do not mix PI-TCN motor speed and thrust control files "
                    "within the same split."
                )
            metadata_overrides[key] = value

        check_finite(
            data_np, dataset_name, str(csv_file_path), feature_slices, feature_names
        )

        trajectory_name = unique_trajectory_name(csv_file_path, used_names)
        trajectories.append(data_np)
        trajectory_names.append(trajectory_name)
        source_files.append(str(csv_file_path))
        trajectory_lengths.append(data_np.shape[0])

    if not trajectories:
        raise FileNotFoundError(f"No CSV trajectories found for {dataset_name}.")

    all_data = np.concatenate(trajectories, axis=0).astype(DATA_DTYPE, copy=False)
    trajectory_starts = np.cumsum([0] + trajectory_lengths[:-1]).astype(np.int64)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)
    split_metadata = dict(metadata)
    split_metadata.update(metadata_overrides)

    hdf5_path = os.path.join(output_split_path, hdf5_file)
    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["dataset_name"] = dataset_name
        hf.attrs["source_dataset"] = split_metadata["source_dataset"]
        hf.attrs["schema_version"] = FULL_STATE_SCHEMA_VERSION
        hf.attrs["dt_seconds"] = DT_SECONDS
        hf.attrs["feature_names"] = json.dumps(feature_names)
        hf.attrs["feature_slices"] = json.dumps(feature_slices)
        hf.attrs["trajectory_names"] = json.dumps(trajectory_names)
        hf.attrs["source_files"] = json.dumps(source_files)
        hf.attrs["control_kind"] = split_metadata["control_kind"]
        hf.attrs["raw_control_unit"] = split_metadata["raw_control_unit"]
        hf.attrs["control_scale"] = split_metadata["control_scale"]
        hf.attrs["quaternion_order"] = "wxyz"
        hf.attrs["raw_quaternion_order"] = split_metadata["raw_quaternion_order"]
        hf.attrs["canonical_velocity_frame"] = split_metadata[
            "canonical_velocity_frame"
        ]
        hf.attrs["angular_velocity_frame"] = split_metadata["angular_velocity_frame"]

        data_dataset = hf.create_dataset("data", data=all_data)
        data_dataset.dims[0].label = "time_steps"
        data_dataset.dims[1].label = "features"

        hf.create_dataset("trajectory_starts", data=trajectory_starts)
        hf.create_dataset("trajectory_lengths", data=trajectory_lengths)

        trajectories_group = hf.create_group("trajectories")
        for trajectory_name, source_file, data_np in zip(
            trajectory_names, source_files, trajectories
        ):
            trajectory_group = trajectories_group.create_group(trajectory_name)
            trajectory_group.attrs["source_file"] = source_file
            trajectory_group.attrs["feature_names"] = json.dumps(feature_names)
            trajectory_data = trajectory_group.create_dataset("data", data=data_np)
            trajectory_data.dims[0].label = "time_steps"
            trajectory_data.dims[1].label = "features"

        hf.flush()

    return hdf5_path


def sorted_csv_files(path):
    return sorted(Path(path).glob("*.csv"))


def convert_standard_full_state_dataset(dataset_name, resources_path):
    spec = DATASET_SPECS[dataset_name]
    source_data_path = Path(resources_path) / "data" / spec["source_dir"]
    output_data_path = Path(resources_path) / "data" / dataset_name
    split_specs = [
        ("train", "train.h5"),
        ("valid", "valid.h5"),
        ("test", "test.h5"),
    ]

    for split_name, hdf5_file in split_specs:
        source_split_path = source_data_path / split_name
        if dataset_name == "pitcnfullstate" and split_name == "test":
            test_trajectories = source_data_path / "test_trajectories"
            if test_trajectories.is_dir():
                source_split_path = test_trajectories

        csv_files = sorted_csv_files(source_split_path)
        output_split_path = output_data_path / split_name
        hdf5_path = write_canonical_hdf5(
            csv_files, output_split_path, hdf5_file, dataset_name, spec
        )
        print(f"Saved {dataset_name} {split_name} split to {hdf5_path}")


def find_nanodrone_data_root(raw_path):
    raw_path = Path(raw_path).expanduser()
    candidates = [
        raw_path / "data",
        raw_path,
        raw_path / "nanodrone-sysid-benchmark" / "data",
    ]
    checked_paths = []

    for candidate in candidates:
        train_dir = candidate / "train"
        test_dir = candidate / "test"
        checked_paths.extend([str(train_dir), str(test_dir)])
        if train_dir.is_dir() and test_dir.is_dir():
            return candidate, checked_paths

    raise FileNotFoundError(
        "Could not find NanoDrone train/test CSV directories. "
        f"Checked paths: {checked_paths}"
    )


def split_nanodrone_csv_files(data_root):
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"

    if valid_dir.is_dir():
        return {
            "train": sorted_csv_files(train_dir),
            "valid": sorted_csv_files(valid_dir),
            "test": sorted_csv_files(test_dir),
        }

    train_files = []
    valid_files = []
    for csv_file in sorted_csv_files(train_dir):
        name = csv_file.name.lower()
        if not any(prefix in name for prefix in ("square", "random", "chirp")):
            continue
        if "run4" in name:
            valid_files.append(csv_file)
        elif any(run in name for run in ("run1", "run2", "run3")):
            train_files.append(csv_file)

    test_files = [
        csv_file
        for csv_file in sorted_csv_files(test_dir)
        if "melon" in csv_file.name.lower()
    ]

    return {
        "train": train_files,
        "valid": valid_files,
        "test": test_files,
    }


def convert_nanodrone_full_state_dataset(args, resources_path):
    dataset_name = "nanodronefullstate"
    spec = DATASET_SPECS[dataset_name]
    data_root, checked_paths = find_nanodrone_data_root(args.nanodrone_raw_path)
    print("NanoDrone raw data root:", data_root)
    print("NanoDrone checked paths:", checked_paths)

    output_data_path = Path(resources_path) / "data" / dataset_name
    split_files = split_nanodrone_csv_files(data_root)
    split_specs = {
        "train": "train.h5",
        "valid": "valid.h5",
        "test": "test.h5",
    }

    for split_name, hdf5_file in split_specs.items():
        csv_files = split_files[split_name]
        if not csv_files:
            raise FileNotFoundError(
                f"No NanoDrone CSV files selected for {split_name} split "
                f"under {data_root}."
            )

        output_split_path = output_data_path / split_name
        hdf5_path = write_canonical_hdf5(
            csv_files, output_split_path, hdf5_file, dataset_name, spec
        )
        print(f"Saved {dataset_name} {split_name} split to {hdf5_path}")


def main():
    args = parse_args()
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"

    if args.dataset == "nanodronefullstate":
        convert_nanodrone_full_state_dataset(args, resources_path)
    elif args.dataset in {"neurobemfullstate", "pitcnfullstate"}:
        convert_standard_full_state_dataset(args.dataset, resources_path)
    else:
        raise ValueError(
            f"Unsupported dataset for full-state HDF5 conversion: {args.dataset}"
        )


if __name__ == "__main__":
    main()
