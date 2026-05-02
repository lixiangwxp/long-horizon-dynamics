import os
import json
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import tqdm
from config import parse_args

DATA_DTYPE = np.float32
CANONICAL_DATASET = "neurobemfullstate"
SOURCE_DATASET_FOR_CANONICAL = "neurobem"
CANONICAL_DT_SECONDS = 0.01

CANONICAL_FEATURE_NAMES = [
    "p_W_x", "p_W_y", "p_W_z",
    "v_W_x", "v_W_y", "v_W_z",
    "q_WB_w", "q_WB_x", "q_WB_y", "q_WB_z",
    "omega_B_x", "omega_B_y", "omega_B_z",
    "a_x", "a_y", "a_z",
    "alpha_x", "alpha_y", "alpha_z",
    "u_1", "u_2", "u_3", "u_4",
    "v_B_x", "v_B_y", "v_B_z",
    "dmot_1", "dmot_2", "dmot_3", "dmot_4",
    "vbat",
]

CANONICAL_FEATURE_SLICES = {
    "p_W": [0, 3],
    "v_W": [3, 6],
    "q": [6, 10],
    "omega_B": [10, 13],
    "a": [13, 16],
    "alpha": [16, 19],
    "u": [19, 23],
    "v_B": [23, 26],
    "dmot": [26, 30],
    "vbat": [30, 31],
}

def extract_data(data, dataset_name):
    if dataset_name == "pi_tcn":
        velocity_data = data[['v_x', 'v_y', 'v_z']].values
        attitude_data = data[['q_w', 'q_x', 'q_y', 'q_z']].values
        angular_velocity_data = data[['w_x', 'w_y', 'w_z']].values
        control_data = data[['u_0', 'u_1', 'u_2', 'u_3']].values * 0.001

    elif dataset_name == "neurobem":
        velocity_data = data[['vel x', 'vel y', 'vel z']].values
        attitude_data = data[['quat w', 'quat x', 'quat y', 'quat z']].values
        angular_velocity_data = data[['ang vel x', 'ang vel y', 'ang vel z']].values
        control_data = data[['mot 1', 'mot 2', 'mot 3', 'mot 4']].values * 0.001

    return velocity_data, attitude_data, angular_velocity_data, control_data


def normalize_and_resample_time(data):
    data = data.copy()
    data['t'] = data['t'] - data['t'].values[0]
    data['t'] = pd.to_datetime(data['t'], unit='s')
    data.set_index('t', inplace=True)
    data = data.resample(f'{CANONICAL_DT_SECONDS}s').mean()
    data.reset_index(inplace=True)
    return data


def quaternion_wxyz_to_rotation_matrix(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
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
    p_W = data[['pos x', 'pos y', 'pos z']].values
    q = data[['quat w', 'quat x', 'quat y', 'quat z']].values
    v_B = data[['vel x', 'vel y', 'vel z']].values
    R_WB = quaternion_wxyz_to_rotation_matrix(q)
    v_W = np.einsum('nij,nj->ni', R_WB, v_B)
    omega_B = data[['ang vel x', 'ang vel y', 'ang vel z']].values
    a = data[['acc x', 'acc y', 'acc z']].values
    alpha = data[['ang acc x', 'ang acc y', 'ang acc z']].values
    u = data[['mot 1', 'mot 2', 'mot 3', 'mot 4']].values * 0.001
    dmot = data[['dmot 1', 'dmot 2', 'dmot 3', 'dmot 4']].values * 0.001
    vbat = data[['vbat']].values

    data_np = np.hstack((p_W, v_W, q, omega_B, a, alpha, u, v_B, dmot, vbat))
    return data_np.astype(DATA_DTYPE, copy=False)


def neurobem_csv_to_canonical_trajectory(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = normalize_and_resample_time(data)
    data_np = extract_neurobem_full_state(data)
    return data_np.astype(DATA_DTYPE, copy=False)


def write_canonical_split_hdf5(source_split_path, output_split_path, hdf5_file):
    os.makedirs(output_split_path, exist_ok=True)

    trajectories = []
    trajectory_names = []
    source_files = []
    trajectory_lengths = []

    for file in tqdm(sorted(os.listdir(source_split_path))):
        if not file.endswith(".csv"):
            continue

        csv_file_path = os.path.join(source_split_path, file)
        data_np = neurobem_csv_to_canonical_trajectory(csv_file_path)

        trajectory_name = os.path.splitext(file)[0]
        trajectories.append(data_np)
        trajectory_names.append(trajectory_name)
        source_files.append(file)
        trajectory_lengths.append(data_np.shape[0])

    all_data = np.concatenate(trajectories, axis=0).astype(DATA_DTYPE, copy=False)
    trajectory_starts = np.cumsum([0] + trajectory_lengths[:-1]).astype(np.int64)
    trajectory_lengths = np.asarray(trajectory_lengths, dtype=np.int64)

    hdf5_path = os.path.join(output_split_path, hdf5_file)
    with h5py.File(hdf5_path, 'w') as hf:
        hf.attrs['dataset_name'] = CANONICAL_DATASET
        hf.attrs['source_dataset'] = SOURCE_DATASET_FOR_CANONICAL
        hf.attrs['schema_version'] = 'v2'
        hf.attrs['dt_seconds'] = CANONICAL_DT_SECONDS
        hf.attrs['feature_names'] = json.dumps(CANONICAL_FEATURE_NAMES)
        hf.attrs['feature_slices'] = json.dumps(CANONICAL_FEATURE_SLICES)
        hf.attrs['trajectory_names'] = json.dumps(trajectory_names)
        hf.attrs['source_files'] = json.dumps(source_files)

        data_dataset = hf.create_dataset('data', data=all_data)
        data_dataset.dims[0].label = 'time_steps'
        data_dataset.dims[1].label = 'features'

        hf.create_dataset('trajectory_starts', data=trajectory_starts)
        hf.create_dataset('trajectory_lengths', data=trajectory_lengths)

        trajectories_group = hf.create_group('trajectories')
        for trajectory_name, source_file, data_np in zip(trajectory_names, source_files, trajectories):
            trajectory_group = trajectories_group.create_group(trajectory_name)
            trajectory_group.attrs['source_file'] = source_file
            trajectory_group.attrs['feature_names'] = json.dumps(CANONICAL_FEATURE_NAMES)
            trajectory_data = trajectory_group.create_dataset('data', data=data_np)
            trajectory_data.dims[0].label = 'time_steps'
            trajectory_data.dims[1].label = 'features'

        hf.flush()

    return hdf5_path


def csv_to_canonical_hdf5(source_data_path, output_data_path):
    split_specs = [
        ('train/', 'train.h5'),
        ('valid/', 'valid.h5'),
        ('test/', 'test.h5'),
    ]

    for folder_name, hdf5_file in split_specs:
        source_split_path = os.path.join(source_data_path, folder_name)
        output_split_path = os.path.join(output_data_path, folder_name)
        hdf5_path = write_canonical_split_hdf5(source_split_path, output_split_path, hdf5_file)
        print(f"Saved {CANONICAL_DATASET} split to {hdf5_path}")


def csv_to_hdf5(args, data_path):

    hdf5(data_path, 'train/', 'train.h5',  args.dataset,  args.history_length, args.unroll_length)
    hdf5(data_path, 'valid/', 'valid.h5',  args.dataset,  args.history_length, args.unroll_length)
    if args.dataset == "pi_tcn" and os.path.isdir(data_path + 'test_trajectories/'):
        hdf5_trajectories(data_path, 'test_trajectories/', args.dataset, args.history_length, 60, output_folder_name='test/')
    else:
        hdf5_trajectories(data_path, 'test/',  args.dataset,  args.history_length, 60)

def hdf5(data_path, folder_name, hdf5_file, dataset, history_length, unroll_length):

    all_X = []
    all_Y = []

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            # Modify time to start from 0
            data['t'] = data['t'] - data['t'].values[0]

            data['t'] = pd.to_datetime(data['t'], unit='s')

            data.set_index('t', inplace=True)
            data = data.resample('0.01s').mean()
            data.reset_index(inplace=True)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)
            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data)).astype(DATA_DTYPE, copy=False)

            num_samples = data_np.shape[0] - history_length - unroll_length

            X = np.zeros((num_samples, history_length, data_np.shape[1]), dtype=DATA_DTYPE)
            Y = np.zeros((num_samples, unroll_length, data_np.shape[1]), dtype=DATA_DTYPE)

            for i in range(num_samples):
                X[i, :, :] =   data_np[i:i+history_length, :]
                Y[i,:,:]   =   data_np[i+history_length:i+history_length+unroll_length,:data_np.shape[1]]

            all_X.append(X)
            all_Y.append(Y)

    X = np.concatenate(all_X, axis=0).astype(DATA_DTYPE, copy=False)
    Y = np.concatenate(all_Y, axis=0).astype(DATA_DTYPE, copy=False)    
        
    # save the data
    # Create the HDF5 file and datasets for inputs and outputs
    with h5py.File(data_path + folder_name + hdf5_file, 'w') as hf:
        inputs_data = hf.create_dataset('inputs', data=X)
        inputs_data.dims[0].label = 'num_samples'
        inputs_data.dims[1].label = 'history_length'
        inputs_data.dims[2].label = 'features'

        outputs_data = hf.create_dataset('outputs', data=Y)
        outputs_data.dims[0].label = 'num_samples'
        outputs_data.dims[1].label = 'unroll_length'
        outputs_data.dims[2].label = 'features'

        # flush and close the file
        hf.flush()
        hf.close()
        
    return X, Y

def hdf5_trajectories(data_path, folder_name, dataset, history_length, unroll_length, output_folder_name=None):
    output_folder_name = output_folder_name or folder_name
    os.makedirs(data_path + output_folder_name, exist_ok=True)

    # load the data
    for file in tqdm(os.listdir(data_path + folder_name)):
        if file.endswith(".csv"):
            csv_file_path = os.path.join(data_path + folder_name, file)
            data = pd.read_csv(csv_file_path)

            # Modify time to start from 0
            data['t'] = data['t'] - data['t'].values[0]

            data['t'] = pd.to_datetime(data['t'], unit='s')

            data.set_index('t', inplace=True)
            data = data.resample('0.01s').mean()
            data.reset_index(inplace=True)

            velocity_data, attitude_data, angular_velocity_data, control_data = extract_data(data, dataset)

            data_np = np.hstack((velocity_data, attitude_data, angular_velocity_data, control_data)).astype(DATA_DTYPE, copy=False)
            num_samples = data_np.shape[0] - history_length - unroll_length

            X = np.zeros((num_samples, history_length, data_np.shape[1]), dtype=DATA_DTYPE)
            Y = np.zeros((num_samples, unroll_length, data_np.shape[1]), dtype=DATA_DTYPE)

            for i in range(num_samples):
                X[i, :, :] =   data_np[i:i+history_length, :]
                Y[i,:,:]   =   data_np[i+history_length:i+history_length+unroll_length,:data_np.shape[1]]

            # Save to hdf5 with the same name as the csv file
            with h5py.File(data_path + output_folder_name + file[:-4] + '.h5', 'w') as hf: 
                inputs_data = hf.create_dataset('inputs', data=X)
                inputs_data.dims[0].label = 'num_samples'
                inputs_data.dims[1].label = 'history_length'
                inputs_data.dims[2].label = 'features'

                outputs_data = hf.create_dataset('outputs', data=Y)
                outputs_data.dims[0].label = 'num_samples'
                outputs_data.dims[1].label = 'unroll_length'
                outputs_data.dims[2].label = 'features'

                # flush and close the file
                hf.flush()
                hf.close()    
                
# load hdf5
def load_hdf5(data_path, hdf5_file):
    with h5py.File(data_path + hdf5_file, 'r') as hf:
        X = hf['inputs'][:]
        Y = hf['outputs'][:]

    return X, Y

if __name__ == "__main__":
    args = parse_args()

    # Set global paths 
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    resources_path = folder_path + "resources/"
    if args.dataset == CANONICAL_DATASET:
        source_data_path = resources_path + "data/" + SOURCE_DATASET_FOR_CANONICAL + "/"
        output_data_path = resources_path + "data/" + CANONICAL_DATASET + "/"
        csv_to_canonical_hdf5(source_data_path, output_data_path)
    else:
        data_path = resources_path + "data/" + args.dataset + "/" 
        csv_to_hdf5(args, data_path)
