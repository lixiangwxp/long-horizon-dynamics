#!/usr/bin/env python
# coding: utf-8

# # neurobemfullstate 数据处理 walkthrough
# 
# 这个 notebook 只看数据处理，不碰训练代码。
# 
# 目标是把原始 NeuroBEM CSV 转成 `neurobemfullstate` canonical trajectory dataset。每个时间步保存 28 个特征，不保存 `t` 到 feature 向量里：
# 
# ```text
# [p_W, v_W, q, omega_B, a, alpha, u, dmot, vbat]
# ```
# 
# 这样后面训练不同模型时，可以从同一个 HDF5 里按需切列，不用再回头处理 CSV。

# ## 0. 导入依赖并定位仓库

# In[16]:


from pathlib import Path
import json
import shutil
import sys
import tempfile

import h5py
from IPython.display import display
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 配置 Matplotlib 中文字体，避免中文标题显示成小方格。
candidate_fonts = ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS", "SimHei"]
available_fonts = {font.name for font in fm.fontManager.ttflist}
chinese_font = next((font for font in candidate_fonts if font in available_fonts), None)
if chinese_font:
    plt.rcParams["font.sans-serif"] = [chinese_font, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "scripts" / "hdf5.py").exists() and (candidate / "resources").exists():
            return candidate


REPO_ROOT = find_repo_root(Path.cwd())
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from hdf5 import (
    CANONICAL_DATASET,
    CANONICAL_FEATURE_NAMES,
    CANONICAL_FEATURE_SLICES,
    CANONICAL_DT_SECONDS,
    csv_to_canonical_hdf5,
    extract_neurobem_full_state,
    neurobem_csv_to_canonical_trajectory,
    normalize_and_resample_time,
    write_canonical_split_hdf5,
)

print("项目根目录:", REPO_ROOT)
print("canonical 数据集名:", CANONICAL_DATASET)


# ## 1. 原始数据和输出目录
# 
# `neurobemfullstate` 的输入还是原始 NeuroBEM CSV，输出是新的 canonical HDF5。

# In[17]:


SOURCE_DATASET = "neurobem"
SPLIT = "train"

SOURCE_DATA_DIR = REPO_ROOT / "resources" / "data" / SOURCE_DATASET
CANONICAL_DATA_DIR = REPO_ROOT / "resources" / "data" / CANONICAL_DATASET
SPLIT_DIR = SOURCE_DATA_DIR / SPLIT

print("原始 NeuroBEM 目录:", SOURCE_DATA_DIR)
print("canonical 输出目录:", CANONICAL_DATA_DIR)
print("当前查看 split:", SPLIT_DIR)


# ## 2. 读取一个原始 CSV

# In[18]:


csv_files = sorted(SPLIT_DIR.glob("*.csv"))
print("CSV 文件数量:", len(csv_files))
for path in csv_files[:10]:
    print(path.relative_to(REPO_ROOT))

csv_path = csv_files[0]
raw = pd.read_csv(csv_path)

print("当前使用文件:", csv_path.relative_to(REPO_ROOT))
print("原始 shape:", raw.shape)
display(raw.head())
raw.columns


# ## 3. 原始 29 列到 canonical 28 列
# 
# 原始 CSV 有 29 列，其中 `t` 只用于重采样和排序，不进入 feature 向量。其余 28 列全部保存。
# 
# | canonical 组 | 原始列 | 含义 |
# |---|---|---|
# | `p_W` | `pos x/y/z` | 世界系位置 |
# | `v_W` | `vel x/y/z` | 世界系速度 |
# | `q` | `quat w/x/y/z` | 姿态四元数，保存顺序统一成 w、x、y、z |
# | `omega_B` | `ang vel x/y/z` | 机体系角速度 |
# | `a` | `acc x/y/z` | 线加速度 |
# | `alpha` | `ang acc x/y/z` | 角加速度 |
# | `u` | `mot 1/2/3/4` | 电机输入，代码里乘 `0.001` |
# | `dmot` | `dmot 1/2/3/4` | 电机输入变化率，代码里乘 `0.001` |
# | `vbat` | `vbat` | 电池电压相关读数 |

# In[19]:


column_summary = pd.DataFrame({
    "列名": raw.columns,
    "dtype": raw.dtypes.astype(str).values,
    "缺失值数量": raw.isna().sum().values,
})
display(column_summary)


# ## 4. 时间归零并重采样
# 
# `scripts/hdf5.py` 里会先把 `t` 变成相对时间，再按 `0.01s` 重采样，也就是 100 Hz。

# In[20]:


resampled = normalize_and_resample_time(raw)

print("原始 shape:", raw.shape)
print("重采样后 shape:", resampled.shape)
print("重采样间隔:", CANONICAL_DT_SECONDS, "秒")
display(resampled.head())


# In[21]:


plot_column = "vel x"

raw_time_seconds = raw["t"] - raw["t"].iloc[0]
resampled_time_seconds = (resampled["t"] - resampled["t"].iloc[0]).dt.total_seconds()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(raw_time_seconds, raw[plot_column], ".", alpha=0.25, label="原始数据")
ax.plot(resampled_time_seconds, resampled[plot_column], "-", linewidth=1.2, label="重采样后 100 Hz")
ax.set_title(f"重采样前后对比: {plot_column}")
ax.set_xlabel("时间 [秒]")
ax.set_ylabel(plot_column)
ax.legend()
ax.grid(True, alpha=0.3)


# ## 5. 转成 28D canonical array

# In[22]:


canonical_data = extract_neurobem_full_state(resampled)

print("canonical_data shape:", canonical_data.shape)
print("feature 数量:", len(CANONICAL_FEATURE_NAMES))
display(pd.DataFrame(canonical_data[:5], columns=CANONICAL_FEATURE_NAMES))


# In[23]:


feature_schema = pd.DataFrame({
    "index": np.arange(len(CANONICAL_FEATURE_NAMES)),
    "feature": CANONICAL_FEATURE_NAMES,
})
display(feature_schema)

slice_schema = pd.DataFrame([
    {"group": name, "start": bounds[0], "end": bounds[1], "features": CANONICAL_FEATURE_NAMES[bounds[0]:bounds[1]]}
    for name, bounds in CANONICAL_FEATURE_SLICES.items()
])
display(slice_schema)


# ## 6. 写成 canonical HDF5 的结构
# 
# 正式转换时，每个 split 会写一个 HDF5：
# 
# ```text
# resources/data/neurobemfullstate/train/train.h5
# resources/data/neurobemfullstate/valid/valid.h5
# resources/data/neurobemfullstate/test/test.h5
# ```
# 
# HDF5 里保留两种读取方式：
# 
# - `data`: 把当前 split 的所有 trajectory 拼起来的二维数组。
# - `trajectory_starts` / `trajectory_lengths`: 每条 trajectory 在 `data` 里的边界。
# - `trajectories/<trajectory_name>/data`: 每条 trajectory 单独保存。

# In[24]:


tmp_root = Path(tempfile.mkdtemp(prefix="neurobemfullstate-demo-"))
tmp_source_split = tmp_root / "source" / SPLIT
tmp_output_split = tmp_root / "out" / SPLIT
tmp_source_split.mkdir(parents=True)
tmp_output_split.mkdir(parents=True)
shutil.copy(csv_path, tmp_source_split / csv_path.name)

h5_path = write_canonical_split_hdf5(str(tmp_source_split), str(tmp_output_split), f"{SPLIT}.h5")
print("临时 HDF5:", h5_path)

with h5py.File(h5_path, "r") as hf:
    print("keys:", list(hf.keys()))
    print("dataset_name:", hf.attrs["dataset_name"])
    print("data shape:", hf["data"].shape)
    print("trajectory_starts:", hf["trajectory_starts"][:])
    print("trajectory_lengths:", hf["trajectory_lengths"][:])
    print("feature_names:", json.loads(hf.attrs["feature_names"])[:8], "...")
    trajectory_name = json.loads(hf.attrs["trajectory_names"])[0]
    print("第一条 trajectory:", trajectory_name)
    print("trajectory data shape:", hf["trajectories"][trajectory_name]["data"].shape)


# ## 7. split 和动态 window 演示
# 
# 这里补上两个容易混在一起的“切分”：
# 
# - **数据集 split**：原始 CSV 已经放在 `train` / `valid` / `test` 三个目录里，脚本会分别生成三个 HDF5。
# - **训练 window**：HDF5 不保存固定的 `history_length` / `unroll_length`，后面的 PyTorch Dataset 训练时再从完整 trajectory 里动态切。
# 

# In[ ]:


split_rows = []
for split_name in ["train", "valid", "test"]:
    source_split = SOURCE_DATA_DIR / split_name
    output_h5 = CANONICAL_DATA_DIR / split_name / f"{split_name}.h5"
    split_rows.append({
        "split": split_name,
        "CSV 数量": len(list(source_split.glob("*.csv"))),
        "输入目录": str(source_split.relative_to(REPO_ROOT)),
        "输出 HDF5": str(output_h5.relative_to(REPO_ROOT)),
    })

display(pd.DataFrame(split_rows))


# ### 动态切 history / future window
# 
# 截图里担心的是：如果 HDF5 直接存 `inputs: [num_samples, history_length, features]` 和 `outputs: [num_samples, unroll_length, features]`，那 `history_length` 或 `unroll_length` 一改，就得重新从 CSV 处理。
# 
# `neurobemfullstate` 这条新路径没有这个问题。它在 HDF5 里存的是完整 trajectory 的 28 列，后面训练时再按需要切：
# 
# - 13D state：`p_W + v_W + q + omega_B`
# - 10D state：`v_W + q + omega_B`
# - control：`u`
# - context：`dmot + vbat`
# 

# In[ ]:


with h5py.File(h5_path, "r") as hf:
    trajectory_name = json.loads(hf.attrs["trajectory_names"])[0]
    trajectory_data = hf["trajectories"][trajectory_name]["data"][:]
    feature_slices = json.loads(hf.attrs["feature_slices"])

print("使用 trajectory:", trajectory_name)
print("完整 trajectory shape:", trajectory_data.shape)

p_W_slice = slice(*feature_slices["p_W"])
v_W_slice = slice(*feature_slices["v_W"])
q_slice = slice(*feature_slices["q"])
omega_B_slice = slice(*feature_slices["omega_B"])
u_slice = slice(*feature_slices["u"])
dmot_slice = slice(*feature_slices["dmot"])
vbat_slice = slice(*feature_slices["vbat"])

state_13 = np.hstack((
    trajectory_data[:, p_W_slice],
    trajectory_data[:, v_W_slice],
    trajectory_data[:, q_slice],
    trajectory_data[:, omega_B_slice],
))
state_10 = np.hstack((
    trajectory_data[:, v_W_slice],
    trajectory_data[:, q_slice],
    trajectory_data[:, omega_B_slice],
))
u = trajectory_data[:, u_slice]
context = np.hstack((trajectory_data[:, dmot_slice], trajectory_data[:, vbat_slice]))

start = 0
H = 20
F = 10

x_hist_13 = state_13[start:start + H]
y_future_13 = state_13[start + H:start + H + F]
y_future_10 = state_10[start + H:start + H + F]
u_hist = u[start:start + H]
u_roll = u[start + H:start + H + F]
c_hist = context[start:start + H]

shape_summary = pd.DataFrame({
    "张量": ["x_hist_13", "y_future_13", "y_future_10", "u_hist", "u_roll", "c_hist"],
    "含义": [
        "过去 H 步 13D 状态",
        "未来 F 步 13D 状态",
        "未来 F 步 10D 状态",
        "过去 H 步电机输入",
        "未来 F 步电机输入",
        "过去 H 步 context：dmot + vbat",
    ],
    "shape": [
        x_hist_13.shape,
        y_future_13.shape,
        y_future_10.shape,
        u_hist.shape,
        u_roll.shape,
        c_hist.shape,
    ],
})
display(shape_summary)


# ### 换 H / F 不需要重写 HDF5
# 
# 下面还是用同一个 `trajectory_data`，只改 `H` 和 `F`。如果 shape 变了，就说明窗口是在训练前临时切的，不是写死在 HDF5 里的。
# 

# In[ ]:


H2 = 50
F2 = 25

x_hist_13_v2 = state_13[start:start + H2]
y_future_13_v2 = state_13[start + H2:start + H2 + F2]
u_roll_v2 = u[start + H2:start + H2 + F2]
c_hist_v2 = context[start:start + H2]

pd.DataFrame({
    "张量": ["x_hist_13_v2", "y_future_13_v2", "u_roll_v2", "c_hist_v2"],
    "shape": [
        x_hist_13_v2.shape,
        y_future_13_v2.shape,
        u_roll_v2.shape,
        c_hist_v2.shape,
    ],
})


# ## 8. 正式生成 neurobemfullstate
# 
# 在终端里运行下面这两行：
# 
# ```bash
# cd /Users/lixiang/Developer/long-horizon-dynamics/scripts
# /opt/anaconda3/envs/long/bin/python hdf5.py --dataset neurobemfullstate
# ```
# 
# 第一行是进入脚本目录。第二行是用 `long` 环境运行数据处理脚本，从 `resources/data/neurobem` 读取 CSV，生成 `resources/data/neurobemfullstate`。

# ## 代码阅读路线
# 
# - `scripts/hdf5.py`: 生成 `neurobemfullstate` canonical HDF5。
# - `CANONICAL_FEATURE_NAMES`: 28D feature 顺序。
# - `CANONICAL_FEATURE_SLICES`: 后续模型按组切列的位置。
