# retarget-toolbox
Human-robot motion retargeting method based on gradient descent with a user-friendly interface

## 🌟 Key Features

- **Batch Processing**: Efficiently process multiple human mocap files in one operation.
- **Motion Visualization**: Visualize motion trajectories and keyframes.

## 📦 Installation

```bash
conda create -n ret python=3.9
conda activate ret
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install urdfpy matplotlib h5py pandas scikit-learn
conda install -c conda-forge networkx=2.5
pip install "numpy<2"
# Install visualization module
cd viz
pip install -e .
```

## 🚀 Get Started

**Data Preprocessing**
```bash
# Requires BlenderPy: https://github.com/TylerGubala/blenderpy
# Otherwise, use the example motion files in the `data` folder
cd preprocess
python mirror_bvh.py
python transform_bvh.py
```

**Whole-body Motion Retargeting**
```bash
cd retarget
python optimize_hi.py
```

**Visualization**
```bash
cd viz
# Human control with sliders
python example/human_control.py
# Play H5 File
python example/h5_control.py
```

## 📄 Citation

If you use this project in your research, please cite it as follows:

```
@misc{retarget-toolbox,
author = {Haodong Zhang},
title = {Human-Robot Motion Retargeting Toolbox},
year = {2025},
publisher = {GitHub},
howpublished = {\url{https://github.com/0aqz0/retarget-toolbox}},
}
```