# retarget-toolbox
Human-robot motion retargeting method based on gradient descent with a user-friendly interface

## Get Started

**Data Preprocessing**
```bash
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