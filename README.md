# SFDS-DETR

[中文说明](README.zh-CN.md)

Official implementation of **SFDS-DETR: Spatial-Frequency Directional Synergy for Small Object Detection in Aerial Images**.

SFDS-DETR is built on Ultralytics RT-DETR 8.3.6. It preserves directional high-frequency details with a Frequency-Aware Feature Extraction Backbone (FAFEB) and maintains them during multi-scale fusion with an Orientation-Sensitive Pyramid Fusion Encoder (OSPFE). This code package focuses on the paper-aligned backbone and neck; bounding-box losses remain the native Ultralytics implementation.

## Highlights

- **FAFEB** uses a cascaded Haar wavelet branch alongside the spatial branch to preserve LL, HL, LH and HH information during progressive downsampling.
- **CAAE** aligns frequency features and injects them into spatial features through SE and two consecutive Channel Exchange Modulation (CEM) modules.
- **OSPFE** introduces the shallow P2 feature into a four-scale P2-P5 fusion path.
- **OSRF** employs Orientation-Sensitive Re-parameterized Convolution (OSR-Conv).
- **OSR-Conv** uses parallel 3x3, 1x1, 1x3 and 3x1 branches during training and can be fused into one 3x3 convolution for inference.

## Architecture

```text
Input
  |
  +-- FAFEB ------------------------------------------------------+
  |   Spatial branch: S1 -> S2 -> S3 -> S4 -> S5                 |
  |   Wavelet branch: F1 -> F2 -> F3 -> F4                        |
  |   Fusion: CAAE(S1,F1) ... CAAE(S4,F4) -> SF1 ... SF4         |
  +---------------------------------------------------------------+
                                  |
                                  v
  +-- OSPFE ------------------------------------------------------+
  |   AIFI -> top-down P5/P4/P3/P2 -> bottom-up P2/P3/P4/P5      |
  |   Six OSRF blocks, each built from OSR-Conv                   |
  +---------------------------------------------------------------+
                                  |
                                  v
                   Native RT-DETR Decoder and Head
```

The paper-aligned model configuration is located at:

```text
ultralytics/cfg/models/rt-detr/sfds-detr.yaml
```

Core modules:

```text
ultralytics/nn/addmodules/fafeb.py   # FAFEB, CAAE, CEM and wavelet transforms
ultralytics/nn/addmodules/ospfe.py   # OSPFE building blocks: OSRF and OSR-Conv
```

## Installation

The supplied `requirements.txt` is condensed from the original server environment. It keeps only direct project dependencies and excludes Conda internals, CUDA runtime subpackages, font libraries, parsers and packages installed transitively by PyTorch or plotting libraries.

```bash
conda create -n sfds-detr python=3.9.23 -y
conda activate sfds-detr

git clone <your-repository-url>
cd SFDS-DETR

pip install -r requirements.txt
pip install -e . --no-deps
```

The pinned PyTorch build is `torch==1.9.0+cu102`, matching the provided CUDA 10.2 server environment. For a different CUDA version, install the corresponding PyTorch and torchvision builds first, then install the remaining requirements.

Do not install another `ultralytics` package from PyPI in this environment. This repository already contains the required customized Ultralytics 8.3.6 source.

## Dataset

Datasets use the standard Ultralytics detection format:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

Example dataset configuration:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: people
  2: bicycle
```

Update the dataset path and class names for VisDrone2019, AI-TOD, UAVDT or a custom dataset.

## Training

Edit the `data` entry and other hyperparameters in `train-rtdetr.py`, then run:

```bash
python train-rtdetr.py
```

Equivalent Python usage:

```python
from ultralytics import RTDETR

model = RTDETR("ultralytics/cfg/models/rt-detr/sfds-detr.yaml")
model.train(
    data="/path/to/dataset.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    workers=4,
    device=0,
    optimizer="AdamW",
    project="runs/train",
    name="SFDS-DETR",
)
```

## Validation

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/SFDS-DETR/weights/best.pt")
metrics = model.val(
    data="/path/to/dataset.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=0,
)
print(metrics.box.map50, metrics.box.map)
```

## Inference

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/SFDS-DETR/weights/best.pt")
model.predict(
    source="/path/to/images",
    imgsz=640,
    conf=0.25,
    device=0,
    save=True,
)
```

## Export

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/SFDS-DETR/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

## Paper Results

The manuscript reports AP50 values of **49.7%** on VisDrone2019, **45.7%** on AI-TOD and **34.6%** on UAVDT, with an inference speed of **57.8 FPS**. Actual results depend on the hardware, dataset preprocessing, training configuration and evaluation protocol.

## Compatibility

- New configurations use the paper terminology: `FAFEB`, `CAAE`, `CEM`, `OSPFE`, `OSRF` and `OSRConv`.
- Legacy class names remain available so that earlier Ultralytics full-model checkpoints can still be loaded.
- The regrouped model retains the original **13,969,552 parameters** and produces identical forward output under identical initialization and input.
- No custom distillation or loss implementation is included.

## License

This project follows the [GNU AGPL-3.0 License](LICENSE) used by Ultralytics.

## Acknowledgements

This implementation is based on [Ultralytics](https://github.com/ultralytics/ultralytics), RT-DETR and `pytorch-wavelets`.
