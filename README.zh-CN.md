# SFDS-DETR

[English](README.md)

本仓库是论文 **“SFDS-DETR: Spatial-Frequency Directional Synergy for Small Object Detection in Aerial Images”** 的实现代码。

SFDS-DETR 基于 Ultralytics RT-DETR 8.3.6 构建。模型使用频率感知特征提取主干 FAFEB 保留方向性高频细节，并通过方向敏感金字塔融合编码器 OSPFE 在多尺度融合过程中持续强化这些细节。本代码包主要对齐论文中的主干与颈部，边界框回归损失保持 Ultralytics 原生实现。

## 主要特点

- **FAFEB**：空间分支与级联 Haar 小波分支并行，在逐级下采样时保留 LL、HL、LH 和 HH 子带信息。
- **CAAE**：完成频率特征对齐，并通过 SE 和两个连续的通道交换调制模块 CEM 将频率细节注入空间特征。
- **OSPFE**：将浅层 P2 特征加入 P2-P5 四尺度特征融合路径。
- **OSRF**：采用方向敏感重参数化卷积 OSR-Conv 构建特征融合块。
- **OSR-Conv**：训练阶段包含 3x3、1x1、1x3 和 3x1 四个并行分支，推理阶段可融合为单个 3x3 卷积。

## 模型结构

```text
输入
  |
  +-- FAFEB ------------------------------------------------------+
  |   空间分支：S1 -> S2 -> S3 -> S4 -> S5                       |
  |   小波分支：F1 -> F2 -> F3 -> F4                              |
  |   特征融合：CAAE(S1,F1) ... CAAE(S4,F4) -> SF1 ... SF4       |
  +---------------------------------------------------------------+
                                  |
                                  v
  +-- OSPFE ------------------------------------------------------+
  |   AIFI -> 自顶向下 P5/P4/P3/P2 -> 自底向上 P2/P3/P4/P5      |
  |   共使用 6 个 OSRF，每个 OSRF 内部采用 OSR-Conv              |
  +---------------------------------------------------------------+
                                  |
                                  v
                      原生 RT-DETR 解码器与检测头
```

论文命名对齐后的模型配置：

```text
ultralytics/cfg/models/rt-detr/sfds-detr.yaml
```

核心模块：

```text
ultralytics/nn/addmodules/fafeb.py   # FAFEB、CAAE、CEM 与小波变换
ultralytics/nn/addmodules/ospfe.py   # OSPFE 基本模块 OSRF 与 OSR-Conv
```

## 环境安装

`requirements.txt` 根据服务器环境进行了精简，只保留项目直接依赖。Conda 内部组件、CUDA 运行时子包、字体库、解析器以及会被 PyTorch 或绘图库自动安装的传递依赖均未单独列出。

```bash
conda create -n sfds-detr python=3.9.23 -y
conda activate sfds-detr

git clone <你的仓库地址>
cd SFDS-DETR

pip install -r requirements.txt
pip install -e . --no-deps
```

依赖文件固定为 `torch==1.9.0+cu102`，与服务器上的 CUDA 10.2 环境一致。如果使用其他 CUDA 版本，请先安装对应版本的 PyTorch 和 torchvision，再安装其余依赖。

请不要额外执行 `pip install ultralytics`。本仓库已经包含定制后的 Ultralytics 8.3.6 源代码，额外安装可能导致 Python 优先加载错误版本。

## 数据集准备

数据集采用标准 Ultralytics 检测格式：

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

数据集配置示例：

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

请根据 VisDrone2019、AI-TOD、UAVDT 或自定义数据集修改路径和类别名称。

## 模型训练

修改 `train-rtdetr.py` 中的 `data` 路径和训练参数，然后执行：

```bash
python train-rtdetr.py
```

也可以直接通过 Python API 训练：

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

## 模型验证

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

## 模型推理

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

## 模型导出

```python
from ultralytics import RTDETR

model = RTDETR("runs/train/SFDS-DETR/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)
```

## 论文结果

论文报告的 AP50 分别为：VisDrone2019 **49.7%**、AI-TOD **45.7%**、UAVDT **34.6%**，推理速度为 **57.8 FPS**。实际复现结果会受到硬件、数据预处理、训练参数和评估流程影响。

## 兼容性说明

- 新模型配置采用论文术语：`FAFEB`、`CAAE`、`CEM`、`OSPFE`、`OSRF` 和 `OSRConv`。
- 保留旧类名兼容入口，可以继续加载早期 Ultralytics 整模型 checkpoint。
- 重组后的模型仍为 **13,969,552 个参数**；在相同初始化和输入下，前向输出与重组前完全一致。
- 本代码包不包含自定义蒸馏模块，也未修改 Ultralytics 原生损失函数。

## 开源协议

本项目遵循 Ultralytics 使用的 [GNU AGPL-3.0 License](LICENSE)。

## 致谢

本项目基于 [Ultralytics](https://github.com/ultralytics/ultralytics)、RT-DETR 和 `pytorch-wavelets` 实现。
