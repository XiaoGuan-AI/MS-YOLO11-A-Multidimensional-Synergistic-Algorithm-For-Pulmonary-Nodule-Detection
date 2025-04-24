# MS-YOLO11: A Multidimensional Synergistic Algorithm For Pulmonary Nodule Detection

## 📅 Project Introduction

**MS-YOLO11** is a pulmonary nodule detection model that enhances feature expression and small object detection via Multidimensional Collaborative Attention (MCA) and Synergistic Multi Attention Transformer (SMAT). It achieves superior accuracy on the LUNA16 dataset and performs especially well when using unsegmented lung CT data.

---

## ⚠️ Original Author Statement

This project is currently under **scientific publication submission**. It is shared **only for academic and educational purposes**.

> **Do not use this project for publication or commercial purposes.**
>
> For special requests or collaborations, please contact me via email: **18313572890@163.com**

---

## 🛋️ Environment Setup

### 1. Create Conda Virtual Environment
```bash
conda create -n ms-yolo11 python=3.10 -y
conda activate ms-yolo11
```

### 2. Install PyTorch (Example: CUDA 12.1)
```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Project Dependencies
**Make sure to comment out any torch/torchaudio/torchvision entries in `requirements.txt` first.**
```bash
pip install -r requirements.txt --timeout 6000
pip install -e .
```
Use a mirror for faster installation (optional):
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Verify Installation
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Should return True
```

---

## 📁 Data Preparation

1. Use the [LUNA16](https://luna16.grand-challenge.org/) dataset (.mhd + .csv format).
2. Preprocess images to `[−1000, 400]` HU window range and convert to RGB `.jpg` format.
3. Convert annotations to YOLO format: `[class, x_center, y_center, width, height]` normalized to [0, 1].
4. Two dataset versions:
   - `Luna16_Undivided`: Original images (recommended)
   - `Luna16_Segmentation`: With lung segmentation preprocessing
5. Split dataset 9:1 into training and validation sets.

---

## 🚀 Training

### Run training with the following command:
```bash
python train.py --data data.yaml --epochs 700 --batch-size 8 --device 0 --img 640 --project runs/train --name MS-YOLO11 --patience 100
```

#### Parameters:
- `--data`: Path to `data.yaml`
- `--img`: Input size (default: 640)
- `--device`: GPU ID
- `--patience`: Early stopping after no improvement

---

## 🔮 Evaluation

Run model validation using:
```bash
python val.py --data data.yaml --weights runs/train/MS-YOLO11/weights/best.pt --img 640 --task val --conf-thres 0.001 --iou-thres 0.5
```

### Metrics:
- `mAP50`: Mean Average Precision @ IoU=0.5
- `APsmall`: Precision on small targets
- `ARsmall`: Recall on small targets

You can visualize training metrics using TensorBoard:
```bash
tensorboard --logdir runs/train
```

---

## 📃 Example Configuration Files

### `requirements.txt`
```txt
numpy
opencv-python
pandas
matplotlib
tqdm
pyyaml
seaborn
scikit-image
tensorboard
pycocotools
```

### `data.yaml`
```yaml
train: /path/to/train/images
val: /path/to/val/images

nc: 1
names: ['nodule']
```

---

## ⚙️ Helper Scripts

### `run.sh` (Linux/macOS)
```bash
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ms-yolo11

python train.py \
    --data data.yaml \
    --epochs 700 \
    --batch-size 8 \
    --img 640 \
    --project runs/train \
    --name MS-YOLO11 \
    --patience 100 \
    --device 0
```

### `run.bat` (Windows)
```bat
@echo off
call conda activate ms-yolo11

python train.py ^
    --data data.yaml ^
    --epochs 700 ^
    --batch-size 8 ^
    --img 640 ^
    --project runs/train ^
    --name MS-YOLO11 ^
    --patience 100 ^
    --device 0
pause
```

---

## 🚧 Optional: Docker Support

### Dockerfile
```Dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3.10 python3-pip python3.10-venv git libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY . /app
RUN sed -i '/torch/d' requirements.txt && pip install -r requirements.txt
CMD ["bash"]
```

### `docker-compose.yml`
```yaml
version: "3.9"
services:
  ms-yolo11:
    build: .
    image: ms-yolo11:latest
    container_name: ms-yolo11
    shm_size: '8gb'
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./datasets:/app/datasets
      - ./runs:/app/runs
    working_dir: /app
    tty: true
```

### Build & Run
```bash
docker-compose build
docker-compose run --rm ms-yolo11
```

---

## ✅ Best Practices

| Scenario           | Recommendation             |
|--------------------|-----------------------------|
| Windows users      | Use Anaconda                |
| Linux/macOS users  | Use Docker or `run.sh`      |
| Cross-platform     | Prefer Docker               |
| Limited GPU memory | Lower `--batch-size`        |

---



