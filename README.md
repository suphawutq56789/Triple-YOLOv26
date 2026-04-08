<div align="center">

# YOLOv26-GPR + DINOv3

### Multi-Scale Cross-Attention Architecture for GPR Subsurface Void Detection

**Research Group, Department of Civil Engineering**
**King Mongkut's University of Technology Thonburi (KMUTT)**

<p align="center">
  <img src="assets/yolov26_triple_dinov3_architecture.svg" width=90%>
  <br><em>YOLOv26-GPR: Triple Input + DINOv3 Multi-Scale Cross-Attention FPN</em>
</p>

</div>

---

## Overview

This repository implements **YOLOv26-GPR**, a novel object detection architecture designed for **Ground Penetrating Radar (GPR) subsurface void detection** in civil engineering inspection.

The key contribution is a genuine architectural innovation: **DINOv3 features are injected at three FPN levels (P3/P4/P5) via cross-attention**, enabling the CNN detector to leverage rich pretrained ViT features at every scale. This is critical for detecting subsurface voids on small datasets (~2,700 images).

---

## Architecture

### Core Innovation: DINOv3 Multi-Scale Cross-Attention FPN

```
9-ch Triple GPR Input (3 scan orientations x 3ch)
        |
        v
+-----------------------------+
|  DINOv3FPN  (layer 0)       |  <- DINOv3-vits16 runs ONCE
|  Caches P3/P4/P5 ViT feats  |     Passthrough (output = input)
+-----------------------------+
        |
        v
  CNN Backbone  (Stem -> P3 -> P4 -> P5)
        |              |          |
        v              v          v
  DINOv3CrossFusion  (x3, one per FPN level)
  CNN(Q) x DINOv3(K,V) cross-attention
  gamma-gated residual (gamma=0 init -> stable training)
        |
        v
  PAFPN Neck -> Detect Head -> Cavity boxes
```

**Previous approach (YOLOv12-rename):** DINOv3 -> 64ch -> CNN (sequential, single scale only)

**This work:** DINOv3 cached features injected at P3 + P4 + P5 via cross-attention (multi-scale, parallel)

### New Modules

| Module | Role |
|---|---|
| `DINOv3FPN` | Runs DINOv3 once; caches multi-scale ViT features; passthrough |
| `DINOv3CrossFusion` | CNN (Q) x DINOv3 (K,V) cross-attention; gamma-gated residual |

---

## Application: GPR Void Detection

**Task:** Detect subsurface voids (cavities) in concrete/road structures using GPR B-scan images.

**Input format:** 9-channel triple input -- 3 GPR scan orientations x 3 channels each

**Dataset:** ~910 images per orientation x 3 orientations = ~2,700 total images

**Class:** `Cavity` (void/hollow beneath surface)

GPR B-scan characteristics this architecture addresses:
- Hyperbolic reflection patterns from subsurface objects
- Depth-ordered signal (vertical axis = time/depth)
- Low contrast, high noise environments
- Small dataset size -> requires strong pretrained features

---

## Model Variants

| Scale | Total Params | Trainable | Frozen (DINOv3) | Recommended for |
|---|---|---|---|---|
| `n` | 25.3M | 3.3M | 22M | Fast testing |
| `s` | 28.6M | 6.5M | 22M | **Training ~2700 imgs** |
| `m` | 41.2M | 19.1M | 22M | Higher accuracy |

DINOv3 backbone options:

| Backbone | dim | Pretrain data |
|---|---|---|
| `dinov3-vits16-pretrain-lvd1689m` | 384 | LVD-1.6B images (default) |
| `dinov3-vitb16-pretrain-lvd1689m` | 768 | LVD-1.6B images |
| `dinov3-vitl16-pretrain-sat493m`  | 1024 | Satellite 493M (for survey GPR) |

---

## Installation

```bash
git clone https://github.com/suphawutq56789/Triple-YOLOv26-DINOv3-.git
cd Triple-YOLOv26-DINOv3-

pip install -r requirements.txt
```

Or minimal install:

```bash
pip install torch torchvision ultralytics transformers timm huggingface_hub
```

---

## Training

### Quick start

```bash
# Phase 1 only (DINOv3 frozen, 100 epochs)
python train_gpr.py

# Full training: phase 1 then phase 2 (unfreeze DINOv3 fine-tune)
python train_gpr.py --phase2

# Larger model
python train_gpr.py --scale m --epochs 120 --phase2
```

### Data config (`data_all.yaml`)

```yaml
path: /path/to/your/GPR/dataset

train: images/primary/train
val:   images/primary/val
test:  images/primary/test

triple_input: true
nc: 1
names: ['Cavity']
```

### Two-phase training strategy

**Phase 1** -- DINOv3 ViT frozen, train CNN + CrossFusion layers:
- Optimizer: AdamW, LR = 0.002, 100 epochs, warmup 5 epochs
- Augmentation: horizontal flip, mosaic, copy-paste void patches

**Phase 2** -- Unfreeze DINOv3, fine-tune entire model:
- LR = 0.0002 (10x lower than phase 1), 50 epochs
- More conservative augmentation

### GPR-specific augmentation rules

| Augmentation | Setting | Reason |
|---|---|---|
| Horizontal flip | `fliplr=0.5` | Hyperbola is left-right symmetric |
| Vertical flip | `flipud=0.0` | DISABLED -- destroys depth ordering |
| Rotation | `degrees=0.0` | DISABLED -- destroys hyperbola shape |
| Brightness | `hsv_v=0.2` | Simulate gain variation OK |
| Copy-paste | `copy_paste=0.3` | Augment rare void class |

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("runs/gpr/phase1_s/weights/best.pt")

# Single image
results = model.predict("gpr_bscan.jpg", conf=0.25)

# With test-time augmentation (recommended)
results = model.predict("gpr_bscan.jpg", augment=True, conf=0.25)

# Batch predict with save
results = model.predict("path/to/test/images/", conf=0.25, save=True)
```

---

## GPR Preprocessing (recommended)

Apply before feeding B-scans to the model:

```python
import numpy as np
import cv2

def preprocess_gpr_bscan(img: np.ndarray) -> np.ndarray:
    """Background removal + Automatic Gain Control for GPR B-scans."""
    img = img.astype(np.float32)

    # 1. Background removal (remove direct wave / air arrival)
    img -= img.mean(axis=0, keepdims=True)

    # 2. AGC: compensate signal amplitude decay with depth
    for col in range(img.shape[1]):
        rms = np.sqrt(np.mean(img[:, col] ** 2)) + 1e-8
        img[:, col] /= rms

    # 3. Normalize to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)
```

---

## Repository Structure

```
Triple-YOLOv26-DINOv3/
|-- ultralytics/
|   |-- cfg/models/v26/
|   |   |-- yolov26_gpr.yaml              <- Main GPR architecture (NEW)
|   |   `-- yolov26_triple_dinov3*.yaml   <- Other variants
|   `-- nn/modules/
|       `-- dinov3.py                     <- DINOv3FPN + DINOv3CrossFusion (NEW)
|-- train_gpr.py                          <- Two-phase training script (NEW)
|-- data_all.yaml                         <- Dataset config
`-- README.md
```

---

## Citation

```bibtex
@misc{yolov26gpr2025,
  title     = {YOLOv26-GPR: Multi-Scale DINOv3 Cross-Attention for GPR Void Detection},
  author    = {KMUTT Civil Engineering Research Group},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/suphawutq56789/Triple-YOLOv26-DINOv3-}
}
```

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) -- base detection framework
- [DINOv3 (Meta AI)](https://github.com/facebookresearch/dinov3) -- pretrained ViT backbone
- [triple_YOLO13](https://github.com/Sompote/triple_YOLO13) -- triple input concept

---

<div align="center">
<b>Department of Civil Engineering, KMUTT</b><br>
GPR Subsurface Void Detection Research
</div>
