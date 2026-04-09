# YOLOv26-GPR-MedSAM — System Flowchart

## 1. Overall Pipeline

```mermaid
flowchart TD
    A[GPR Survey\n3 scan orientations] --> B[Image Preprocessing\nstack → 9-channel B-scan]
    B --> C[YOLOv26-GPR-MedSAM\nDetection Model]
    C --> D[Bounding Box Output\nVoid / Cavity location]
    D --> E[Civil Engineering Report\nsubsurface void map]

    style A fill:#e8f4f8,stroke:#2196F3
    style C fill:#fff3e0,stroke:#FF9800
    style E fill:#e8f5e9,stroke:#4CAF50
```

---

## 2. Model Architecture

```mermaid
flowchart TD
    INPUT["Input\n9-channel triple GPR\n640×640"]

    subgraph MEDSAM["MedSAMFPN  (Layer 0 — passthrough)"]
        MS1["resize 9ch→3ch\nConv 1×1 adapter"]
        MS2["resize to 512×512\nbilinear interpolation"]
        MS3["MedSAM ViT-B\n12 transformer blocks\npatch16  dim=768"]
        MS4["forward hooks"]
        MS5["P3 cache\nblock 4"]
        MS6["P4 cache\nblock 8"]
        MS7["P5 cache\nblock 12"]
        MS1 --> MS2 --> MS3 --> MS4
        MS4 --> MS5
        MS4 --> MS6
        MS4 --> MS7
    end

    subgraph CNN["YOLOv26 CNN Backbone"]
        C1["Conv 64 s2\nP1/2"]
        C2["Conv 128 s2 dil=2\nP2/4"]
        C3["C3k2 256\n"]
        C4["Conv 256 s2\nP3/8"]
        C5["C3k2 512"]
        CF3["MedSAMCrossFusion p3\nCNN Q × MedSAM K,V\nγ·attn residual"]
        C7["Conv 512 s2\nP4/16"]
        C8["A2C2f 512"]
        CF4["MedSAMCrossFusion p4\nCNN Q × MedSAM K,V\nγ·attn residual"]
        C10["Conv 1024 s2\nP5/32"]
        C11["A2C2f 1024"]
        CF5["MedSAMCrossFusion p5\nCNN Q × MedSAM K,V\nγ·attn residual"]

        C1 --> C2 --> C3 --> C4 --> C5 --> CF3
        CF3 --> C7 --> C8 --> CF4
        CF4 --> C10 --> C11 --> CF5
    end

    subgraph HEAD["FPN Head"]
        UP1["Upsample ×2"]
        CAT1["Concat P4"]
        A2C1["A2C2f 512"]
        UP2["Upsample ×2"]
        CAT2["Concat P3"]
        A2C2["A2C2f 256\nP3 out"]
        DW1["Conv s2"]
        CAT3["Concat P4"]
        A2C3["A2C2f 512\nP4 out"]
        DW2["Conv s2"]
        CAT4["Concat P5"]
        C3K["C3k2 1024\nP5 out"]

        CF5 --> UP1 --> CAT1 --> A2C1
        CF4 --> CAT1
        A2C1 --> UP2 --> CAT2 --> A2C2
        CF3 --> CAT2
        A2C2 --> DW1 --> CAT3 --> A2C3
        A2C1 --> CAT3
        A2C3 --> DW2 --> CAT4 --> C3K
        CF5 --> CAT4
    end

    DETECT["Detect\nP3 + P4 + P5\nnc=1 Cavity"]

    INPUT --> MEDSAM
    INPUT --> CNN
    MS5 --> CF3
    MS6 --> CF4
    MS7 --> CF5
    A2C2 --> DETECT
    A2C3 --> DETECT
    C3K  --> DETECT

    style MEDSAM fill:#e3f2fd,stroke:#1565C0
    style CNN fill:#fff8e1,stroke:#F57F17
    style HEAD fill:#f3e5f5,stroke:#6A1B9A
    style DETECT fill:#e8f5e9,stroke:#2E7D32
```

---

## 3. MedSAMCrossFusion Detail

```mermaid
flowchart LR
    CNN_FEAT["CNN feature map\nB × C × H × W"]
    MED_CACHE["MedSAM cache\nB × 768 × h × w"]

    PROJ["Conv 1×1\n768 → C\n+ BN + SiLU"]
    RESIZE["interpolate\nh,w → H,W"]
    FLAT_Q["flatten → transpose\nB × N × C"]
    FLAT_KV["flatten → transpose\nB × N × C"]
    NORM_Q["LayerNorm Q"]
    NORM_KV["LayerNorm K,V"]
    ATTN["MultiheadAttention\nQ=CNN  K=V=MedSAM\nbatch_first=True"]
    OUT_PROJ["Conv 1×1\nout_proj"]
    GAMMA["× γ\n(init=0)"]
    ADD["residual add\nout = CNN + γ·attn"]

    CNN_FEAT --> FLAT_Q --> NORM_Q --> ATTN
    MED_CACHE --> PROJ --> RESIZE --> FLAT_KV --> NORM_KV --> ATTN
    ATTN --> OUT_PROJ --> GAMMA --> ADD
    CNN_FEAT --> ADD

    style GAMMA fill:#ffccbc,stroke:#BF360C
    style ADD fill:#e8f5e9,stroke:#2E7D32
```

---

## 4. Training Phases

```mermaid
flowchart TD
    START([Start Training])

    subgraph P1["Phase 1  —  MedSAM FROZEN"]
        P1A["Load yolov26_gpr_medsam.yaml\n90.7M total params"]
        P1B["Freeze MedSAM ViT-B\n87.3M frozen"]
        P1C["Train CNN + CrossFusion\n3.4M trainable"]
        P1D["AdamW  lr=0.002\n100 epochs  batch=16"]
        P1E["γ: 0.0 → learns gradually\nhow much MedSAM to use"]
        P1A --> P1B --> P1C --> P1D --> P1E
    end

    subgraph AUG["GPR-Safe Augmentation"]
        A1["fliplr=0.5   ✓ horizontal OK"]
        A2["flipud=0.0   ✗ NO vertical\ndepth ordering"]
        A3["degrees=0.0  ✗ NO rotation\nhyperbola shape"]
        A4["hsv_h/s=0.0  ✗ NO color\nGPR is amplitude"]
        A5["copy_paste=0.3  paste voids\nhelps rare class"]
    end

    subgraph P2["Phase 2  —  MedSAM UNFROZEN  (optional)"]
        P2A["Load best.pt from Phase 1"]
        P2B["Unfreeze MedSAM ViT-B\n90.7M trainable"]
        P2C["AdamW  lr=0.0002\n50 epochs  batch=8"]
        P2D["ViT adapts GPR domain\nfrom medical domain"]
        P2A --> P2B --> P2C --> P2D
    end

    subgraph OUT["Output"]
        O1["best.pt\nruns/gpr_medsam/phase1_s/"]
        O2["best.pt\nruns/gpr_medsam/phase2_s/"]
        O3["results.csv\nP  R  mAP50  mAP50-95"]
    end

    START --> P1
    P1 --> AUG
    AUG --> CKPT{Phase 2?}
    CKPT -- No --> O1
    CKPT -- Yes --> P2 --> O2
    O1 --> O3
    O2 --> O3

    style P1 fill:#e3f2fd,stroke:#1565C0
    style P2 fill:#fce4ec,stroke:#880E4F
    style AUG fill:#f9fbe7,stroke:#558B2F
    style OUT fill:#e8f5e9,stroke:#2E7D32
```

---

## 5. Domain Transfer Rationale

```mermaid
flowchart LR
    subgraph MEDSAM_DOMAIN["MedSAM Training Domain"]
        US["Ultrasound\npulse-echo wave\nhyperbolic artifacts\nspeckle noise"]
        CT["CT scan\ncross-section\nvoid detection"]
        MRI["MRI\ninterface reflections\ntissue boundaries"]
    end

    subgraph GPR_DOMAIN["GPR Target Domain"]
        GPR["GPR B-scan\npulse-echo EM\nhyperbolic diffraction\nclutter noise"]
        VOID["Void / Cavity\nsignal dropout\ndepth layering"]
    end

    TRANSFER["Feature Transfer\n↓ domain gap\nvs DINOv3 (natural photos)"]

    US -- "same physics" --> GPR
    CT -- "structural void detection" --> VOID
    MEDSAM_DOMAIN --> TRANSFER --> GPR_DOMAIN

    style MEDSAM_DOMAIN fill:#e3f2fd,stroke:#1565C0
    style GPR_DOMAIN fill:#fff3e0,stroke:#E65100
    style TRANSFER fill:#e8f5e9,stroke:#2E7D32
```

---

## 6. Training Commands

```bash
# Phase 1 only (recommended first run)
python train_medsam.py --scale s --epochs 100 --batch 16

# Phase 1 + Phase 2
python train_medsam.py --scale s --epochs 100 --phase2

# Larger model
python train_medsam.py --scale m --epochs 150 --batch 8 --phase2

# Skip Phase 1, start Phase 2 from checkpoint
python train_medsam.py --weights runs/gpr_medsam/phase1_s/weights/best.pt --phase2

# Compare MedSAM vs DINOv3 (same hyperparams)
python train_medsam.py --compare --scale s --epochs 100
```
