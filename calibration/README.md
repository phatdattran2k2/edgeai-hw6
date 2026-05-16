# Calibration README

## How to regenerate calibration data

```bash
DATASET_DIR=/home/jetson/lab9/dataset/train/images
mkdir -p calibration/calibration_data

python3 - "$DATASET_DIR" << 'PY'
import random, shutil, sys
from pathlib import Path
src = Path(sys.argv[1])
dst = Path('calibration/calibration_data')
random.seed(42)
imgs = sorted(src.glob('*.jpg'))
sample = random.sample(imgs, k=min(500, len(imgs)))
for s in sample:
    shutil.copy(s, dst / s.name)
print(f'Copied {len(sample)} images')
PY
```

## How to rebuild INT8 engine

```bash
cd ~/edgeai-hw6
python3 calibration/calibrate_int8.py
```

## Results (last calibration: 2026-05-16)

| Precision | Size | mAP@50 |
|-----------|------|--------|
| FP16 | 7 MB | 0.3322 |
| INT8 | 4 MB | 0.3332 |

Delta: +0.0010 (INT8 within threshold)
