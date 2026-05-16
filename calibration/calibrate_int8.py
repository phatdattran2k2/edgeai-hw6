#!/usr/bin/env python3
# Copyright (c) 2026 陳發達
# Tatung University — I4210 AI實務專題
"""calibration/calibrate_int8.py — Build an INT8-calibrated TensorRT engine.

From best.pt using a representative subset of the HW5/Lab 9 training set.
Run on the Jetson once; commit best_int8.engine to the repo.
"""

import tempfile
from pathlib import Path
import yaml
from ultralytics import YOLO

CAL_DATA = Path(__file__).parent / "calibration_data"
YAML_TEMPLATE = Path(__file__).parent / "calibration.yaml"
WEIGHTS = Path(__file__).parent.parent / "best.pt"
OUT = Path(__file__).parent.parent / "best_int8.engine"


def main() -> None:
    """Run the INT8 calibration and export process."""
    if not CAL_DATA.exists() or len(list(CAL_DATA.glob("*.jpg"))) < 50:
        raise SystemExit(f"Need >=50 calibration images at {CAL_DATA}")

    # Ultralytics' check_det_dataset() resolves YAML `path:` against its
    # settings.datasets_dir (default <repo>/datasets), so a relative
    # 'calibration_data' goes to the wrong place. Rewrite the YAML at
    # runtime with an absolute path computed from this script's location.
    # Keeps the committed YAML portable (relative path) and the runtime
    # behaviour correct on any machine — no hardcoded /home/<user>/...
    template = yaml.safe_load(YAML_TEMPLATE.read_text())
    template["path"] = str(CAL_DATA.resolve())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(template, tmp)
        runtime_yaml = tmp.name

    model = YOLO(str(WEIGHTS), task="detect")
    model.export(
        format="engine",
        int8=True,
        data=runtime_yaml,
        imgsz=320,
        batch=1,
        verbose=True,
    )

    src = WEIGHTS.with_suffix(".engine")
    src.rename(OUT)
    print(f"Wrote {OUT}, size = {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()