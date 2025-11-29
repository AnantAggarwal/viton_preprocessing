#!/bin/bash
set -e
echo "======================================="
echo " VITON-HD (Kaggle) environment bootstrap"
echo "======================================="

# ---------------------------
# Notes for Kaggle users:
# - This script avoids building CMU OpenPose (too heavy for Kaggle).
# - We install OpenPifPaf (pip) for pose estimation (compatible alternative).
# - We install Detectron2 (build from source if necessary) and DensePose (editable).
# - We install SCHP and U2Net repos and download pretrained weights.
# - After running: RESTART THE KERNEL in Kaggle UI.
# ---------------------------

# 0) Ensure apt packages (Kaggle allows apt)
echo "[1/8] Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y build-essential cmake git libatlas-base-dev libopencv-dev pkg-config

# 1) Upgrade pip/setuptools/wheel
echo "[2/8] Upgrading pip, setuptools..."
python -m pip install --upgrade pip setuptools wheel

# 2) Install core Python libs (avoid forcing torch if Kaggle already has a good version)
echo "[3/8] Installing core Python packages..."
python -m pip install numpy pillow opencv-python scikit-image tqdm matplotlib cython pycocotools torchvision --upgrade

# 3) Install OpenPifPaf for pose (lightweight, pip installable)
echo "[4/8] Installing OpenPifPaf (pose/keypoints alternative to OpenPose)..."
python -m pip install git+https://github.com/openpifpaf/openpifpaf.git

# 4) Install Detectron2 (DensePose depends on Detectron2)
#    We try to install a compatible Detectron2 wheel; if that fails we fallback to building from source.
echo "[5/8] Installing Detectron2 (required for DensePose)..."
PYTORCH_VERSION=$(python - <<PY
import torch,sys
v = torch.__version__
print(v)
PY)
echo "Detected torch version: $PYTORCH_VERSION"

# Try to install a prebuilt detectron2 wheel for common CUDA; fall back to git source if pip wheel unavailable.
# Kaggle typically uses CUDA 11.x; try cu118 wheel first.
set +e
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' --quiet
if [ $? -ne 0 ]; then
  echo "[WARN] pip-install detectron2 failed — attempting build-from-source via git (may take time)"
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
fi
set -e

# 5) Clone DensePose and install editable
echo "[6/8] Cloning and installing DensePose (editable mode)..."
if [ ! -d "DensePose" ]; then
  git clone https://github.com/facebookresearch/DensePose.git
fi
python -m pip install -e DensePose

# 6) Clone SCHP (human parsing used by VITON-HD) and install its requirements
echo "[7/8] Cloning SCHP (human parsing) ..."
if [ ! -d "Self-Correction-Human-Parsing" ] && [ ! -d "SCHP" ]; then
  git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git SCHP
fi

# install SCHP python requirements if provided
if [ -f "SCHP/requirements.txt" ]; then
  python -m pip install -r SCHP/requirements.txt
else
  python -m pip install torchvision scikit-image
fi

# 7) Clone U-2-Net (cloth mask) and make importable
echo "[8/8] Cloning U-2-Net (U²-Net)..."
if [ ! -d "U-2-Net" ]; then
  git clone https://github.com/xuebinqin/U-2-Net.git U-2-Net
fi
python -m pip install -r U-2-Net/requirements.txt || true

# 8) Download commonly used pretrained models (u2net, schp, viton agnostic placeholders)
echo "[Downloading pretrained weights - this may take a minute]"
mkdir -p pretrained_models/u2net
if [ ! -f "pretrained_models/u2net/u2netp.pth" ]; then
  wget -q -O pretrained_models/u2net/u2netp.pth \
    https://huggingface.co/levihsu/O-VITON/resolve/main/pretrained_models/u2net/u2netp.pth || true
fi

mkdir -p pretrained_models/schp
if [ ! -f "pretrained_models/schp/schp.pth" ]; then
  # try common SCHP weight (if available). If not available, user can upload to Kaggle dataset.
  echo "[INFO] SCHP weight not included automatically; place schp.pth in pretrained_models/schp/ if needed."
fi

mkdir -p pretrained_models/vitonhd
echo "[INFO] If you have official VITON-HD agnostic/parse weights, upload to pretrained_models/vitonhd/"

# 9) Make repos importable by adding to PYTHONPATH (persist in ~/.bashrc)
REPO_DIR=$(pwd)
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/DensePose" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/SCHP" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/U-2-Net" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR" >> ~/.bashrc

# Source for current shell (note: Kaggle still requires kernel restart for Jupyter)
source ~/.bashrc || true

# 10) Quick import test
echo "=================================="
echo " Running quick import tests (Python)"
echo "=================================="
python - <<'PYTEST'
print("Python:", __import__("sys").version.splitlines()[0])
ok = True
try:
    import openpifpaf
    print("[OK] openpifpaf (pose) import")
except Exception as e:
    ok = False
    print("[FAIL] openpifpaf import:", e)

try:
    import detectron2
    print("[OK] detectron2 import")
except Exception as e:
    ok = False
    print("[FAIL] detectron2 import (DensePose relies on this):", e)

try:
    import densepose
    print("[OK] densepose import")
except Exception as e:
    ok = False
    print("[FAIL] densepose import:", e)

try:
    import SCHP
    print("[OK] SCHP package import (module name may vary)")
except Exception as e:
    # Try a common module import fallback
    try:
        import networks
        print("[OK] SCHP networks import")
    except Exception as e2:
        ok = False
        print("[FAIL] SCHP import (try adjusting PYTHONPATH):", e2)

try:
    # U-2-Net's model file is typically named "model.py" and provides U2NET / U2NETP classes
    from U_2_Net import model as u2model  # try a namespaced import
    print("[INFO] Attempted U-2-Net import via package alias (may need minor path fixes).")
except Exception:
    try:
        # fallback to direct model import if repo root exposes model.py
        from model import U2NETP
        print("[OK] U2NETP import")
    except Exception as e:
        ok = False
        print("[FAIL] U2Net import (you may need to add U-2-Net path to sys.path in notebooks):", e)

if ok:
    print("All main imports succeeded (or partial). You must RESTART the kernel for changes to take effect.")
else:
    print("One or more imports failed. See messages above. You may still proceed but expect to adjust PYTHONPATH or upload pretrained weights.")

PYTEST

echo "======================================="
echo " Done. IMPORTANT: Restart Kaggle kernel "
echo " (Kernel -> Restart) before running code."
echo "======================================="
