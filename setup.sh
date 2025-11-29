#!/usr/bin/env bash
# setup.sh
# Universal setup script for the VITON-HD preprocessing repo
# - installs python deps (torch + basics)
# - clones repositories: Detectron2 (for DensePose), DensePose, SCHP, U2Net
# - downloads common pretrained artifacts (ONNX openpose model, U2Net, placeholder SCHP)
# - does basic import checks
#
# Notes:
# - You may need to adapt the torch install line (CUDA version) to your machine.
# - Detectron2 installation may require a specific torch+cuda combination; if pip install -e fails,
#   check detectron2 docs for the correct wheel for your CUDA/PyTorch.
# - Run: bash setup.sh

set -e
echo "============================================"
echo " VITON-HD Preprocessing - Universal Setup"
echo "============================================"

# 1) System packages (Ubuntu-like)
echo "[1/6] Installing system packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y git wget unzip build-essential python3-dev ffmpeg
else
  echo "[WARN] apt-get not found. Please ensure build tools, wget, git, python3-dev are installed."
fi

# 2) Python packages (core)
echo "[2/6] Installing Python packages (core)..."
python3 -m pip install --upgrade pip wheel setuptools

# NOTE: The torch line below is generic; adjust the index-url for CUDA version if you want GPU support.
# For example for CUDA 11.8, use the official index-url from PyTorch (change as needed).
python3 -m pip install torch torchvision torchaudio --upgrade || true

# base libs
python3 -m pip install \
    numpy \
    pillow \
    opencv-python \
    onnxruntime \
    matplotlib \
    tqdm \
    scikit-image \
    pycocotools \
    yacs \
    albumentations \
    cython \
    einops \
    torchvision --upgrade

# 3) Create repo layout
ROOT_DIR="$(pwd)"
REPO_DIR="${ROOT_DIR}/repositories"
MODELS_DIR="${ROOT_DIR}/models"
mkdir -p "${REPO_DIR}"
mkdir -p "${MODELS_DIR}"

# 4) Clone required repos
echo "[3/6] Cloning required repositories into ${REPO_DIR}..."
cd "${REPO_DIR}"

if [ ! -d "detectron2" ]; then
  echo "Cloning Detectron2 (as a placeholder - recommended to follow Detectron2 install docs if pip install fails)..."
  git clone https://github.com/facebookresearch/detectron2.git detectron2 || true
fi

if [ ! -d "DensePose" ]; then
  echo "Cloning DensePose..."
  git clone https://github.com/facebookresearch/DensePose.git DensePose || true
fi

if [ ! -d "SCHP" ]; then
  echo "Cloning SCHP (human parsing repo)..."
  git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git SCHP || true
fi

if [ ! -d "U-2-Net" ]; then
  echo "Cloning U-2-Net (cloth mask)..."
  git clone https://github.com/xuebinqin/U-2-Net.git "U-2-Net" || true
fi

cd "${ROOT_DIR}"

# 5) Install detectron2 & densepose (editable)
echo "[4/6] Installing detectron2 and DensePose (editable mode). If this fails, follow detectron2 installation docs."
# Prefer pip installation of a matching detectron2 wheel if you know CUDA/PyTorch versions.
# Fallback to editable install from source (may require additional system libs)
python3 -m pip install -e "${REPO_DIR}/detectron2" || {
  echo "[WARN] pip install -e detectron2 failed — try installing detectron2 wheel matching your CUDA/PyTorch manually."
}

echo "[4/6] Installing DensePose (editable)..."
python3 -m pip install -e "${REPO_DIR}/DensePose" || {
  echo "[WARN] pip install -e DensePose failed — try installing dependencies manually."
}

# 6) Download recommended models (ONNX OpenPose BODY-25, U2Net)
echo "[5/6] Downloading recommended models into ${MODELS_DIR}..."

# 6.1 OpenPose BODY-25 ONNX (community converted) - placeholder URL
OPENPOSE_ONNX="${MODELS_DIR}/openpose_body_25.onnx"
if [ ! -f "${OPENPOSE_ONNX}" ]; then
  echo "Downloading example OpenPose BODY-25 ONNX (sample URL)."
  # You should replace this with a real stable URL or add your own model to models/
  # Example placeholder below (will likely 404) — replace with your own hosting or release.
  curl -L -o "${OPENPOSE_ONNX}" "https://github.com/TMElyralab/OpenPose-Body25-ONNX/releases/download/v1.0/openpose_body_25.onnx" || true
  if [ ! -f "${OPENPOSE_ONNX}" ]; then
    echo "[WARN] OpenPose ONNX did not download automatically. Place openpose_body_25.onnx into ${MODELS_DIR} manually."
  fi
else
  echo "OpenPose ONNX already exists."
fi

# 6.2 U2Net checkpoint
U2NET_PTH="${MODELS_DIR}/u2net.pth"
if [ ! -f "${U2NET_PTH}" ]; then
  echo "Downloading U2Net checkpoint (u2netp recommended)..."
  # Replace URL if you have your own hosting
  curl -L -o "${U2NET_PTH}" "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth" || true
  if [ ! -f "${U2NET_PTH}" ]; then
    echo "[WARN] U2Net checkpoint not downloaded automatically. Place your u2net checkpoint (u2netp.pth or u2net.pth) in ${MODELS_DIR}."
  fi
else
  echo "U2Net checkpoint already exists."
fi

# 6.3 SCHP placeholder checkpoint
SCHP_PTH="${MODELS_DIR}/schp.pth"
if [ ! -f "${SCHP_PTH}" ]; then
  echo "No SCHP checkpoint downloaded automatically. If you have schp pretrained weights, place them in ${SCHP_PTH}."
else
  echo "SCHP checkpoint found."
fi

# 7) Add repositories to PYTHONPATH via a helper env file (do not modify ~/.bashrc here)
ENV_FILE="${ROOT_DIR}/env.sh"
echo "[6/6] Creating helper env file: ${ENV_FILE}"
cat > "${ENV_FILE}" <<EOF
# env.sh - add repo roots to PYTHONPATH for this shell
export VITON_PREPROCESS_ROOT="${ROOT_DIR}"
export PYTHONPATH="\$PYTHONPATH:${REPO_DIR}/DensePose:${REPO_DIR}/detectron2:${REPO_DIR}/SCHP:${REPO_DIR}/U-2-Net:${ROOT_DIR}"
EOF

echo "============================================"
echo " Setup completed (possibly with warnings)."
echo ""
echo " Important next steps:"
echo "  1) Inspect ${MODELS_DIR} and place missing model files there (openpose_body_25.onnx, u2net.pth, schp.pth)."
echo "  2) Source the env helper in any shell or add it to your environment:"
echo "      source env.sh"
echo "  3) If detectron2 install failed, follow official install instructions matching your PyTorch/CUDA:"
echo "      https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md"
echo "  4) If you will use GPU/onnxruntime-gpu, install onnxruntime-gpu and ensure CUDA drivers are installed."
echo ""
echo " To test basic Python imports, run:"
echo "   python3 -c \"import onnxruntime, cv2, torch; print('onnxruntime, cv2, torch OK')\""
echo ""
echo " Repository ready at: ${ROOT_DIR}"
echo "============================================"
