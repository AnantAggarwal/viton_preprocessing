#!/bin/bash

echo "============================"
echo " Setting up VITON-HD Pipeline"
echo "============================"

#-------------------------------------
# 0. Install system dependencies
#-------------------------------------
sudo apt update -y
sudo apt install -y build-essential cmake git libopencv-dev python3-dev python3-pip \
                     libgoogle-glog-dev libboost-all-dev libatlas-base-dev

pip install --upgrade pip

#-------------------------------------
# 1. Install Python dependencies
#-------------------------------------
pip install numpy pillow opencv-python matplotlib tqdm scipy pycocotools \
            torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install cython wheel torchvision albumentations einops timm

#-------------------------------------
# 2. Clone repositories
#-------------------------------------

echo "[Clone] OpenPose"
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

echo "[Clone] DensePose"
git clone https://github.com/facebookresearch/DensePose.git

echo "[Clone] SCHP Human Parsing"
git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git SCHP

echo "[Clone] U2Net for cloth-mask"
git clone https://github.com/xuebinqin/U-2-Net.git U-2-Net

#-------------------------------------
# 3. Build OpenPose with Python API
#-------------------------------------
echo "[Build] OpenPose"

cd openpose
git submodule update --init --recursive

mkdir -p build
cd build

# USE GPU BUILD
cmake -DBUILD_PYTHON=ON -DWITH_CUDA=ON -DUSE_OPENCV=ON ..

make -j$(nproc)

cd ../..

#-------------------------------------
# 4. Install Detectron2 (for DensePose)
#-------------------------------------

echo "[Install] Detectron2"
pip install 'git+https://github.com/facebookresearch/detectron2.git'

#-------------------------------------
# 5. Install DensePose
#-------------------------------------
pip install -r DensePose/requirements.txt

#-------------------------------------
# 6. Add everything to PYTHONPATH
#-------------------------------------
echo "[Fix] Updating PYTHONPATH"

REPO_DIR=$(pwd)

echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/openpose/build/python" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/DensePose" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/SCHP" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR/U-2-Net" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_DIR" >> ~/.bashrc

source ~/.bashrc

#-------------------------------------
# 7. Install SCHP requirements
#-------------------------------------
pip install -r SCHP/requirements.txt

#-------------------------------------
# 8. Install U2Net dependencies
#-------------------------------------
pip install scikit-image

#-------------------------------------
# 9. Test imports
#-------------------------------------
echo "============================"
echo " Running import tests"
echo "============================"

python3 <<EOF
print("Testing imports...")

try:
    from openpose import pyopenpose as op
    print("[OK] OpenPose import")
except Exception as e:
    print("[FAIL] OpenPose:", e)

try:
    import detectron2
    print("[OK] Detectron2 import")
except Exception as e:
    print("[FAIL] Detectron2:", e)

try:
    import densepose
    print("[OK] DensePose import")
except Exception as e:
    print("[FAIL] DensePose:", e)

try:
    import networks  # SCHP networks
    print("[OK] SCHP import")
except Exception as e:
    print("[FAIL] SCHP:", e)

try:
    from model import U2NET
    print("[OK] U2Net import")
except Exception as e:
    print("[FAIL] U2Net:", e)

print("Setup Complete.")
EOF

echo "======================================="
echo " VITON-HD Preprocessing Setup Completed "
echo "======================================="
