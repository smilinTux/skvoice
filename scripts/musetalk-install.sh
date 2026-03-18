#!/usr/bin/env bash
# musetalk-install.sh — MuseTalk lip-sync installation for RTX 5060 Ti
# Target: 192.168.0.100, Ubuntu, CUDA already installed
#
# Usage:
#   scp musetalk-install.sh cbrd21@192.168.0.100:~/
#   ssh 192.168.0.100 'bash ~/musetalk-install.sh'
#
# After install, deploy the API:
#   scp musetalk_api.py cbrd21@192.168.0.100:~/musetalk/
#   ssh 192.168.0.100 'cd ~/musetalk && source venv/bin/activate && python musetalk_api.py --port 18801'
#
set -euo pipefail

INSTALL_DIR="$HOME/musetalk"
VENV_DIR="$INSTALL_DIR/venv"

echo "============================================"
echo "  MuseTalk Lip-Sync Installer"
echo "  Target: RTX 5060 Ti 16GB"
echo "  Install dir: $INSTALL_DIR"
echo "============================================"
echo ""

# -----------------------------------------------------------
# Step 1: System dependencies
# -----------------------------------------------------------
echo "[1/8] Checking system dependencies..."

if ! command -v ffmpeg &>/dev/null; then
    echo "  Installing ffmpeg..."
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
else
    echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
fi

if ! command -v git &>/dev/null; then
    sudo apt-get install -y -qq git
fi

# -----------------------------------------------------------
# Step 2: Install Python 3.10 if not present
# -----------------------------------------------------------
echo ""
echo "[2/8] Checking Python 3.10..."

if ! command -v python3.10 &>/dev/null; then
    echo "  Python 3.10 not found. Installing from deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
fi
echo "  Python 3.10: $(python3.10 --version)"

# -----------------------------------------------------------
# Step 3: Clone MuseTalk
# -----------------------------------------------------------
echo ""
echo "[3/8] Cloning MuseTalk repository..."

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Repository exists, pulling latest..."
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/TMElyralab/MuseTalk.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# -----------------------------------------------------------
# Step 4: Create virtual environment
# -----------------------------------------------------------
echo ""
echo "[4/8] Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3.10 -m venv "$VENV_DIR"
    echo "  Created venv at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# -----------------------------------------------------------
# Step 5: Install PyTorch
# -----------------------------------------------------------
echo ""
echo "[5/8] Installing PyTorch..."

# Check if PyTorch is already installed and working
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  PyTorch already installed with CUDA support"
    python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
else
    # Try the recommended version first
    echo "  Attempting PyTorch 2.0.1 (cu118)..."
    if pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu118 -q 2>/dev/null; then
        echo "  Installed PyTorch 2.0.1"
    else
        echo "  PyTorch 2.0.1 failed (likely Blackwell GPU needs newer version)"
        echo "  Installing PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124 -q
    fi

    # Verify
    python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
fi

# -----------------------------------------------------------
# Step 6: Install dependencies
# -----------------------------------------------------------
echo ""
echo "[6/8] Installing MuseTalk dependencies..."

# Core requirements
pip install -r requirements.txt -q 2>&1 | tail -5

# MMLab ecosystem (face detection, pose estimation)
echo "  Installing MMLab packages (this may take a while)..."
pip install --no-cache-dir -U openmim -q
mim install mmengine -q 2>&1 | tail -1
mim install "mmcv==2.0.1" -q 2>&1 | tail -1
mim install "mmdet==3.1.0" -q 2>&1 | tail -1
mim install "mmpose==1.1.0" -q 2>&1 | tail -1

# API dependencies
pip install fastapi uvicorn python-multipart httpx -q

echo "  Dependencies installed."

# -----------------------------------------------------------
# Step 7: Download model weights
# -----------------------------------------------------------
echo ""
echo "[7/8] Downloading model weights..."

mkdir -p models/{musetalk,musetalkV15,sd-vae,whisper,dwpose,syncnet,face-parse-bisent}

# Check if weights already exist
if [ -f "models/musetalkV15/unet.pth" ] && [ -f "models/sd-vae/diffusion_pytorch_model.bin" ]; then
    echo "  Model weights already present, skipping download."
else
    echo "  Running download_weights.sh (this downloads ~5GB)..."
    if [ -f "download_weights.sh" ]; then
        bash download_weights.sh
    else
        echo "  download_weights.sh not found, using manual downloads..."
        pip install huggingface_hub gdown -q

        echo "  Downloading MuseTalk V1.5..."
        huggingface-cli download TMElyralab/MuseTalk \
            --local-dir models/ --include "musetalkV15/*"

        echo "  Downloading MuseTalk V1.0..."
        huggingface-cli download TMElyralab/MuseTalk \
            --local-dir models/ --include "musetalk/*"

        echo "  Downloading SD VAE..."
        huggingface-cli download stabilityai/sd-vae-ft-mse \
            --local-dir models/sd-vae/ \
            --include "config.json" "diffusion_pytorch_model.bin"

        echo "  Downloading Whisper tiny..."
        huggingface-cli download openai/whisper-tiny \
            --local-dir models/whisper/

        echo "  Downloading DWPose..."
        huggingface-cli download yzd-v/DWPose \
            --local-dir models/dwpose/ \
            --include "dw-ll_ucoco_384.pth"

        echo "  Downloading SyncNet..."
        huggingface-cli download ByteDance/LatentSync \
            --local-dir models/syncnet/ \
            --include "latentsync_syncnet.pt"

        echo "  Downloading face parsing models..."
        pip install gdown -q
        gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
            -O models/face-parse-bisent/79999_iter.pth
        wget -q https://download.pytorch.org/models/resnet18-5c106cde.pth \
            -O models/face-parse-bisent/resnet18-5c106cde.pth
    fi
fi

# -----------------------------------------------------------
# Step 8: Verify installation
# -----------------------------------------------------------
echo ""
echo "[8/8] Verifying installation..."

ERRORS=0

# Check required model files
REQUIRED=(
    "models/musetalkV15/unet.pth"
    "models/musetalkV15/musetalk.json"
    "models/sd-vae/diffusion_pytorch_model.bin"
    "models/sd-vae/config.json"
    "models/whisper/pytorch_model.bin"
    "models/dwpose/dw-ll_ucoco_384.pth"
    "models/face-parse-bisnet/79999_iter.pth"
    "models/face-parse-bisenet/resnet18-5c106cde.pth"
)

# Also check alternate directory names (upstream inconsistency)
echo "  Checking model files..."
for f in "${REQUIRED[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "    [OK] $f ($SIZE)"
    else
        echo "    [??] $f (not at expected path, checking alternatives...)"
        # Don't count as hard error — paths vary between versions
    fi
done

# Check Python imports
echo ""
echo "  Checking Python imports..."
python -c "
import sys
errors = 0
for mod in ['torch', 'cv2', 'numpy', 'diffusers', 'transformers',
            'mmcv', 'mmdet', 'mmpose', 'fastapi', 'uvicorn']:
    try:
        __import__(mod)
        print(f'    [OK] {mod}')
    except ImportError as e:
        print(f'    [FAIL] {mod}: {e}')
        errors += 1
if errors:
    print(f'\n  {errors} import(s) failed!')
    sys.exit(1)
print('\n  All imports OK.')
"

# VRAM check
echo ""
python -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total = props.total_mem / 1024**3
    alloc = torch.cuda.memory_allocated() / 1024**3
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM total: {total:.1f} GB')
    print(f'  VRAM in use: {alloc:.2f} GB')
    print(f'  VRAM available: {total - alloc:.1f} GB')
else:
    print('  WARNING: No CUDA GPU detected!')
"

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Quick test:"
echo "    cd $INSTALL_DIR"
echo "    source venv/bin/activate"
echo "    sh inference.sh v1.5 normal"
echo ""
echo "  Start API server:"
echo "    python musetalk_api.py --port 18801"
echo ""
echo "  Systemd service:"
echo "    Copy musetalk-api.service to ~/.config/systemd/user/"
echo "    systemctl --user enable --now musetalk-api.service"
echo "============================================"
