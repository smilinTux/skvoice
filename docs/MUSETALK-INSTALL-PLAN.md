# MuseTalk Lip-Sync Installation Plan

**Target Machine**: 192.168.0.100 (RTX 5060 Ti 16GB, Ubuntu, CUDA installed)
**Date**: 2026-03-18
**Status**: RESEARCH — do not run without review

---

## Architecture Summary

MuseTalk is a real-time lip-sync model from TMElyralab that works by inpainting
mouth regions in latent space with a single-step UNet, conditioned on Whisper
audio embeddings. It is NOT a multi-step diffusion model — it runs one forward
pass per frame, enabling 30fps+ on decent GPUs.

**Pipeline**: Audio WAV -> Whisper feature extraction -> UNet inpainting in
latent space -> VAE decode -> blend mouth region onto portrait -> output frames

**Integration with SKVoice**: SKVoice (port 18800) generates TTS audio via
Chatterbox (port 18793). The MuseTalk API service (proposed port 18801) takes
that audio + a portrait image and returns lip-synced video frames. This enables
the Sovereign FaceTime pipeline: STT -> LLM -> TTS -> MuseTalk -> WebRTC.

---

## 1. VRAM Budget Analysis

Current GPU residents on .100:

| Service          | VRAM (est.) | Notes                        |
|------------------|-------------|------------------------------|
| Chatterbox TTS   | ~2-3 GB     | Port 18793                   |
| SenseVoice/STT   | ~1-2 GB     | faster-whisper, port 18794   |
| Ollama (idle)    | ~0 GB       | Only loads on request         |
| ComfyUI (idle)   | ~0 GB       | Only loads on request         |
| **Subtotal**     | **~3-5 GB** |                              |

MuseTalk inference VRAM:

| Component        | VRAM (est.) | Notes                        |
|------------------|-------------|------------------------------|
| VAE (fp16)       | ~0.5 GB     | sd-vae-ft-mse                |
| UNet (fp16)      | ~1.5 GB     | Single-step inpainting       |
| Whisper tiny     | ~0.3 GB     | Feature extraction only      |
| DWPose           | ~0.5 GB     | Landmark detection           |
| Face parsing     | ~0.3 GB     | BiSeNet + ResNet18           |
| Working memory   | ~1-2 GB     | Tensors, batches             |
| **Subtotal**     | **~4-5 GB** | With fp16                    |

**Total estimated**: 7-10 GB of 16 GB. This FITS with headroom.

Key notes:
- A user confirmed MuseTalk runs on a GTX 1060 3GB (slowly)
- The `--use_float16` flag is critical — always use it
- Ollama and ComfyUI should not be loaded simultaneously during video calls
- If tight, unload Ollama models first: `ollama stop`

---

## 2. Installation Script

```bash
#!/usr/bin/env bash
# musetalk-install.sh — MuseTalk lip-sync installation for RTX 5060 Ti
# Target: 192.168.0.100, Ubuntu, CUDA already installed
# Run as: bash musetalk-install.sh
set -euo pipefail

INSTALL_DIR="$HOME/musetalk"
VENV_DIR="$INSTALL_DIR/venv"
PORT=18801

echo "=== MuseTalk Installation ==="
echo "Install dir: $INSTALL_DIR"
echo "API port: $PORT"
echo ""

# -------------------------------------------------------
# Step 1: Clone the repository
# -------------------------------------------------------
if [ -d "$INSTALL_DIR" ]; then
    echo "[SKIP] $INSTALL_DIR already exists"
    cd "$INSTALL_DIR"
    git pull
else
    git clone https://github.com/TMElyralab/MuseTalk.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# -------------------------------------------------------
# Step 2: Create Python venv (Python 3.10 recommended,
#         but 3.12 may work — test first)
# -------------------------------------------------------
# MuseTalk officially supports Python 3.10. If you have 3.10:
#   python3.10 -m venv "$VENV_DIR"
# If only 3.12 is available, try it but be ready for numpy/tf issues:

if [ ! -d "$VENV_DIR" ]; then
    # Try python3.10 first, fall back to python3.12
    if command -v python3.10 &>/dev/null; then
        python3.10 -m venv "$VENV_DIR"
        echo "[OK] Created venv with Python 3.10"
    else
        echo "[WARN] Python 3.10 not found, using system python3"
        echo "[WARN] If builds fail, install python3.10:"
        echo "  sudo add-apt-repository ppa:deadsnakes/ppa"
        echo "  sudo apt install python3.10 python3.10-venv python3.10-dev"
        python3 -m venv "$VENV_DIR"
    fi
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# -------------------------------------------------------
# Step 3: Install PyTorch (CUDA 11.8 wheels work on 12.x)
# -------------------------------------------------------
# For RTX 5060 Ti (Blackwell), you may need newer PyTorch.
# Try the official recommendation first:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118 || {
    echo "[WARN] PyTorch 2.0.1 failed, trying latest stable..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

# -------------------------------------------------------
# Step 4: Install MuseTalk dependencies
# -------------------------------------------------------
pip install -r requirements.txt

# MMLab ecosystem (required for face detection/pose)
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

# -------------------------------------------------------
# Step 5: Install ffmpeg (system)
# -------------------------------------------------------
if ! command -v ffmpeg &>/dev/null; then
    echo "[INFO] Installing ffmpeg..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi
export FFMPEG_PATH=$(which ffmpeg)

# -------------------------------------------------------
# Step 6: Download model weights
# -------------------------------------------------------
echo "=== Downloading model weights ==="

# Create model directories
mkdir -p models/{musetalk,musetalkV15,sd-vae,whisper,dwpose,syncnet,face-parse-bisent}

# Use the official download script
bash download_weights.sh

# If the script fails, here are manual fallbacks:
# pip install huggingface_hub gdown
#
# # MuseTalk V1.5 UNet (primary model)
# huggingface-cli download TMElyralab/MuseTalk \
#     --local-dir models/ --include "musetalkV15/*"
#
# # MuseTalk V1.0 (optional)
# huggingface-cli download TMElyralab/MuseTalk \
#     --local-dir models/ --include "musetalk/*"
#
# # SD VAE
# huggingface-cli download stabilityai/sd-vae-ft-mse \
#     --local-dir models/sd-vae/ \
#     --include "config.json" "diffusion_pytorch_model.bin"
#
# # Whisper tiny
# huggingface-cli download openai/whisper-tiny \
#     --local-dir models/whisper/
#
# # DWPose
# huggingface-cli download yzd-v/DWPose \
#     --local-dir models/dwpose/ \
#     --include "dw-ll_ucoco_384.pth"
#
# # SyncNet
# huggingface-cli download ByteDance/LatentSync \
#     --local-dir models/syncnet/ \
#     --include "latentsync_syncnet.pt"
#
# # Face parsing (from Google Drive)
# gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face-parse-bisent/79999_iter.pth
# wget https://download.pytorch.org/models/resnet18-5c106cde.pth \
#     -O models/face-parse-bisent/resnet18-5c106cde.pth

# -------------------------------------------------------
# Step 7: Verify model directory structure
# -------------------------------------------------------
echo ""
echo "=== Model directory structure ==="
find models/ -type f | head -30
echo ""
echo "=== Verifying required files ==="
REQUIRED_FILES=(
    "models/musetalkV15/unet.pth"
    "models/sd-vae/diffusion_pytorch_model.bin"
    "models/whisper/pytorch_model.bin"
    "models/dwpose/dw-ll_ucoco_384.pth"
    "models/face-parse-bisent/79999_iter.pth"
    "models/face-parse-bisent/resnet18-5c106cde.pth"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done
if [ "$ALL_OK" = false ]; then
    echo "[ERROR] Some model files are missing. Use manual download commands above."
fi

# -------------------------------------------------------
# Step 8: Quick smoke test
# -------------------------------------------------------
echo ""
echo "=== Smoke test ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== Installation complete ==="
echo "To test inference:"
echo "  cd $INSTALL_DIR && source venv/bin/activate"
echo "  sh inference.sh v1.5 normal"
echo ""
echo "To start the API server (after creating musetalk_api.py):"
echo "  cd $INSTALL_DIR && source venv/bin/activate"
echo "  python musetalk_api.py --port $PORT"
```

---

## 3. FastAPI Wrapper Service

Save this as `$INSTALL_DIR/musetalk_api.py`:

```python
#!/usr/bin/env python3
"""
MuseTalk FastAPI wrapper for SKVoice integration.
Takes audio WAV + portrait image, returns lip-synced video.

Usage:
    python musetalk_api.py --port 18801 --use_float16

Endpoint:
    POST /lipsync
    - audio: WAV file (UploadFile)
    - portrait: PNG/JPG image (UploadFile)
    - bbox_shift: int (default 0, adjust if mouth alignment is off)
    Returns: MP4 video file

    GET /health
    Returns: {"status": "ok", "gpu": "...", "vram_free": "..."}
"""

import argparse
import io
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import imageio
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# Add MuseTalk to path
MUSETALK_DIR = Path(__file__).parent
sys.path.insert(0, str(MUSETALK_DIR))

# MuseTalk imports (loaded after sys.path modification)
from musetalk.utils.utils import load_all_model, get_image, datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image as blend_image

app = FastAPI(title="MuseTalk Lip-Sync API", version="1.0.0")

# Global model state
models = {}
device = None
use_fp16 = False


def load_models(fp16: bool = True):
    """Load all MuseTalk models into GPU memory."""
    global models, device, use_fp16
    use_fp16 = fp16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MuseTalk API] Loading models on {device} (fp16={fp16})...")
    t0 = time.time()

    # Load VAE, UNet, positional encoder
    vae, unet, pe = load_all_model(
        unet_model_path=str(MUSETALK_DIR / "models/musetalkV15/unet.pth"),
        vae_type="sd-vae",
        unet_config=str(MUSETALK_DIR / "models/musetalkV15/musetalk.json"),
        device=device,
    )

    if fp16:
        vae = vae.half()
        unet.model = unet.model.half()
        pe = pe.half()

    models["vae"] = vae
    models["unet"] = unet
    models["pe"] = pe

    # Audio processor (Whisper)
    from musetalk.utils.audio import AudioProcessor
    models["audio_processor"] = AudioProcessor(
        model_path=str(MUSETALK_DIR / "models/whisper")
    )

    elapsed = time.time() - t0
    print(f"[MuseTalk API] Models loaded in {elapsed:.1f}s")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[MuseTalk API] VRAM allocated: {allocated:.2f} GB")


@app.on_event("startup")
async def startup():
    load_models(fp16=use_fp16)


@app.get("/health")
async def health():
    info = {"status": "ok", "models_loaded": bool(models)}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = f"{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f}"
        info["vram_allocated_gb"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f}"
        info["vram_free_gb"] = f"{(torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()) / 1024**3:.2f}"
    return JSONResponse(info)


@app.post("/lipsync")
async def lipsync(
    audio: UploadFile = File(...),
    portrait: UploadFile = File(...),
    bbox_shift: int = Form(default=0),
):
    """
    Generate lip-synced video from audio + portrait image.

    Args:
        audio: WAV audio file
        portrait: PNG/JPG portrait image (face clearly visible)
        bbox_shift: Vertical shift for mouth bounding box (default 0)

    Returns:
        MP4 video file with lip-synced animation
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"musetalk_{job_id}_"))

    try:
        # Save uploaded files
        audio_path = work_dir / "input.wav"
        portrait_path = work_dir / "portrait.png"
        output_path = work_dir / "output.mp4"

        audio_data = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        portrait_data = await portrait.read()
        with open(portrait_path, "wb") as f:
            f.write(portrait_data)

        # Read portrait image
        img = cv2.imread(str(portrait_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid portrait image")

        # Extract audio features
        audio_proc = models["audio_processor"]
        whisper_feature, audio_length = audio_proc.get_audio_feature(str(audio_path))

        # Calculate number of frames at 25fps
        fps = 25
        num_frames = int(audio_length * fps)
        if num_frames < 1:
            raise HTTPException(status_code=400, detail="Audio too short")

        # Create frame list (repeat portrait for each frame)
        input_img_list = [str(portrait_path)] * num_frames

        # Get facial landmarks
        coord_list, frame_list = get_landmark_and_bbox(
            input_img_list, bbox_shift=bbox_shift
        )

        # Get whisper chunks aligned to frames
        whisper_chunks = audio_proc.get_whisper_chunk(
            whisper_feature, num_frames, fps
        )

        # Run inference
        vae = models["vae"]
        unet = models["unet"]
        pe = models["pe"]

        output_frames = []
        gen = datagen(
            whisper_chunks, vae, frame_list, coord_list,
            batch_size=8, device=device
        )

        for whisper_batch, latent_batch in gen:
            if use_fp16:
                whisper_batch = whisper_batch.half()
                latent_batch = latent_batch.half()

            audio_features = pe(whisper_batch)
            timesteps = torch.tensor([0], device=device)
            pred_latents = unet.model(
                latent_batch, timesteps,
                encoder_hidden_states=audio_features
            ).sample
            recon = vae.decode_latents(pred_latents)

            for i, res_frame in enumerate(recon):
                # Blend mouth region back onto original frame
                result = get_image(
                    img.copy(), res_frame,
                    coord_list[len(output_frames) + i]
                )
                output_frames.append(result)

        # Encode to video with audio
        frames_dir = work_dir / "frames"
        frames_dir.mkdir()
        for i, frame in enumerate(output_frames):
            cv2.imwrite(str(frames_dir / f"{i:06d}.png"), frame)

        # Use ffmpeg to combine frames + audio
        os.system(
            f'ffmpeg -y -r {fps} -i "{frames_dir}/%06d.png" '
            f'-i "{audio_path}" -c:v libx264 -pix_fmt yuv420p '
            f'-c:a aac -shortest "{output_path}" 2>/dev/null'
        )

        if not output_path.exists():
            raise HTTPException(status_code=500, detail="Video encoding failed")

        return FileResponse(
            str(output_path),
            media_type="video/mp4",
            filename=f"lipsync_{job_id}.mp4",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Note: temp dir cleanup should be handled by a background task
    # or periodic cron job. Not cleaned here so FileResponse can read it.


@app.post("/lipsync/frames")
async def lipsync_frames(
    audio: UploadFile = File(...),
    portrait: UploadFile = File(...),
    bbox_shift: int = Form(default=0),
    fps: int = Form(default=25),
):
    """
    Stream-friendly variant: returns raw frame data as a ZIP of PNGs.
    Better for real-time WebRTC integration where you want individual frames.
    """
    # Similar to /lipsync but returns frames as ZIP instead of encoded MP4
    # Implementation follows same pattern as above, skip video encoding step
    raise HTTPException(status_code=501, detail="Not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18801)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--use_float16", action="store_true", default=True)
    parser.add_argument("--no_float16", action="store_true")
    args = parser.parse_args()

    if args.no_float16:
        use_fp16 = False

    uvicorn.run(app, host=args.host, port=args.port)
```

---

## 4. Systemd Service

Save as `~/.config/systemd/user/musetalk-api.service`:

```ini
[Unit]
Description=MuseTalk Lip-Sync API
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/cbrd21/musetalk
ExecStart=/home/cbrd21/musetalk/venv/bin/python musetalk_api.py --port 18801 --use_float16
Restart=on-failure
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0
Environment=FFMPEG_PATH=/usr/bin/ffmpeg

[Install]
WantedBy=default.target
```

Enable with:
```bash
systemctl --user daemon-reload
systemctl --user enable --now musetalk-api.service
```

---

## 5. TensorRT Optimization

MuseTalk does NOT ship with built-in TensorRT support, but you can convert
the UNet and VAE to TensorRT engines for 2-4x inference speedup:

```bash
# Install TensorRT (in the musetalk venv)
pip install tensorrt torch-tensorrt

# Convert UNet to TensorRT (example script)
python -c "
import torch
import torch_tensorrt
from musetalk.models.unet import UNet

# Load the model
model = UNet(...)  # load with config
model.load_state_dict(torch.load('models/musetalkV15/unet.pth'))
model.eval().cuda().half()

# Trace and compile
example_latent = torch.randn(1, 8, 32, 32).half().cuda()
example_timestep = torch.tensor([0]).cuda()
example_audio = torch.randn(1, 50, 384).half().cuda()

trt_model = torch_tensorrt.compile(model,
    inputs=[
        torch_tensorrt.Input(example_latent.shape, dtype=torch.float16),
        torch_tensorrt.Input(example_timestep.shape, dtype=torch.int64),
        torch_tensorrt.Input(example_audio.shape, dtype=torch.float16),
    ],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30,
)
torch.jit.save(trt_model, 'models/musetalkV15/unet_trt.ts')
"
```

**Caveats**:
- TensorRT compilation takes 5-20 minutes but only needs to happen once
- The compiled engine is GPU-specific (RTX 5060 Ti engine won't work on other GPUs)
- Input shapes are fixed at compile time — batch size changes require recompilation
- RTX 5060 Ti (Blackwell) needs TensorRT 10.x+, check NVIDIA compatibility

**Simpler alternative** — `torch.compile`:
```python
# In the API, after model loading:
unet.model = torch.compile(unet.model, mode="reduce-overhead")
# First inference is slow (compiles), subsequent ones are 20-30% faster
```

---

## 6. LivePortrait Comparison

| Feature              | MuseTalk                          | LivePortrait                     |
|----------------------|-----------------------------------|----------------------------------|
| **Primary use**      | Audio-driven lip sync             | Motion retargeting (video-driven)|
| **Audio input**      | YES (core feature)                | NO (no audio support)            |
| **Speed**            | 30fps+ real-time                  | ~30fps with torch.compile        |
| **VRAM**             | ~4-5 GB (fp16)                    | ~4-6 GB                          |
| **Output quality**   | Good mouth sync, 256x256 face     | Higher quality motion transfer   |
| **Python**           | 3.10                              | 3.10                             |
| **ComfyUI plugin**   | Community nodes exist             | Official ComfyUI integration     |
| **Lip sync quality** | Purpose-built, good               | N/A — not designed for this      |

**Verdict**: LivePortrait is NOT a MuseTalk alternative for lip-sync.
LivePortrait does motion retargeting (copy head pose/expression from driving
video to portrait). It has zero audio-driven capability. For audio-to-lip-sync,
MuseTalk is the correct choice.

**Complementary use**: You could use LivePortrait for head motion/expression
and MuseTalk for mouth sync. Some community pipelines combine both, but this
adds complexity and VRAM.

**Other actual alternatives for audio lip-sync**:
- **Wav2Lip** — older, lower quality, but lighter (~2 GB VRAM)
- **SadTalker** — full head motion + lip sync, heavier (~6 GB)
- **LatentSync** (ByteDance) — newer diffusion-based, higher quality but slower
- **V-Express** — Tencent, audio + reference pose driven

---

## 7. Known Issues & Gotchas

1. **Python 3.12 compatibility**: MuseTalk pins `numpy==1.23.5` and
   `tensorflow==2.12.0` which may not install on Python 3.12. Use 3.10.

2. **RTX 5060 Ti (Blackwell)**: PyTorch 2.0.1 may not support this GPU.
   You may need PyTorch 2.5+ with CUDA 12.4. The install script has a fallback.

3. **mmcv build**: mmcv 2.0.1 compiles CUDA kernels and can take 10-30 minutes.
   Alternatively use `mim install mmcv==2.0.1` which downloads prebuilt wheels.

4. **First inference is slow**: Face detection, landmark extraction, and model
   warmup happen on first call. Subsequent calls reuse cached landmarks if the
   same portrait is used.

5. **Portrait requirements**: Face must be clearly visible, front-facing,
   good lighting. 512x512 or larger recommended. The model crops and resizes
   internally to 256x256.

6. **Temp file cleanup**: The API creates temp dirs for each request. Add a
   cron job or background task to clean `/tmp/musetalk_*` older than 1 hour.

---

## 8. Integration with SKVoice

In `skvoice`, after TTS generates audio, call the MuseTalk API:

```python
import httpx

async def generate_lipsync(audio_bytes: bytes, portrait_path: str) -> bytes:
    """Call MuseTalk API to generate lip-synced video."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "http://localhost:18801/lipsync",
            files={
                "audio": ("speech.wav", audio_bytes, "audio/wav"),
                "portrait": ("portrait.png", open(portrait_path, "rb"), "image/png"),
            },
            data={"bbox_shift": "0"},
        )
        resp.raise_for_status()
        return resp.content  # MP4 bytes
```

---

## Port Map (192.168.0.100)

| Port  | Service          | Status   |
|-------|------------------|----------|
| 18793 | Chatterbox TTS   | Running  |
| 18794 | faster-whisper   | Running  |
| 18800 | SKVoice          | Running  |
| 18801 | MuseTalk API     | PLANNED  |
| 11434 | Ollama           | Running  |
| 8188  | ComfyUI          | Running  |
