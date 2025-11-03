# GPU Setup Instructions

## For Systems WITH NVIDIA GPU

### Prerequisites

- NVIDIA GPU (GTX/RTX series)
- Windows/Linux/Mac

### Installation Steps

1. **Install base requirements** (CPU versions):

   ```bash
   pip install -r requirements-cpu.txt
   ```

2. **Uninstall CPU-only PyTorch**:

   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```

3. **Install GPU-enabled PyTorch** (CUDA 11.8):

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

   **Note**: This downloads ~2.8GB, so be patient!

4. **Verify GPU is detected**:

   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

   You should see:

   ```
   CUDA available: True
   GPU: NVIDIA GeForce GTX/RTX XXXX
   ```

### Expected Performance

- **EasyOCR**: GPU accelerated ⚡ (much faster)
- **TensorFlow**: CPU (still fast)
- **First image**: ~10-15 seconds (model loading)
- **Subsequent images**: ~3-5 seconds

---

## For Systems WITHOUT GPU (CPU-only)

### Installation Steps

1. **Install CPU requirements**:
   ```bash
   pip install -r requirements-cpu.txt
   ```

That's it! The code automatically detects no GPU and uses CPU.

### Expected Performance

- **EasyOCR**: CPU mode (slower but works)
- **TensorFlow**: CPU mode
- **First image**: ~15-20 seconds (model loading)
- **Subsequent images**: ~5-10 seconds

---

## Troubleshooting

### "CUDA available: False" on GPU system

- Install CUDA Toolkit 11.8 from NVIDIA
- Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu118`

### "Out of memory" errors

- Your GPU VRAM is full
- Close other GPU-intensive applications
- Restart the server

### Slow performance on GPU

- Check if GPU is actually being used:
  ```bash
  nvidia-smi
  ```
- Look for `python.exe` in the process list

---

## Files Overview

- `requirements-cpu.txt` - For CPU-only systems (your friends)
- `req_with_gpu.txt` - Same as requirements-cpu.txt (PyTorch CPU versions)
- This file - GPU setup instructions

**Important**: You cannot install GPU PyTorch from a requirements file!
You must use the special PyTorch index URL.
