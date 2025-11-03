#!/bin/bash
# Setup script for Socipedia backend

echo "🚀 Socipedia Backend Setup"
echo "=========================="
echo ""

# Detect if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected!"
    echo "📋 GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    echo "🔧 Setting up with GPU support..."
    echo ""
    
    # Install CPU requirements first
    echo "📦 Installing base requirements..."
    pip install -r requirements-cpu.txt
    
    # Uninstall CPU PyTorch
    echo "🗑️  Removing CPU-only PyTorch..."
    pip uninstall torch torchvision torchaudio -y
    
    # Install GPU PyTorch
    echo "⚡ Installing GPU-enabled PyTorch (this may take a while - ~2.8GB download)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Verify GPU
    echo ""
    echo "🔍 Verifying GPU support..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    
else
    echo "⚠️  No NVIDIA GPU detected"
    echo "🔧 Setting up with CPU-only support..."
    echo ""
    
    # Install CPU requirements
    echo "📦 Installing CPU requirements..."
    pip install -r requirements-cpu.txt
    
    echo ""
    echo "ℹ️  Note: Image moderation will work but may be slower without GPU"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  python manage.py runserver"
