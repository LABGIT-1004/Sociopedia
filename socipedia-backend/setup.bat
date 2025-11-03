@echo off
REM Setup script for Socipedia backend (Windows)

echo ========================================
echo    Socipedia Backend Setup (Windows)
echo ========================================
echo.

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
    echo.
    echo GPU Information:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
    echo Setting up with GPU support...
    echo.
    
    REM Install CPU requirements first
    echo Installing base requirements...
    python -m pip install -r requirements-cpu.txt
    
    REM Uninstall CPU PyTorch
    echo.
    echo Removing CPU-only PyTorch...
    python -m pip uninstall torch torchvision torchaudio -y
    
    REM Install GPU PyTorch
    echo.
    echo Installing GPU-enabled PyTorch (2.8GB download - please wait)...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    REM Verify GPU
    echo.
    echo Verifying GPU support...
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'; print('GPU Device:', gpu_name)"
    
) else (
    echo No NVIDIA GPU detected
    echo Setting up with CPU-only support...
    echo.
    
    REM Install CPU requirements
    echo Installing CPU requirements...
    python -m pip install -r requirements-cpu.txt
    
    echo.
    echo Note: Image moderation will work but may be slower without GPU
)

echo.
echo Setup complete!
echo.
echo To start the server, run:
echo   python manage.py runserver
echo.
pause
