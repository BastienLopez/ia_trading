@echo off
REM Configuration de l'environnement pour le projet crypto-trading

REM Activation de l'environnement virtuel (si vous en utilisez un)
REM call venv\Scripts\activate

REM Détection automatique du chemin CUDA
for /f "tokens=*" %%a in ('where cuda') do (
    set CUDA_PATH=%%~dpa
    goto :found_cuda
)
:found_cuda

REM Si CUDA n'est pas trouvé, essayer les chemins courants
if not defined CUDA_PATH (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7
    ) else (
        echo CUDA non trouvé. Vérifiez l'installation de CUDA.
        exit /b 1
    )
)

REM Configuration des variables d'environnement CUDA
set PATH=%CUDA_PATH%\bin;%PATH%
set CUDA_VISIBLE_DEVICES=0

REM Variables pour optimiser PyTorch sur GPU
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM Configuration pour TensorFlow avec GPU
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set TF_CPP_MIN_LOG_LEVEL=2
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set XLA_FLAGS=--xla_gpu_cuda_data_dir="%CUDA_PATH%"

echo Configuration de l'environnement terminée !
echo Chemin CUDA détecté: %CUDA_PATH%

REM Création d'un fichier temporaire pour le script Python
echo import torch > check_gpu.py
echo import tensorflow as tf >> check_gpu.py
echo import subprocess >> check_gpu.py
echo. >> check_gpu.py
echo print("=== Informations GPU ===") >> check_gpu.py
echo try: >> check_gpu.py
echo     gpu_info = subprocess.check_output('nvidia-smi --query-gpu=gpu_name --format=csv,noheader', shell=True).decode().strip() >> check_gpu.py
echo     print("Nom GPU:", gpu_info) >> check_gpu.py
echo except: >> check_gpu.py
echo     print("Impossible de récupérer les informations GPU") >> check_gpu.py
echo. >> check_gpu.py
echo print("\n=== Informations PyTorch ===") >> check_gpu.py
echo print("Version PyTorch:", torch.__version__) >> check_gpu.py
echo print("GPU disponible:", torch.cuda.is_available()) >> check_gpu.py
echo if torch.cuda.is_available(): >> check_gpu.py
echo     print("Nom GPU:", torch.cuda.get_device_name(0)) >> check_gpu.py
echo. >> check_gpu.py
echo print("\n=== Informations TensorFlow ===") >> check_gpu.py
echo print("Version TensorFlow:", tf.__version__) >> check_gpu.py
echo gpus = tf.config.list_physical_devices('GPU') >> check_gpu.py
echo print("GPU disponible:", len(gpus) ^> 0) >> check_gpu.py
echo if len(gpus) ^> 0: >> check_gpu.py
echo     print("Nom GPU:", gpus[0].name) >> check_gpu.py

REM Exécution du script Python
python check_gpu.py

REM Nettoyage du fichier temporaire
del check_gpu.py

REM Information sur l'exécution du projet
echo.
echo Pour lancer les tests, exécutez : python -m pytest ai_trading/tests/ -v
echo. 