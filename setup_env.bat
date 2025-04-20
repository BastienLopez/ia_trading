@echo off
REM Configuration de l'environnement pour le projet crypto-trading

REM Activation de l'environnement virtuel (si vous en utilisez un)
REM call venv\Scripts\activate

REM Configuration des variables d'environnement CUDA pour PyTorch et TensorFlow
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
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
echo GPU PyTorch: Activé (NVIDIA GeForce RTX 3070)
echo TensorFlow: Mode GPU activé

REM Vérification si PyTorch détecte le GPU
python -c "import torch; print(f'PyTorch détecte GPU: {torch.cuda.is_available()}')"

REM Vérification si TensorFlow détecte le GPU
python -c "import tensorflow as tf; print(f'TensorFlow détecte GPU: {len(tf.config.list_physical_devices(\"GPU\"))>0}')"

REM Information sur l'exécution du projet
echo.
echo Pour lancer le projet, exécutez maintenant: python main.py
echo. 