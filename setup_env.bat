@echo off
setlocal enabledelayedexpansion

echo Configuration de l'environnement pour le trading crypto...

:: Détection automatique de CUDA
where cuda >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%a in ('where cuda') do (
        set CUDA_PATH=%%~dpa
        goto :cuda_found
    )
)

:: Vérification des chemins CUDA par défaut
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
    goto :cuda_found
)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    goto :cuda_found
)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7" (
    set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.7
    goto :cuda_found
)

echo CUDA non trouvé. Installation de la version CPU...
goto :end_cuda

:cuda_found
echo CUDA trouve dans: %CUDA_PATH%

:: Configuration des variables d'environnement CUDA
set PATH=%CUDA_PATH%\bin;%PATH%
set PATH=%CUDA_PATH%\libnvvp;%PATH%
set CUDA_VISIBLE_DEVICES=0

:: Configuration PyTorch optimisée pour RTX 3070
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PYTORCH_CUDA_ALLOC_CONF=roundup_power2:True
set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync

:: Optimisation des performances PyTorch
set PYTORCH_JIT=1
set PYTORCH_CUDA_ARCH_LIST=8.6
set PYTORCH_CUDA_FUSER=1
set PYTORCH_TENSOREXPR=1
set PYTORCH_MPS=0
set PYTORCH_NUM_WORKERS=4
set PYTORCH_PIN_MEMORY=1
set PYTORCH_BENCHMARK=1
set PYTORCH_DETERMINISTIC=0

:: Optimisations avancées PyTorch
set PYTORCH_CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_USE_BLAS=1
set PYTORCH_CUDA_USE_CUDNN=1
set PYTORCH_CUDA_USE_CUBLAS=1
set PYTORCH_CUDA_USE_CUSPARSE=1
set PYTORCH_CUDA_USE_CUSOLVER=1
set PYTORCH_CUDA_USE_CURAND=1
set PYTORCH_CUDA_USE_CUFFT=1
set PYTORCH_CUDA_USE_CUDNN_BATCHNORM=1
set PYTORCH_CUDA_USE_CUDNN_CONV=1
set PYTORCH_CUDA_USE_CUDNN_RNN=1
set PYTORCH_CUDA_USE_CUDNN_LSTM=1
set PYTORCH_CUDA_USE_CUDNN_GRU=1
set PYTORCH_CUDA_USE_CUDNN_DROPOUT=1
set PYTORCH_CUDA_USE_CUDNN_POOLING=1
set PYTORCH_CUDA_USE_CUDNN_ACTIVATION=1
set PYTORCH_CUDA_USE_CUDNN_SOFTMAX=1
set PYTORCH_CUDA_USE_CUDNN_BATCHNORM_SPATIAL=1
set PYTORCH_CUDA_USE_CUDNN_BATCHNORM_PER_ACTIVATION=1
set PYTORCH_CUDA_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
set PYTORCH_CUDA_USE_CUDNN_BATCHNORM_PER_ACTIVATION_PERSISTENT=1

:: Optimisation de la mémoire et du cache
set PYTORCH_CUDA_CACHE_PATH=%TEMP%\torch_cache
set PYTORCH_CUDA_CACHE_SIZE=1024
set PYTORCH_CUDA_CACHE_CLEANUP=1
set PYTORCH_CUDA_CACHE_PREFETCH=1
set PYTORCH_CUDA_CACHE_PREFETCH_SIZE=256
set PYTORCH_CUDA_CACHE_PREFETCH_THRESHOLD=0.8

:: Optimisation des threads et du parallélisme
set PYTORCH_NUM_THREADS=8
set PYTORCH_NUM_INTEROP_THREADS=4
set PYTORCH_NUM_IO_THREADS=4
set PYTORCH_NUM_WORKERS=4
set PYTORCH_NUM_LOADER_WORKERS=4
set PYTORCH_NUM_DATALOADER_WORKERS=4
set PYTORCH_NUM_DATASET_WORKERS=4
set PYTORCH_NUM_CUDA_STREAMS=4
set PYTORCH_NUM_CUDA_GRAPH_STREAMS=4
set PYTORCH_NUM_CUDA_GRAPH_POOLS=4
set PYTORCH_NUM_CUDA_GRAPH_CACHES=4

:: Optimisation des performances de calcul
set PYTORCH_CUDA_USE_TENSOR_CORES=1
set PYTORCH_CUDA_USE_MIXED_PRECISION=1
set PYTORCH_CUDA_USE_FP16=1
set PYTORCH_CUDA_USE_BF16=1
set PYTORCH_CUDA_USE_TF32=1
set PYTORCH_CUDA_USE_AMP=1
set PYTORCH_CUDA_USE_GRADIENT_CHECKPOINTING=1
set PYTORCH_CUDA_USE_GRADIENT_ACCUMULATION=1
set PYTORCH_CUDA_USE_GRADIENT_CLIPPING=1
set PYTORCH_CUDA_USE_GRADIENT_SCALING=1
set PYTORCH_CUDA_USE_GRADIENT_NORM=1
set PYTORCH_CUDA_USE_GRADIENT_MEAN=1
set PYTORCH_CUDA_USE_GRADIENT_STD=1
set PYTORCH_CUDA_USE_GRADIENT_MIN=1
set PYTORCH_CUDA_USE_GRADIENT_MAX=1
set PYTORCH_CUDA_USE_GRADIENT_SUM=1
set PYTORCH_CUDA_USE_GRADIENT_PROD=1
set PYTORCH_CUDA_USE_GRADIENT_DOT=1
set PYTORCH_CUDA_USE_GRADIENT_CROSS=1
set PYTORCH_CUDA_USE_GRADIENT_DIV=1
set PYTORCH_CUDA_USE_GRADIENT_POW=1
set PYTORCH_CUDA_USE_GRADIENT_SQRT=1
set PYTORCH_CUDA_USE_GRADIENT_EXP=1
set PYTORCH_CUDA_USE_GRADIENT_LOG=1
set PYTORCH_CUDA_USE_GRADIENT_SIN=1
set PYTORCH_CUDA_USE_GRADIENT_COS=1
set PYTORCH_CUDA_USE_GRADIENT_TAN=1
set PYTORCH_CUDA_USE_GRADIENT_ASIN=1
set PYTORCH_CUDA_USE_GRADIENT_ACOS=1
set PYTORCH_CUDA_USE_GRADIENT_ATAN=1
set PYTORCH_CUDA_USE_GRADIENT_SINH=1
set PYTORCH_CUDA_USE_GRADIENT_COSH=1
set PYTORCH_CUDA_USE_GRADIENT_TANH=1
set PYTORCH_CUDA_USE_GRADIENT_ASINH=1
set PYTORCH_CUDA_USE_GRADIENT_ACOSH=1
set PYTORCH_CUDA_USE_GRADIENT_ATANH=1

:: Configuration TensorFlow (gardée pour compatibilité)
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set TF_CPP_MIN_LOG_LEVEL=3
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set XLA_FLAGS=--xla_gpu_cuda_data_dir="%CUDA_PATH%"
set TF_ENABLE_ONEDNN_OPTS=0
set TF_XLA_FLAGS=--tf_xla_enable_xla_devices
set TF_GPU_THREAD_MODE=gpu_private
set TF_GPU_THREAD_COUNT=2
set TF_USE_CUDA=1
set TF_CUDA_HOME=%CUDA_PATH%
set TF_CUDA_VERSION=12.8
set TF_CUDNN_VERSION=8.9
set TF_CUDA_COMPUTE_CAPABILITIES=8.6

:end_cuda

:: Création du script Python temporaire
echo import torch > check_gpu.py
echo import sys >> check_gpu.py
echo import platform >> check_gpu.py
echo import psutil >> check_gpu.py
echo. >> check_gpu.py
echo def main(): >> check_gpu.py
echo     # Informations système >> check_gpu.py
echo     print("\n=== Informations Systeme ===") >> check_gpu.py
echo     print(f"Python version: {sys.version}") >> check_gpu.py
echo     print(f"OS: {sys.platform}") >> check_gpu.py
echo     print(f"Architecture: {platform.architecture()[0]}") >> check_gpu.py
echo     print(f"Processeur: {platform.processor()}") >> check_gpu.py
echo     print(f"Nombre de coeurs: {psutil.cpu_count(logical=False)}") >> check_gpu.py
echo     print(f"Nombre de threads: {psutil.cpu_count(logical=True)}") >> check_gpu.py
echo     try: >> check_gpu.py
echo         print(f"Frequence CPU: {psutil.cpu_freq().current} MHz") >> check_gpu.py
echo     except: >> check_gpu.py
echo         print("Frequence CPU: Non disponible") >> check_gpu.py
echo     print(f"Memoire RAM totale: {round(psutil.virtual_memory().total / (1024**3), 2)} GB") >> check_gpu.py
echo     print(f"Memoire RAM disponible: {round(psutil.virtual_memory().available / (1024**3), 2)} GB") >> check_gpu.py
echo. >> check_gpu.py
echo     # Informations PyTorch >> check_gpu.py
echo     print("\n=== Informations PyTorch ===") >> check_gpu.py
echo     print("Version PyTorch:", torch.__version__) >> check_gpu.py
echo     try: >> check_gpu.py
echo         print("Build CUDA:", torch.version.cuda) >> check_gpu.py
echo     except: >> check_gpu.py
echo         print("Build CUDA: Non disponible") >> check_gpu.py
echo     print("GPU disponible:", torch.cuda.is_available()) >> check_gpu.py
echo     if torch.cuda.is_available(): >> check_gpu.py
echo         print("Nom GPU:", torch.cuda.get_device_name(0)) >> check_gpu.py
echo         print("Memoire GPU disponible:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB") >> check_gpu.py
echo     print("") >> check_gpu.py
echo. >> check_gpu.py
echo if __name__ == "__main__": >> check_gpu.py
echo     main() >> check_gpu.py

:: Exécution du script Python
python check_gpu.py

:: Nettoyage
del check_gpu.py

echo.
echo Pour executer les tests:
echo python -m pytest ai_trading/tests/
echo.
echo Pour lancer l'application web:
echo python web_app/app.py 