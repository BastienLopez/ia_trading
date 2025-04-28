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

:: Configuration PyTorch
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8

:: Configuration TensorFlow
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_GPU_ALLOCATOR=cuda_malloc_async
set TF_CPP_MIN_LOG_LEVEL=3
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set XLA_FLAGS=--xla_gpu_cuda_data_dir="%CUDA_PATH%"
set TF_ENABLE_ONEDNN_OPTS=0

:end_cuda

:: Création du script Python temporaire dans un fichier
set TEMP_SCRIPT=%TEMP%\check_gpu_temp.py
set OUTPUT_FILE=%TEMP%\gpu_info.txt

echo import torch > "%TEMP_SCRIPT%"
echo import tensorflow as tf >> "%TEMP_SCRIPT%"
echo import sys >> "%TEMP_SCRIPT%"
echo import os >> "%TEMP_SCRIPT%"
echo import platform >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo print("") >> "%TEMP_SCRIPT%"
echo print("=== Informations Systeme ===") >> "%TEMP_SCRIPT%"
echo print("Python version:", sys.version) >> "%TEMP_SCRIPT%"
echo print("OS:", sys.platform) >> "%TEMP_SCRIPT%"
echo print("Architecture:", platform.architecture()[0]) >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo print("") >> "%TEMP_SCRIPT%"
echo print("=== Informations PyTorch ===") >> "%TEMP_SCRIPT%"
echo print("Version PyTorch:", torch.__version__) >> "%TEMP_SCRIPT%"
echo try: >> "%TEMP_SCRIPT%"
echo     print("Build CUDA:", torch.version.cuda) >> "%TEMP_SCRIPT%"
echo except: >> "%TEMP_SCRIPT%"
echo     print("Build CUDA: Non disponible") >> "%TEMP_SCRIPT%"
echo print("GPU disponible:", torch.cuda.is_available()) >> "%TEMP_SCRIPT%"
echo if torch.cuda.is_available(): >> "%TEMP_SCRIPT%"
echo     print("Nom GPU:", torch.cuda.get_device_name(0)) >> "%TEMP_SCRIPT%"
echo     print("Memoire GPU disponible:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB") >> "%TEMP_SCRIPT%"
echo. >> "%TEMP_SCRIPT%"
echo print("") >> "%TEMP_SCRIPT%"
echo print("=== Informations TensorFlow ===") >> "%TEMP_SCRIPT%"
echo print("Version TensorFlow:", tf.__version__) >> "%TEMP_SCRIPT%"
echo try: >> "%TEMP_SCRIPT%"
echo     print("Build CUDA:", tf.sysconfig.get_build_info()['cuda_version']) >> "%TEMP_SCRIPT%"
echo except: >> "%TEMP_SCRIPT%"
echo     print("Build CUDA: Non disponible") >> "%TEMP_SCRIPT%"
echo gpus = tf.config.list_physical_devices('GPU') >> "%TEMP_SCRIPT%"
echo print("GPU disponible:", len(gpus) ^> 0) >> "%TEMP_SCRIPT%"
echo if len(gpus) ^> 0: >> "%TEMP_SCRIPT%"
echo     print("Nom GPU:", gpus[0].name) >> "%TEMP_SCRIPT%"
echo     try: >> "%TEMP_SCRIPT%"
echo         print("Memoire GPU disponible:", tf.config.experimental.get_memory_info(gpus[0].name)['available'] / 1024**3, "GB") >> "%TEMP_SCRIPT%"
echo     except: >> "%TEMP_SCRIPT%"
echo         print("Impossible de recuperer la memoire GPU") >> "%TEMP_SCRIPT%"

:: Exécution du script et affichage des résultats
python "%TEMP_SCRIPT%" > "%OUTPUT_FILE%" 2>nul
type "%OUTPUT_FILE%"

:: Nettoyage des fichiers temporaires
del "%TEMP_SCRIPT%" 2>nul
del "%OUTPUT_FILE%" 2>nul

echo.
echo Pour executer les tests:
echo python -m pytest ai_trading/tests/
echo.
echo Pour lancer l'application web:
echo python web_app/app.py 