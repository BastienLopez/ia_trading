@echo off
echo Construction de l'image Docker de test...
docker build -t ai-trading-test -f Dockerfile.test .

echo.
echo Execution des tests standards...
docker run --rm --gpus all ai-trading-test

echo.
echo Pour executer tous les tests (incluant les tests lents), utilisez:
echo docker run --rm --gpus all -e RUN_SLOW_TESTS=1 ai-trading-test