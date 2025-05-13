install:
	pip install -r requirements.txt

test:
	python -m pytest ai_trading/tests/ -v -rs 

format:
	autoflake --in-place --remove-all-unused-imports --recursive ai_trading/
	isort ai_trading/
	black ai_trading/

docker:
	docker compose -f docker/docker-compose.yml up --build

docker-test:
	# Construction de l'image Docker de test...
	docker build -t ai-trading-test -f docker/Dockerfile.test .

	# Execution des tests standards...
	docker run --rm --gpus all ai-trading-test

	# Pour executer tous les tests (incluant les tests lents), utilisez:
	docker run --rm --gpus all -e RUN_SLOW_TESTS=1 ai-trading-test

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} + 

tests: 
# Test RL PASSED OR FAILED en ligne
python -m pytest ai_trading/tests/ -v -rs 

# Optimisation check 
python -m ai_trading.optim.check_all_optimizations --check-all-opti

# Dashboard 
python -m ai_trading.dashboard.run

