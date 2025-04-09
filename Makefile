install:
	pip install -r requirements.txt

test:
	python -m pytest ai_trading/tests/ -v

lint:
	isort --check-only ai_trading/
	black --check ai_trading/

format:
	autoflake --in-place --remove-all-unused-imports --recursive ai_trading/
	isort ai_trading/
	black ai_trading/

docker-test:
	docker-compose build && docker-compose up tests

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} + 