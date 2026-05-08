.PHONY: install train backtest api dashboard test docker clean

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

backtest:
	python scripts/backtest.py --ticker RELIANCE.NS

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest tests/ -v

docker:
	docker compose up --build

clean:
	rm -rf __pycache__ .pytest_cache results/* models_saved/*.pkl models_saved/*.pt
