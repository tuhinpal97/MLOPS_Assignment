setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	python scripts/download_data.py

eda:
	python -m src.eda

train:
	python -m src.train

test:
	pytest -q

serve:
	uvicorn src.api.main:app --reload --port 8000
