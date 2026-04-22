setup:
	pip install -e .

data:
	python -c "from src.data import generate_data, load_config; generate_data(load_config())"

train:
	python -c "import joblib; from src.data import generate_data, load_config; from src.model import train_model; cfg=load_config(); train_model(cfg, *generate_data(cfg)[0])"

test:
	PYTHONPATH=. pytest tests/

clean:
	rm -rf data/* outputs/figures/*
