# filepath: /home/Michel850101/mlflow-deploy/Makefile

install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

train:
	python3 src/train.py

validate:
	python3 src/validate.py