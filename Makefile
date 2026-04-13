.PHONY: all setup train run clean

VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python3
PIP=${VENV_NAME}/bin/pip

all: setup train run

setup:
	@echo "Setting up virtual environment and installing dependencies..."
	python3 -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete."

train:
	@echo "Training the model..."
	$(PYTHON) train.py --epochs 5 --batch-size 64
	@echo "Training complete."

run:
	@echo "Processing images in input_images directory..."
	$(PYTHON) main.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	rm -rf data
	rm -f mnist_ffn.safetensors
	rm -f training_log.csv
	rm -f loss_curve.png
	rm -f time_log.txt
	rm -f output_labels.txt
	@echo "Clean complete."
