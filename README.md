# MNIST Handwritten Digit Classifier (TinyGrad + GPU)

## Code Project Description

**Purpose:** This project builds an automated Python batch processor for classifying handwritten decimal digits (0-9). The goal is to provide a complete computational pipeline: defining a Feedforward Neural Network, training it on the MNIST dataset using GPU acceleration via TinyGrad, and running fully batched image processing that scans directories, normalizes local images, and yields structured text labeling.

**Algorithms/Kernels:** The fundamental algorithm applied is a multi-layer perceptron (Feedforward Neural Network). Under the hood, **TinyGrad** dynamically compiles JIT kernels (CUDA/OpenCL/Metal depending on the hardware available) for remarkably fast array operations while having an installation footprint of under 2MB. Raw image preprocessing acts by using `Pillow` to transform unconstrained image files (.png/.jpg) down strictly to exactly 28x28 grayscale matrices and standardizes tensor deviation prior to GPU evaluation. 

**Lessons Learned:** 
1. **Handling Heterogenous Data Types:** A key challenge was standardizing files fetched blindly from a folder (`input_images`) effectively transforming high-resolution RGB arbitrary photography consistently into properly constrained arrays mimicking MNIST features natively. 
2. **Execution Portability:** By leveraging TinyGrad, the model trains seamlessly whether executed on a massive GPU cluster or a basic laptop, automatically locating appropriate compilation backends without heavy 500MB dependencies.
3. **Execution Analytics:** Training times have been strictly tracked in a persistent log so performance scale measurements can be extracted.

## How to Compile and Run

Support files (`Makefile` and `run.sh`) have been provided. Please follow the instructions below from your terminal:

**1. Setup Environment**  
This command securely builds a Python virtual environment and installs necessary tiny dependencies via `pip`.
```bash
./run.sh setup
# OR: make setup
```

**2. Train the Model**  
Once the environment is active, you can train your local model. You can specify different parameters via CLI arguments such as `--epochs` and `--batch-size`.
```bash
# General training via Makefile wrapper
./run.sh train

# OR run python directly for more CLI arguments
./venv/bin/python3 train.py --epochs 5 --batch-size 64 --lr 0.001
```
*Note: Doing this automatically outputs total execution time into `time_log.txt`.*

**3. Run the Batch Processor**  
Perform classification across an entire dataset blindly. Place your pictures in the `input_images` directory.
```bash
./run.sh run
# OR: make run
```

## Proof of Execution
When you run the pipeline, it outputs:
1. `time_log.txt` explicitly containing cluster training wall-time.
2. `training_log.csv` & `loss_curve.png` testing logic graphs.
3. `output_labels.txt` holding generated labels against raw incoming filenames.
