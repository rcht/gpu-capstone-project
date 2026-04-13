import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import gzip

from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict, safe_save
import tinygrad.nn.optim as optim
from model import MNISTClassifier

def fetch_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    os.makedirs('data', exist_ok=True)
    def parse(file):
        filepath = os.path.join('data', file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
        with gzip.open(filepath, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16 if 'images' in file else 8)

    X_train = parse("train-images-idx3-ubyte.gz").reshape(-1, 28*28)
    Y_train = parse("train-labels-idx1-ubyte.gz")
    
    # Normalize
    X_train = (X_train.astype(np.float32) / 255.0 - 0.1307) / 0.3081
    return X_train, Y_train

def train(epochs, batch_size, lr):
    print("Preparing MNIST data...")
    X_train_np, Y_train_np = fetch_mnist()
    
    model = MNISTClassifier()
    optimizer = optim.Adam([t for t in get_state_dict(model).values() if isinstance(t, Tensor)], lr=lr)

    log_data = []
    print("Starting training on TinyGrad...")
    start_time = time.time()
    
    dataset_size = X_train_np.shape[0]
    batches = dataset_size // batch_size
    
    for epoch in range(epochs):
        Tensor.training = True
        running_loss = 0.0
        
        indices = np.random.permutation(dataset_size)
        X_shuf = X_train_np[indices]
        Y_shuf = Y_train_np[indices]
        
        for i in range(batches):
            x = Tensor(X_shuf[i * batch_size : (i + 1) * batch_size])
            y = Tensor(Y_shuf[i * batch_size : (i + 1) * batch_size])
            
            out = model(x)
            
            loss = out.sparse_categorical_crossentropy(y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 200 == 199:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{batches}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / batches
        log_data.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"--- Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} ---")
        
    end_time = time.time()
    train_duration = end_time - start_time
    print(f"Training completed in {train_duration:.2f} seconds.")
    
    # Save training time
    with open("time_log.txt", "w") as f:
        f.write(f"Total training time: {train_duration:.2f} seconds\n")
        f.write(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}\n")
    print("Saved time_log.txt")
    
    # Save the model
    model_path = "mnist_ffn.safetensors"
    state_dict = get_state_dict(model)
    safe_save(state_dict, model_path)
    print(f"Model saved to {model_path}")
    
    # Save execution proof artifacts
    df = pd.DataFrame(log_data)
    df.to_csv("training_log.csv", index=False)
    print("Saved training_log.csv")
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(df['epoch'], df['loss'], marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch (TinyGrad)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Saved loss_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST Feedforward Neural Network using TinyGrad")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
