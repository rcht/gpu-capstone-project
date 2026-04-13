import os
import urllib.request
import gzip
import numpy as np
from PIL import Image

def fetch_mnist_test():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    os.makedirs('data', exist_ok=True)
    def parse(file):
        filepath = os.path.join('data', file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, filepath)
        with gzip.open(filepath, 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16 if 'images' in file else 8)

    X_test = parse("t10k-images-idx3-ubyte.gz").reshape(-1, 28, 28)
    Y_test = parse("t10k-labels-idx1-ubyte.gz")
    return X_test, Y_test

def main():
    X_test, Y_test = fetch_mnist_test()
    
    # Randomly sample 20 images
    np.random.seed(42) # For reproducibility
    indices = np.random.choice(len(X_test), 20, replace=False)
    
    out_dir = "input_images"
    os.makedirs(out_dir, exist_ok=True)
    
    # Clear old ones like dummy.png
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
        
    for i, idx in enumerate(indices):
        img_array = X_test[idx]
        label = Y_test[idx]
        
        img = Image.fromarray(img_array)
        img.save(os.path.join(out_dir, f"sample_{i:02d}_true_{label}.png"))
        
    print(f"Generated 20 random MNIST test images in '{out_dir}/'")

if __name__ == "__main__":
    main()
