import os
import glob
from PIL import Image
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_load, load_state_dict
from model import MNISTClassifier

def load_model():
    model = MNISTClassifier()
    state_dict = safe_load("mnist_ffn.safetensors")
    load_state_dict(model, state_dict)
    return model

def main():
    input_dir = "input_images"
    output_file = "output_labels.txt"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory '{input_dir}'. Please place images inside it and run again.")
        return

    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    # Eliminate potential duplicates due to case-insensitivity extensions
    image_files = list(set(image_files))

    if not image_files:
        print(f"No images found in '{input_dir}'. Please place images inside and run again.")
        return

    print("Loading model...")
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model (have you run train.py?): {e}")
        return

    print(f"Found {len(image_files)} images. Running inference...")
    
    with open(output_file, 'w') as f:
        for img_path in sorted(image_files):
            try:
                # Open image using Pillow
                img = Image.open(img_path)
                
                # Convert to grayscale ("L")
                img = img.convert("L")
                
                # Resize to 28x28 exactly as MNIST demands
                img_resized = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to raw numpy array float 32
                img_np = np.array(img_resized).astype(np.float32)
                
                # Normalize pixels according to MNIST standard preprocessing
                img_np = (img_np / 255.0 - 0.1307) / 0.3081
                
                # Bind arrays to TinyGrad tensor framework and ensure we are in inference mode
                Tensor.training = False
                img_tensor = Tensor(img_np.reshape(1, 28 * 28))
                
                # Execute inference feed-forward (compiled into GPU kernels)
                output = model(img_tensor)
                
                # Identify maximum probability tensor structure
                pred_idx = output.numpy().argmax(axis=1)[0]
                
                filename = os.path.basename(img_path)
                
                # Format into output batch sequence
                f.write(f"{filename},{pred_idx}\n")
                print(f"Processed: {filename} -> Output label [{pred_idx}]")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    print(f"Inference completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
