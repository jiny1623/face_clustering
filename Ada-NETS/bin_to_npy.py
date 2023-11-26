import numpy as np
import os
import sys

def process_and_save_file(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    feature_path = './data/feature/' + file_name + '.npy'
    label_path = './data/label/' + file_name + '.npy'

    # Read binary file
    arr = np.fromfile(file_path, dtype=np.float32)
    
    # Reshape
    arr = arr.reshape((-1, 256))
    
    # L2 Normalization
    arr_norm = arr / np.linalg.norm(arr, axis=1, keepdims=True)

    # Save as .npy
    np.save(feature_path, arr_norm)
    print(f"Processed feature saved at: {feature_path}")

    # make dummy label
    zero_array = np.zeros(arr.shape[0])
    
    np.save(label_path, zero_array)
    print(f"Processed label saved at: {label_path}")



if __name__ == "__main__":
    # Check if a file path is provided as a command line argument
    if len(sys.argv) != 2:
        print("Usage: python bin_to_npy.py <file_path>")
    else:
        file_path = sys.argv[1]
        if not os.path.isfile(file_path):
            print(f"Error: File not found at {file_path}")
        else:
            process_and_save_file(file_path)