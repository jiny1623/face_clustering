import numpy as np

with open('./data/labels/part1_test.meta', 'r') as file:
    labels = [int(line.strip()) for line in file]

# Convert the labels to a NumPy array and save it
labels_array = np.array(labels)
print(labels_array.shape)
# print(labels_array)
np.save('./data/labels/part1_test.npy', labels_array)