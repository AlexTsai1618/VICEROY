import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split




(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Define the size of the segments and the amount of overlap
segment_size = 28  
overlap_size = 14  

# number of segments that can be extracted from each image
num_segments = (X_train_full.shape[2] - segment_size) // (segment_size - overlap_size) + 1

# Extract the segments 
X_train_segments = np.zeros((X_train_full.shape[0] * num_segments, segment_size, segment_size))
y_train_segments = np.zeros((X_train_full.shape[0] * num_segments,))
for i in range(X_train_full.shape[0]):
    for j in range(num_segments):
        start = j * (segment_size - overlap_size)
        end = start + segment_size
        X_train_segments[i*num_segments+j] = X_train_full[i, :, start:end]
        y_train_segments[i*num_segments+j] = y_train_full[i]

# Split the segmented dataset into training and validation sets
X_train_segments, X_val_segments, y_train_segments, y_val_segments = train_test_split(
    X_train_segments, y_train_segments, test_size=0.2, random_state=42)
    
#print the training and validation segments 

print(X_train_segments.shape)
print(X_val_segments.shape)
print(y_train_segments.shape)
print(y_val_segments.shape)

