import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from unetRGB import *

# Step 1: Load the trained model
myunet = myUnet()
model = myunet.get_unet()
model.load_weights('unet.keras')

# Step 2: Load and preprocess the image
# Example: Load an image from file
image_path = 'data/test/image/2.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Use IMREAD_COLOR if it's a color image

# Resize the image to match the input size expected by your U-Net (e.g., 256x256)
img_resized = cv2.resize(img, (512, 512))

# Normalize the image (same preprocessing as used during training)
img_normalized = img_resized / 255.0  # If you normalized the training data

# Expand the dimensions to match the model input shape (batch_size, height, width, channels)
img_input = np.expand_dims(img_normalized, axis=3)  # Add channel dimension
img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

# Step 3: Run inference
prediction = model.predict(img_input)

# Step 4: Postprocess and visualize the result
# For binary segmentation (thresholding)
prediction = prediction.squeeze()  # Remove batch and channel dimensions if it's single-channel output
prediction = (prediction > 0.5).astype(np.uint8)  # Apply threshold to get binary mask

# Visualize the result
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_resized, cmap='gray')
plt.axis('off')

# Prediction (Segmented Output)
plt.subplot(1, 2, 2)
plt.title("Model Prediction")
plt.imshow(prediction, cmap='gray')
plt.axis('off')

plt.show()
