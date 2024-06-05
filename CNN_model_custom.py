import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar100
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

import matplotlib.pyplot as plt

# Path to the folder containing the test images
test_images_folder = './Test_Images'
print(os.listdir('./Test_Images'))

model = load_model('./CNN_model.keras')

# load the new class_labels from class_names.txt
class_labels = []
class_names_file = './class_names.txt'
if os.path.exists(class_names_file):
    with open(class_names_file, 'r') as file:
        for line in file:
            class_labels.append(line.strip())
else:
    raise ValueError("File class_names.txt not found")

# Initialize arrays to store results and real labels
results = []
real_labels = []

# Iterate over all the images in the folder
for filename in os.listdir(test_images_folder):
    # Load and preprocess the image
    img_path = os.path.join(test_images_folder, filename)
    img = image.load_img(img_path, target_size=(32, 32))
    img = image.img_to_array(img)
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)

    # Make predictions
    predictions = model.predict(img)
    predicted_class = predictions.argmax(axis=-1)
    predicted_label = class_labels[predicted_class[0]]

    print(f"Image: {filename.split(' ')[0]}, Predicted Class: {predicted_label}")

    # Save results and real labels
    results.append(predicted_label)
    real_labels.append(filename.split(' ')[0])

# Calculate and print accuracy score
accuracy = accuracy_score(real_labels, results)
print(f"Accuracy Score: {accuracy}")
