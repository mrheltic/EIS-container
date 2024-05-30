import os
from keras.models import load_model
from keras.preprocessing import image
from keras.datasets import cifar10
from PIL import Image
import socket

# Path to the folder containing the test images
test_images_folder = 'Container_Script/Test_Images'

# Load the Keras model
model = load_model('CNN_model.keras')

# Load the CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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

    print(f'Image: {filename}, Predicted Class: {predicted_label}')


