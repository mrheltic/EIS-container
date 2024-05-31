import os
from keras.models import load_model
from keras.preprocessing import image
from keras.datasets import cifar100
from PIL import Image
import socket

# Path to the folder containing the test images
test_images_folder = 'Container_Script/Test_Images'

# Load the Keras model
model = load_model('Container_Script/CNN_model.keras')

# Load the CIFAR-100 class labels
standard_class_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
               'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
               'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
               'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
               'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
               'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
               'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
               'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
               'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# load the new class_labels from class_names.txt
class_labels = []
with open('Container_Script/class_names.txt', 'r') as file:
    for line in file:
        class_labels.append(line.strip())

# Print the new class labels without the standard CIFAR-100 labels
print('New Class Labels:')
for label in class_labels:
    if label not in standard_class_labels:
        print(label)


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


