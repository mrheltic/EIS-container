
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import cv2
import subprocess


"""(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
               'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
               'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
               'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
               'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
               'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
               'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
               'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
               'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']"""

class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'squirrel', 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'spider', 'squirrel', 'Parrot', 'frog']

test_images_dir = './Train_Images'

train_images_dir = './Train_Images'
validation_images_dir = './Validation_Images'

images = np.empty((0, 32, 32, 3))
labels = np.empty((0, 1))

for filename in os.listdir(train_images_dir):
    # Load the image
    image = cv2.imread(os.path.join(train_images_dir, filename))
    
    # Resize the image to 32x32
    image = cv2.resize(image, (32, 32))
    
    # Remove the alpha channel if it exists
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Extract the label from the filename
    label = filename.split(' ')[0]

    # Se il nome della classe non è presente nella lista, la aggiunge
    if label not in class_names:
        class_names.append(label)

    templabel = []
    templabel.append(class_names.index(label))
    
    # Add the image and label to the train dataset
    images = np.append(images, [image], axis=0)
    labels = np.append(labels, [templabel], axis=0)
else:
    print(f'Image {filename} hasn\'t 3 channels, skipping...')

test_images = np.empty((0, 32, 32, 3))
test_labels = np.empty((0, 1))

test_images_folder = './Test-Parrot'
# Iterate over all the images in the folder
for filename in os.listdir(test_images_folder):
    # Load and preprocess the image
    # Load and preprocess the image
    image = cv2.imread(os.path.join(test_images_folder, filename))
    image = cv2.resize(image, (32, 32))
    if image.shape[2] == 4:
        image = image[:, :, :3]
    img = image / 255.0

    templabel = []
    templabel.append(class_names.index(label))
    # Add the image and label to the test dataset
    test_images = np.append(test_images, [img], axis=0)
    test_labels = np.append(test_labels, [templabel], axis=0)


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Print the number of images in the training and test datasets
print(f'Number of training images: {train_images.shape[0]}')
print(f'Number of test images: {test_images.shape[0]}')

# Model configuration
batch_size = 30
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = class_names.__len__()
no_epochs = 20
optimizer = Adam()
validation_split = 0.2
verbosity = 1


def create_model():
    global model
    # Create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(no_classes, activation='softmax'))
    
    return model


counter = 0

for filename in os.listdir(validation_images_dir):
    # Load the image
    image = cv2.imread(os.path.join(validation_images_dir, filename))
    
    # Resize the image to 32x32
    image = cv2.resize(image, (32, 32))
    
    # Remove the alpha channel if it exists
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Extract the label from the filename
    label = filename.split(' ')[0]

    # Se il nome della classe non è presente nella lista, la aggiunge
    if label not in class_names:
        class_names.append(label)

    templabel = []
    templabel.append(class_names.index(label))
    
    # Add the image and label to the validation dataset
    images = np.append(images, [image], axis=0)
    labels = np.append(labels, [templabel], axis=0)

    counter += 1
    acc = []
    loss = []
    
    if counter % 20 == 0:
        # Normalize pixel values to be between 0 and 1


        images = images / 255.0
        # Compile the model
        create_model()
        optimizer = Adam()
        model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

        history = model.fit(train_images, train_labels,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=0,
            validation_split=validation_split)

        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

        #Do the prediction on the test images
        predictions = model.predict(test_images)
        predicted_classes = [class_names[i] for i in np.argmax(predictions, axis=1)]

        # Save the accuracy and the loss in the array
        acc.append(test_acc)
        loss.append(test_loss)

        # Eliminate the model
        del model

        counter = 0
else:
    print(f'Image {filename} hasn\'t 3 channels, skipping...')
