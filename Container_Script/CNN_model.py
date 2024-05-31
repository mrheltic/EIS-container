import os
from keras.models import load_model
from keras.preprocessing import image
from keras.datasets import cifar100
from PIL import Image
import socket
import numpy as np
import matplotlib.pyplot as plt

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


# Open a file to write the results

# Control if exists a file with name "results2.txt" in the folder
if os.path.exists("Container_Script/results2.txt"):
    # If the file exists remove both results1.txt and results2.txt
    os.remove("Container_Script/results2.txt")

    # Control if exists a file with name "results1.txt" in the folder
    if os.path.exists("Container_Script/results1.txt"):
        os.remove("Container_Script/results1.txt")
        f = open("Container_Script/results1.txt", "w")
    else:
        # Throw an error if the file doesn't exist
        raise ValueError("File results1.txt not found")

    # Open a file with name "results1.txt" in the folder
    f = open("Container_Script/results1.txt", "w")
    path="Container_Script/results1.txt"
else:
    # Open a file with name "results2.txt" in the folder
    f = open("Container_Script/results2.txt", "w")
    path="Container_Script/results2.txt"

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

    # Write the results to the file
    f.write(f"File name: {filename}, Image: {filename.split(' ')[0]}, Predicted Class: {predicted_label}\n")

# Close the file
f.close()

# If path is results2.txt do a confrontation between results1.txt and results2.txt, creating a confusion matrix knowing the real label (Image) and the predicted label (Predicted Class)

if path == "Container_Script/results2.txt":
    # Open the two files
    f1 = open("Container_Script/results1.txt", "r")
    f2 = open("Container_Script/results2.txt", "r")

    # Create a confusion matrix
    confusion_matrix = [[0 for _ in range(len(class_labels))] for _ in range(len(class_labels))]

    # Iterate over the lines of the two files
    for line1, line2 in zip(f1, f2):
        # Extract the predicted class and real label from the lines
        predicted_class = line1.split(",")[2].strip().split(":")[1].strip()
        real_label = line2.split(",")[2].strip().split(":")[1].strip()

        # Update the confusion matrix
        predicted_index = class_labels.index(predicted_class)
        real_index = class_labels.index(real_label)
        confusion_matrix[real_index][predicted_index] += 1

    # Close the files
    f1.close()
    f2.close()

    # Convert the confusion matrix to a numpy array
    confusion_matrix = np.array(confusion_matrix)

    # Plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()

    # Set the tick labels
    plt.xticks(np.arange(len(class_labels)), class_labels, rotation=90)
    plt.yticks(np.arange(len(class_labels)), class_labels)

    # Set the axis labels
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Label')

    # Set the title
    plt.title('Confusion Matrix')

    # Save the plot to a file
    plt.savefig('Container_Script/confusion_matrix.png')

    # Create a file to compare the results only about the predicted class and the real label
    f = open("Container_Script/comparison.txt", "w")

    # Open the two files
    f1 = open("Container_Script/results1.txt", "r")
    f2 = open("Container_Script/results2.txt", "r")

    # Initialize the variables to count the number of right values and the number of same values
    right_values = 0
    same_values = 0

    # Iterate over the lines of the two files
    for line1, line2 in zip(f1, f2):
        # Extract the predicted class and real label from the lines
        result1 = line1.split(",")[2].strip().split(":")[1].strip()
        result2 = line2.split(",")[2].strip().split(":")[1].strip()
        real_label = line1.split(",")[1].strip().split(":")[1].strip()

        # If the two results are the same write the comparison to the file
        if result1 == result2:
            f.write(f"Both run prediction: {result1}, Real Label: {real_label}\n")
            same_values += 1
        else:
            f.write(f"First run: {result1}, Second run: {result2}, Real Label: {real_label}\n")
            # If the results2 is right, increment the right_values
            if result2 == real_label:
                right_values += 1 #

    # If the model is improved, based on the right values, save the model
    if right_values > 0 and right_values > same_values:
        model.save('Container_Script/CNN_model.keras')
        print('Model improved and saved')
    else:
        print('Model not improved')
        # Delete the file if it exists
        os.remove('Container_Script/CNN_model.keras')


    # Close the files
    f1.close()
    f2.close()



