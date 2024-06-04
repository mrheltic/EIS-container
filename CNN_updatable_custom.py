import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

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

class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'squirrel', 'dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'mucca', 'spider', 'squirrel']

test_images_dir = './Train_Images'

train_images_dir = './Raw-Images'
validation_images_dir = './Validation_Images'

images = np.empty((0, 32, 32, 3))
labels = np.empty((0, 1))

for filename in os.listdir(train_images_dir):
    # Load the image
    image = plt.imread(os.path.join(train_images_dir, filename))
    # Resize the image to 32x32
    image = tf.image.resize(image, (32, 32))

    
    # Remove the alpha channel if it exists
    if image.shape[2] == 3:
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
    


train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


"""for filename in os.listdir(train_images_dir):
    # Load the image
    image = plt.imread(os.path.join(train_images_dir, filename))
    
    # Extract the label from the filename
    label = filename.split(' ')[0]

    # Se il nome della classe non è presente nella lista, la aggiunge
    if label not in class_names:
        class_names.append(label)

    templabel = []
    templabel.append(class_names.index(label))
    
    # Resize the image to 32x32
    image = tf.image.resize(image, (32, 32))
    
    # Add the image and label to the train dataset
    train_images = np.append(train_images, [image], axis=0)
    train_labels = np.append(train_labels, [templabel], axis=0)

for filename in os.listdir(validation_images_dir):
    # Load the image
    image = plt.imread(os.path.join(validation_images_dir, filename))
    
    # Extract the label from the filename
    label = filename.split(' ')[0]

    # Se il nome della classe non è presente nella lista, la aggiunge
    if label not in class_names:
        # Se non presente lancia un errore
        raise ValueError(f'Class name {label} not found in class_names list')

    templabel = []
    templabel.append(class_names.index(label))
    
    # Resize the image to 32x32
    image = tf.image.resize(image, (32, 32))
    
    # Add the image and label to the train dataset
    test_images = np.append(test_images, [image], axis=0)
    test_labels = np.append(test_labels, [templabel], axis=0)"""
    
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Print the number of images in the training and test datasets
print(f'Number of training images: {train_images.shape[0]}')
print(f'Number of test images: {test_images.shape[0]}')


"""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()"""

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = class_names.__len__()
no_epochs = 10
optimizer = Adam()
validation_split = 0.2
verbosity = 1

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


# Compile the model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit data to model
history = model.fit(train_images, train_labels,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

"""# Controlla se il modello esiste già, se si carica le le informazioni di loss e accuracy
if os.path.exists('./history.txt'):
    with open('./history.txt', 'r') as f:
        lines = f.readlines()
        old_test_loss, old_test_acc = float(lines[0].split(':')[1].split(',')[0]), float(lines[0].split(':')[2])
        print(f'Loaded test loss: {old_test_loss}, test accuracy: {old_test_acc}')
        # Controlla se old_test_loss e old_test_acc sono numeri validi o diversi da 0
        if old_test_loss == 0 or old_test_acc == 0:
            old_test_loss, old_test_acc = 100, 1
            print('Invalid values found in history file, starting from scratch')
else :
    old_test_loss, old_test_acc = 100, 0
    print('No history file found, starting from scratch')

# Salva il modello se non ce'già un modello (a prescindere) oppure se il modello attuale ha una loss function migliore (2,5%) rispetto a quello precedente
if not os.path.exists('./CNN_model.keras') or (test_loss < 99.75*old_test_loss and test_acc > 1.0025*old_test_acc):
    model.save('./CNN_model.keras')
    print(f'Model saved to CNN_model.keras')
    print(f'Loss improvement: {((old_test_loss - test_loss) / old_test_loss) * 100:.3f}%, Accuracy improvement: {((test_acc - old_test_acc) / old_test_acc) * 100:.3f}%')
"""

model.save('./CNN_model.keras')
print(f'Model saved to CNN_model.keras')

# Salva su file la history del modello, con i valori di loss e accuracy
with open('./history.txt', 'w') as f:
    f.write(f'Test loss: {test_loss}, Test accuracy: {test_acc}\n')
    f.write('Epoch\tLoss\tAccuracy\tVal_loss\tVal_accuracy\n')
    for i, hist in enumerate(history.history['loss']):
        f.write(f'{i+1}\t{hist}\t{history.history["accuracy"][i]}\t{history.history["val_loss"][i]}\t{history.history["val_accuracy"][i]}\n')

# Control if the file already exists
if os.path.exists('./class_names.txt'):
    # Delete the file if it exists
    os.remove('./class_names.txt')

# Save the class names to a file
with open('./class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f'{name}\n')

# Close the file
f.close()

print("CNN updated successfully!")