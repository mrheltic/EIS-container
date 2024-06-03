import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
from sklearn.model_selection import GridSearchCV
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()
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
               'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

test_images_dir = './Train_Images'

train_images_dir = './Train_Images'
validation_images_dir = './Validation_Images'

# Set the number of CPU cores to use
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
os.environ["OMP_NUM_THREADS"] = "5"
os.environ["MKL_NUM_THREADS"] = "5"
os.environ["OPENBLAS_NUM_THREADS"] = "5"

for filename in os.listdir(train_images_dir):
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
    test_labels = np.append(test_labels, [templabel], axis=0)
    
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Print the number of images in the training and test datasets
print(f'Number of training images: {train_images.shape[0]}')
print(f'Number of test images: {test_images.shape[0]}')

# Define the parameter grid
param_grid = {
    'batch_size': [32, 64, 128],
    'no_epochs': [10, 20, 30],
    'dropout_rate': [0.3, 0.5, 0.7]
}

verbosity = 1
loss_function = sparse_categorical_crossentropy

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = class_names.__len__()
no_epochs = 10
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Define the model


def createModel():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(no_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer=optimizer,
                loss=loss_function,
                metrics=['accuracy'])

    return model



# Crea il modello
model = KerasClassifier(build_fn=createModel(), epochs=no_epochs, batch_size=batch_size, verbose=0)

# Definisci i parametri della grid
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)

# Crea la Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3, cv=3)

PYDEVD_DISABLE_FILE_VALIDATION=1
grid_result = grid.fit(train_images, train_labels)

# Set the number of CPU cores to use
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

best_model = grid.best_estimator_

best_params = grid.best_params_

# Print the best parameters
print("Best Parameters:", best_params)




# Compile the best model
best_model.compile(loss=loss_function,
                   optimizer=best_params['optimizer'],
                   metrics=['accuracy'])


# Fit data to the best model
history = best_model.fit(train_images, train_labels,
                         batch_size=best_params['batch_size'],
                         epochs=best_params['no_epochs'],
                         verbose=verbosity,
                         validation_split=validation_split)

# Evaluate the best model on the test data
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)

# Save the best model
best_model.save('./best_model.keras')


model.save('./CNN_model.keras')
