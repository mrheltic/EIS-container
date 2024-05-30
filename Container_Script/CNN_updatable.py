import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

test_images_dir = 'Container_Script/Train_Images'

for filename in os.listdir(test_images_dir):
    # Load the image
    image = plt.imread(os.path.join(test_images_dir, filename))
    
    # Extract the label from the filename
    label = filename.split(' ')[0]
    templabel = []
    templabel.append(class_names.index(label))
    
    
    # Resize the image to 32x32
    image = tf.image.resize(image, (32, 32))
    
    # Add the image and label to the dataset
    train_images = np.append(train_images, [image], axis=0)
    train_labels = np.append(train_labels, [templabel], axis=0)
    templabel = []

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

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

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# Controlla se il modello esiste già, se si carica le le informazioni di loss e accuracy
if os.path.exists('history.txt'):
    with open('history.txt', 'r') as f:
        lines = f.readlines()
        old_test_loss, old_test_acc = float(lines[0].split(':')[1].split(',')[0]), float(lines[0].split(':')[2])
        print(f'Loaded test loss: {old_test_loss}, test accuracy: {old_test_acc}')
else :
    old_test_loss, old_test_acc = 100, 0
    print('No history file found, starting from scratch')

# Salva il modello se non ce'già un modello (a prescindere) oppure se il modello attuale ha una loss function migliore (2,5%) rispetto a quello precedente
if not os.path.exists('CNN_model.keras') or (test_loss < 99.75*old_test_loss and test_acc > 1.0025*old_test_acc):
    model.save('Container_Script/CNN_model.keras')
    print(f'Model saved to CNN_model.keras')
    print(f'Loss improvement: {((old_test_loss - test_loss) / old_test_loss) * 100:.3f}%, Accuracy improvement: {((test_acc - old_test_acc) / old_test_acc) * 100:.3f}%')

# Salva su file la history del modello, con i valori di loss e accuracy
with open('history.txt', 'w') as f:
    f.write(f'Test loss: {test_loss}, Test accuracy: {test_acc}\n')
    f.write('Epoch\tLoss\tAccuracy\tVal_loss\tVal_accuracy\n')
    for i, hist in enumerate(history.history['loss']):
        f.write(f'{i+1}\t{hist}\t{history.history["accuracy"][i]}\t{history.history["val_loss"][i]}\t{history.history["val_accuracy"][i]}\n')