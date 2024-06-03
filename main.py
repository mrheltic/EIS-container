import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt

# Run CNN_Model.py in the container
subprocess.run(["python3", "CNN_model.py"])

# Run CNN_Updatable.py in the container
subprocess.run(["python3", "CNN_updatable.py"])

# Run CNN_Model.py in the container
subprocess.run(["python3", "CNN_model.py"])

class_labels = []
class_names_file = './class_names.txt'
if os.path.exists(class_names_file):
    with open(class_names_file, 'r') as file:
        for line in file:
            class_labels.append(line.strip())

# Open the two files
f1 = open("./results1.txt", "r")
f2 = open("./results2.txt", "r")

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
plt.savefig('./confusion_matrix.png')

# Create a file to compare the results only about the predicted class and the real label
f = open("./comparison.txt", "w")

# Open the two files
f1 = open("./results1.txt", "r")
f2 = open("./results2.txt", "r")

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
    print('Model improved and saved')
else:
    print('Model not improved')
    # Delete the file if it exists
    os.remove('./CNN_model.keras')

print("Finished execution!")

# Close the files
f1.close()
f2.close()
