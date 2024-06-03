import subprocess
import CNN_model
import CNN_updatable

# Import the trained model and the "new model" from files


# Run CNN_Model.py in the container
subprocess.run(["python3", "CNN_model.py"])
print("CNN runned on the standard model")

# Run CNN_Updatable.py in the container
subprocess.run(["python3", "CNN_updatable.py"])
print("CNN trained on the new images")

# Run CNN_Model.py in the container
subprocess.run(["python3", "CNN_model.py"])
print("CNN runned on the updated model")
