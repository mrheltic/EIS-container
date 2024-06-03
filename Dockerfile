# Use the official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /.

# Expose a port for incoming connections
EXPOSE 80

# Print the current directory
RUN pwd

# Install the dependencies
RUN pip install keras matplotlib numpy

# Copy the script files
COPY ./CNN_model.py .
COPY ./CNN_updatable.py .
COPY ./main.py .
COPY ./Train_Images Train_Images
COPY ./Test_Images Test_Images
COPY ./Validation_Images Validation_Images
COPY ./Trained_Model.keras .
COPY ./results1.txt .
COPY ./results2.txt .

# Set the entrypoint command
CMD ["python3", "main.py"]
