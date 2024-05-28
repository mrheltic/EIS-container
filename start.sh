#!/bin/bash
docker run --gpus all -d -p 192.168.1.135:8888:8888 --name eis-container eis-container

echo "EIS container is running."
echo "You can access it at http://192.168.1.135:8888"
