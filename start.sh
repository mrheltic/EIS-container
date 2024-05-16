#!/bin/bash
docker run --gpus all -d -p 192.168.208.211:8888:8888 --name eis-container eis-container

echo "EIS container is running."
echo "You can access it at http://192.168.208.201:8888"
