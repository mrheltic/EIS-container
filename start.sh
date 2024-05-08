#!/bin/bash
docker run --gpus all -d -p 8888:8888 --name eis-container eis-container

echo "EIS container is running."
echo "You can access it at http://localhost:8888"