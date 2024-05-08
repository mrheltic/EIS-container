#!/bin/bash
docker run --gpus all -d -p 8888:8888 --name eis-container eis:latest