#!/bin/bash

# Stop the container
docker stop eis-container || docker rm eis-container

# Print a nicer output
echo "Container stopped successfully."
