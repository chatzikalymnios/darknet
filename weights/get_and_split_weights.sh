#!/bin/sh

# Get YOLOv3 weights
wget https://pjreddie.com/media/files/yolov3.weights

# Split Jetson weights
dd if=yolov3.weights ibs=1 skip=0 count=1120660 of=yolov3-jetson.weights

# Split Server weights
dd if=yolov3.weights ibs=1 skip=0 count=20 of=preamble.weights
dd if=yolov3.weights ibs=1 skip=1120660 of=rest.weights
cat preamble.weights rest.weights > yolov3-server.weights

# Cleanup
rm preamble.weights
rm rest.weights
