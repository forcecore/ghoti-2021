#!/bin/bash

# Download pretrained weights if exists.
# Comment out this part and the mention of this in Dockerfile, if the link gets out-of-date.
# The neural network knows how to download the pretrained weights.
weights=resnet50-0676ba61.pth
if [ ! -e $weights ] ; then
    curl "https://download.pytorch.org/models/$weights" > $weights
fi

docker build -t ghoti .
