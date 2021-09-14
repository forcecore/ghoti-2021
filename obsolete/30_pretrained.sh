#!/bin/bash

PRETRAINED_DIR=`pwd`/pretrained
mkdir -p $PRETRAINED_DIR && cd $PRETRAINED_DIR
curl "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz" > inc4.tar.gz
tar xzf inc4.tar.gz
