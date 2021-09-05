#!/bin/bash

GHOTI=`pwd`
DATASET_DIR=$GHOTI/mydataset
COCO_STYLE_OUTPUT=$DATASET_DIR/cropped_coco_style
TFRECORDS_OUTPUT=$DATASET_DIR/cropped_tfrecords

CAMERTRAPS_DIR=$GHOTI/CameraTraps
MEGADETECTOR_PB="${CAMERTRAPS_DIR}/pbs/md_v4.1.0.pb"

cd $CAMERTRAPS_DIR/data_management/databases/classification

python make_classification_dataset.py \
    $GHOTI/data/ghoti.json \
    $GHOTI/data/ \
    $MEGADETECTOR_PB \
    --coco_style_output $COCO_STYLE_OUTPUT \
    --tfrecords_output $TFRECORDS_OUTPUT \
    --location_key location
