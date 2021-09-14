#!/bin/bash

GHOTI=`pwd`
DATASET_DIR=$GHOTI/mydataset
COCO_STYLE_OUTPUT=$DATASET_DIR/cropped_coco_style
TFRECORDS_OUTPUT=$DATASET_DIR/cropped_tfrecords

CAMERTRAPS_DIR=$GHOTI/CameraTraps
MEGADETECTOR_PB="${CAMERTRAPS_DIR}/pbs/md_v4.1.0.pb"

cd $CAMERTRAPS_DIR/data_management/databases/classification

python cropped_camera_trap_dataset_statistics.py \
    $GHOTI/data/ghoti.json \
    $COCO_STYLE_OUTPUT/train.json \
    $COCO_STYLE_OUTPUT/test.json \
    --classlist_output $COCO_STYLE_OUTPUT/classlist.txt
