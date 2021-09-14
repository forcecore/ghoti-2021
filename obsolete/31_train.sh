#!/bin/bash

export GHOTI=`pwd`
# cd CameraTraps/classification
export PYTHONPATH=`pwd`/CameraTraps:$PYTHONPATH

# python train_classifier.py run_ghoti $GHOTI/mydataset/cropped_coco_style
python -m classification.train_classifier run_ghoti $GHOTI/mydataset/cropped_coco_style \
    -m "resnet50" --pretrained --finetune=0 --label-weighted \
    --epochs=50 --batch-size=512 --lr=3e-4 \
    --num-workers=12 --seed=1234 \
    --logdir=$GHOTI/run_ghoti
