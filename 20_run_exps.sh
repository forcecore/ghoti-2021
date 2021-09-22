#!/bin/bash

root_dir=`pwd`/data
COMMON_PARAMS="-S dataset.root_dir=$root_dir"

echo $COMMON_PARAMS
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=43
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=44
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=45
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=46
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=47
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=48
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=49
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=50
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=51
dvc exp run --queue $COMMON_PARAMS -S dataset.split_seed=52

dvc exp run --run-all --jobs 4
