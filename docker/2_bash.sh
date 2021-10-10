#!/bin/bash

# --privileged makes nvidia devs to work in the docker
# --ipc=host : Fixes the following error message : ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)
docker run \
    --gpus all \
    --privileged \
    --ipc=host \
    -v `pwd`/..:/workspace/ghoti-2021 \
    -it ghoti \
    /bin/bash
