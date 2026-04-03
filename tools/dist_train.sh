#!/usr/bin/env bash
# Distributed training launcher.
#
# Usage:
#   ./tools/dist_train.sh <CONFIG> <NUM_GPUS> [--cfg-options key=value ...]
#
# Example:
#   ./tools/dist_train.sh configs/denoiser/segformer_b2_512x512_40k_denoise.py 4

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch \
    ${@:3}
