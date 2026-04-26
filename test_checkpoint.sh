#!/bin/bash
# Quick script to test a checkpoint
# Usage: ./test_checkpoint.sh <config> <checkpoint> [--disable-wandb]
#
# Example:
#   ./test_checkpoint.sh configs/camely_abmil.py runs/camely-abmil-001/checkpoints/best.ckpt
#   ./test_checkpoint.sh configs/camely_abmil.py runs/camely-abmil-001/checkpoints/best.ckpt --disable-wandb

if [ $# -lt 2 ]; then
    echo "Usage: $0 <config> <checkpoint> [--disable-wandb]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/camely_abmil.py runs/camely-abmil-001/checkpoints/best.ckpt"
    echo "  $0 configs/camely_abmil.py runs/camely-abmil-001/checkpoints/best.ckpt --disable-wandb"
    exit 1
fi

python -m wsi_classification.test_checkpoint --config "$1" --checkpoint "$2" "${@:3}"
