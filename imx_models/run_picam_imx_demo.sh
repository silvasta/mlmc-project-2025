#!/bin/bash

# choose the desired demo task
# DEMO_FILE=imx500_object_detection_demo.py
DEMO_FILE=imx500_object_detection_demo_mp.py

# set the model
MODEL=yolo_n_1/network.rpk

# choose frames or pass as argument
FRAMES=${1:-17} # default=17

# class labels for dataset
LABELS=labels.txt

# additional options
declare -a OPTS
OPTS=(
  --bbox-normalization
)

python $DEMO_FILE --model $MODEL --fps "$FRAMES" --labels $LABELS "${OPTS[@]}"

# echo $DEMO_FILE --model $MODEL --fps "$FRAMES" --labels $LABELS "${OPTS[@]}"
