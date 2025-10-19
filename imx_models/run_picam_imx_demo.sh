#!/bin/bash

# choose the desired demo task

# DEMO_FILE=imx500_object_detection_demo.py
DEMO_FILE=imx500_object_detection_demo_mp.py

# set the model
# MODEL=yolo_n_1/network.rpk
# ### working
# MODEL=n_to_convergence_all-2025-06-23_03-34-40/network.rpk
# MODEL=n_640/network.rpk # bad
# ### testing
# MODEL=n_to_convergence_all-2025-06-23_03-38-14/network.rpk
# ### problem
MODEL=s_to_convergence_all-2025-06-23_03-49-35/network.rpk
# MODEL=s_to_convergence-2025-06-23_03-41-47/network.rpk

# choose frames or pass as argument
FRAMES=${1:-10} # default=17

# class labels for dataset
LABELS=labels.txt

# additional options
declare -a OPTS
OPTS=(
  --bbox-normalization
)

python $DEMO_FILE --model $MODEL --fps "$FRAMES" --labels $LABELS "${OPTS[@]}"

# echo $DEMO_FILE --model $MODEL --fps "$FRAMES" --labels $LABELS "${OPTS[@]}"
