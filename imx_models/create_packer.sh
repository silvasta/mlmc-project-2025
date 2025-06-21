#!/bin/bash

### transforms output from model compress to camera format

# setup environment
source /home/silvan/mlmc/.venv/bin/activate

# input
project_home=/home/silvan/mlmc
# relative=experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_model/best_imx.onnx
# relative=experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_pqt/best_imx.onnx
relative=experiments/african-wildlife/train_s_to_convergence/weights/best_imx_model/best_imx.onnx
IMX_IN_FILE=$project_home/$relative

# output
IMX_FOLDERNAME=${1:-"model"}
IMX_OUT_NAME="$IMX_FOLDERNAME-$(date +%F)-$(date +%R)"
# IMX_OUT_NAME=${1:-"model_$(date +%F)-$(date +%R)"}
IMX_OUT_DIR="$project_home/imx_models"
IMX_OUT_PATH="$IMX_OUT_DIR/$IMX_OUT_NAME"

# additional options
declare -a OPTS
OPTS=(
  # --no-input-persistency
  # --overwrite-output
  --memory-report
)
# finallly, do the conversion
imxconv-pt -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"

echo -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"
