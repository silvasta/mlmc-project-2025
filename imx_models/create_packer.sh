#!/bin/bash

### transforms output from model compress to camera format

# setup environment
source /home/silvan/mlmc/.venv/bin/activate

# input
project_home=/home/silvan/mlmc
model_1=n_to_convergence_all
model_2=n_to_convergence_all
model_3=s_to_convergence
model_4=s_to_convergence_all

PRE=experiments/african-wildlife/train_
SUC=weights/best_imx_model/best_imx.onnx

# additional options
declare -a OPTS
OPTS=(
  --no-input-persistency
  --overwrite-output
  --memory-report
)
array=("$model_1" "$model_2" "$model_3" "$model_4")

for relative in "${array[@]}"; do

  IMX_IN_FILE=$project_home/$PRE$relative/$SUC
  # echo "$IMX_IN_FILE"

  # output
  # IMX_FOLDERNAME=${1:-"model"}
  IMX_FOLDERNAME=$"$relative"
  IMX_OUT_NAME="$IMX_FOLDERNAME-$(date +%F)-$(date +%R)"
  # IMX_OUT_NAME=${1:-"model_$(date +%F)-$(date +%R)"}
  IMX_OUT_DIR="$project_home/imx_models"
  IMX_OUT_PATH="$IMX_OUT_DIR/$IMX_OUT_NAME"

  # finallly, do the conversion
  imxconv-pt -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"

  # echo -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"
done

IMX_IN_FILE=experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_pqt/best_imx.onnx

# output
IMX_FOLDERNAME=best_imx_pqt
IMX_OUT_NAME="$IMX_FOLDERNAME-$(date +%F)-$(date +%R)"
IMX_OUT_DIR="$project_home/imx_models"
IMX_OUT_PATH="$IMX_OUT_DIR/$IMX_OUT_NAME"

# finallly, do the conversion
imxconv-pt -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"

# echo -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" "${OPTS[@]}"
