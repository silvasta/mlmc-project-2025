from utils import load_device_print_information
from ultralytics import YOLO
from PathHandler import PathHandler


ph = PathHandler

device = load_device_print_information()

model_types = [
    "n",
    # "s",
    # "m",
    # "l",
    # "x",
]
for t in model_types:
    experiment = f"train_{t}_to_convergence_all"
    model = YOLO(f"{ph.exp_data}/{experiment}/{ph.best}")
    print("model loaded")

    imx_model = model.export(
        format="imx",
        imgsz=640,
        int8=True,
        data=ph.data,
        fraction=1,
        device=0,  # cpu
        name=f"imx_model/yolo_{t}",
    )

"""
imxconv-pt -i best_imx.onnx -o /home/silvan/mlmc/imx_models/yolo_n

imxconv-pt -i /home/silvan/mlmc/experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_model/best_imx.onnx -o /home/silvan/mlmc/imx_models/yolo_n_1 --no-input-persistency --overwrite-output
"""
