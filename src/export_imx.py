from utils import load_device_print_information
from ultralytics import YOLO
from PathHandler import PathHandler

# --- DEBUG
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# --- DEBUG

ph = PathHandler

device = load_device_print_information()

model_types = [
    "n",
    # "s",
    # "m",
    # # "l",
    # "x",
]
for t in model_types:
    experiment = f"train_{t}_to_convergence_all"
    # experiment = f"train_{t}_to_convergence"
    print()
    print(experiment)
    print()
    model = YOLO(f"{ph.exp_data}/{experiment}/{ph.best}")
    print("model loaded")
    print()

    imx_model = model.export(
        format="imx",
        imgsz=640,
        int8=True,
        data=ph.data,
        fraction=1,
        device=device,
        name=f"imx_model/yolo_{t}_2",
    )

    #
# /home/silvan/PolyBox/Master/Machine Learning on MicroController FS25/project

# def pin_memory(data, device=None):
#     if isinstance(data, torch.Tensor):
#         #TODO: remove print, device?
#         # print("device was..... :",device)
#         device="cuda"
#         return data.pin_memory(device)
