from utils import load_device_print_information
from ultralytics import YOLO
from PathHandler import PathHandler

# --- DEBUG
# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# --- DEBUG

ph = PathHandler

device = load_device_print_information()

model_types = [
    # "train_n_to_convergence_all",
    # "train_n_to_convergence",
    # "train_s_to_convergence_all",
    # "train_s_to_convergence",
    "train_l_to_convergence_all",
    # "n",
    # "s",
    # "m",
    # "l",
    # "x",
]
for name in model_types:
    try:
        model_path = f"{ph.exp_data}/{name}/{ph.best}"
        model = YOLO(model_path)
        print()
        print(f"loaded {model_path}")
        print()

        imx_model = model.export(
            format="imx",
            imgsz=512,
            int8=True,
            data=ph.data,
            fraction=1,
            device=device,
            name=f"imx_model/{name}_end",
        )
    except:
        print()
        print(f"not worked for {name}")
        print()


# def pin_memory(data, device=None):
#     if isinstance(data, torch.Tensor):
#         #TODO: remove print, device? in file exporter.py (ultralytics)
#         # print("device was..... :",device)
#         device="cuda"
#         return data.pin_memory(device)
