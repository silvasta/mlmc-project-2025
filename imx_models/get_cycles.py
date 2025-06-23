import sys
import time

import numpy as np
from model_compression_toolkit import get_model_info
# Replace with your model’s framework (e.g., PyTorch, Keras)

models = [
    "n_640/network.rpk",
    "n_to_convergence_all-2025-06-23_03-34-40/network.rpk",
    "n_to_convergence_all-2025-06-23_03-38-14/network.rpk",
    "s_to_convergence-2025-06-23_03-41-47/network.rpk",
    "s_to_convergence_all-2025-06-23_03-49-35/network.rpk",
    "yolo_n_1/network.rpk",
]


def get_cycles(model):
    model_info = get_model_info(model)
    print(f"MACs: {model_info.macs}, FLOPs: {model_info.flops}")


if __name__ == "__main__":
    # This must be called before instantiation of Picamera2
    results = {}
    for model in models:
        imx500 = IMX500(model)
        # intrinsics = imx500.network_intrinsics
        # if not intrinsics:
        #     intrinsics = NetworkIntrinsics()
        #     intrinsics.task = "object detection"
        # elif intrinsics.task != "object detection":
        #     print("Network is not an object detection task", file=sys.stderr)
        #     exit()
        get_cycles(model)
