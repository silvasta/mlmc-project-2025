from dataclasses import dataclass
from datetime import datetime


@dataclass
class PathHandler:
    dataset = "african-wildlife"
    exp_data = "experiments/african-wildlife"
    data = f"{dataset}.yaml"
    experiment_name: str = ""
    experiment_path = f"experiment/{dataset}/{experiment_name}"

    best_path = f"{experiment_path}/weights/best.pt"
    last_path = f"{experiment_path}/weights/results.pt"
    best = "weights/best.pt"
    last = "weights/results.pt"

    time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    imx_characteristic = ""
    # imx_name = f"imx_{imx_characteristic}_{time}"
    # imx_pre = f"{time}_{imx_characteristic}_"
