from ultralytics import YOLO
from utils import load_device_print_information

device = load_device_print_information()

project_folder = "experiments"

models = [
    # "experiments/african-wildlife/train_l_to_convergence/weights/best.pt",
    # "experiments/african-wildlife/train_l_to_convergence_all/weights/best.pt",
    # "experiments/african-wildlife/train_m_to_convergence/weights/best.pt",
    # "experiments/african-wildlife/train_m_to_convergence_all/weights/best.pt",
    # "experiments/african-wildlife/train_n_to_convergence/weights/best.pt",
    # "experiments/african-wildlife/train_n_to_convergence_all/weights/best.pt",
    # "experiments/african-wildlife/train_s_to_convergence/weights/best.pt",
    # "experiments/african-wildlife/train_s_to_convergence_all/weights/best.pt",
    # "experiments/african-wildlife/train_x_to_convergence/weights/best.pt",
    # "experiments/african-wildlife/train_x_to_convergence_all/weights/best.pt",
    # "experiments/coco128/train_n_to_convergence/weights/best.pt",
    # "experiments/coco8/train_n_to_convergence/weights/best.pt",
    # "experiments/coco8/train_n_to_convergence2/weights/best.pt",
]
models += [
    # onnx
    # "experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_model/best_imx.onnx",
    "experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_pqt/best_imx.onnx",
    "experiments/african-wildlife/train_s_to_convergence/weights/best_imx_model/best_imx.onnx",
    "experiments/african-wildlife/train_s_to_convergence_all/weights/best_imx_model/best_imx.onnx",
    # VisDrone
    "experiments/VisDrone/train_n_to_convergence/weights/best.pt",
]

for path in models:
    dataset_name = path.split("/")[1]
    data = f"{dataset_name}.yaml"
    project_folder = "experiments"
    project = f"{project_folder}/{dataset_name}"
    # print(project)
    train = path.split("/")[2]
    letter = train.split("_")[1:]
    name = "val_" + "_".join(letter)
    # print(name)
    # load model
    model = YOLO(path)
    metrics = model.val(
        # data=data,
        # # device=device,
        # # imgsz=640,
        # project=project,
        # name=name,
        # val=True,
        # plots=True,
    )
    file = metrics.to_df().to_csv(f"{project}/{name}/metric.csv")


# Run inference
# results = imx_model("https://ultralytics.com/images/bus.jpg")
