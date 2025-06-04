from ultralytics import YOLO
from utils import load_device_print_information

device = load_device_print_information()

dataset_name = "african-wildlife"
# dataset_name = "coco8"
# dataset_name = "coco128"
# dataset_name = "VisDrone"
# VOC ???
# xView ???

data = f"{dataset_name}.yaml"

project_name = "test_yolo"
# experiment_name = "train_n_to_convergence"
experiment_name = "train_s_to_convergence"

optimizer = "auto"  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp

resume = False
resume = True
if resume:
    model_path = f"{project_name}/{dataset_name}/{experiment_name}/weights/last.pt"
    model = YOLO(model_path)
    ### train
    results = model.train(resume=resume)
else:
    # model_path = "yolo11n.pt"
    model_path = "yolo11s.pt"
    model = YOLO(model_path)
    ### train
    results = model.train(
        data=data,
        device=device,
        epochs=500,
        # time = None,
        patience=20,  # 100,
        # batch=16,
        # imgsz=640,
        # cache = False,
        project=f"{project_name}/{dataset_name}",
        name=experiment_name,
        optimizer=optimizer,
        # classes=None,
        # resume=resume,  # False,
        # fraction=1.0,
        # val=True,
        plots=True,
    )

print("Training done")
