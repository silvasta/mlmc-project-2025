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

optimizer = "auto"  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp

resume = False
# resume = True

for t in ["n", "s", "m", "l", "x"]:
    batch_size = 8 if t == "x" else 16
    model_type = t  # one of: [n, s, m, l, x] for detection
    experiment_name = f"train_{model_type}_to_convergence_all"

    if resume:
        model_path = f"{project_name}/{dataset_name}/{experiment_name}/weights/last.pt"
        model = YOLO(model_path)
        ### train
        results = model.train(resume=resume)
    else:
        # model_path = "yolo11n.pt"
        model_path = f"yolo11{model_type}.pt"
        model = YOLO(model_path)
        ### train
        results = model.train(
            data=data,
            device=device,
            epochs=500,
            # time = None,
            patience=30,  # 100,
            batch=batch_size,
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

    print(f"{experiment_name} done")

print("Training completely done!")
