from ultralytics import YOLO
from utils import load_device_print_information

device = load_device_print_information()

model = YOLO("yolo11n.pt")
# model = YOLO("models/yolo11n.pt").to(device)
# model = YOLO("/home/silvan/ultralytics/models/yolo11n.pt")
# model = YOLO("/home/silvan/Coding/mlmc-project-2025/src/ultralytics/models/yolo11n.pt")
# print(model)

### params

# dataset_name = "african-wildlife"
# dataset_name = "coco8"
# dataset_name = "coco128"
dataset_name = "VisDrone"
# VOC ???
# xView ???

data = f"{dataset_name}.yaml"

experiment_name = "initial_setup"

optimizer = "auto"  # SGD, Adam, AdamW, NAdam, RAdam, RMSProp

### train
results = model.train(
    data=data,
    device=device,
    epochs=5,
    # time = None,
    patience=50,  # 100
    # batch=16,
    # imgsz=640,
    # cache = False,
    project=f"test_yolo/{dataset_name}",
    name=experiment_name,
    optimizer=optimizer,
    # classes=None,
    # resume=False,
    # fraction=1.0,
    # val=True,
    plots=True,
)

# print()
# print("Results")
# print({results})
# print()
print("Training done")
