import torch
from ultralytics import YOLO

# Check for CUDA device and set it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Total device memory: {torch.cuda.get_device_properties(0).total_memory}")
print()
print()

# load a pretrained model (recommended for training)
model = YOLO(
    "/home/silvan/Coding/mlmc-project-2025/src/ultralytics/test_yolo_cifar_2/runs/classify/train5/weights/last.pt"
).to(device)
print()
print()

# train
results = model.train(
    # data="../../datasets/cifar100",
    data="/home/silvan/Coding/mlmc-project-2025/datasets/cifar100/",
    epochs=200,
    batch=128,
    imgsz=32,
)
# prints
print()
print()
print({results})
print()
print("Training done")
