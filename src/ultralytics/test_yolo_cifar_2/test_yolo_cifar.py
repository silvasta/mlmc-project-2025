import torch
from ultralytics import YOLO

# Check for CUDA device and set it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Total device memory: {torch.cuda.get_device_properties(0).total_memory}")

batches = [
    12500,
    8192,
    1024,
    128,
]

for batch in batches:
    print(f"Start with {batch}")
    # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-cls.pt").to(device)
    # train
    results = model.train(
        data="../../datasets/cifar100",
        epochs=350,
        batch=batch,
        imgsz=32,
    )
    # prints
    print(f"result for batch {batch} is:")
    print({results})

print("Training done")
