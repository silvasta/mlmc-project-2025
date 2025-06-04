import torch
from ultralytics import YOLO

# Check for CUDA device and set it
device = "cuda" if torch.cuda.is_available() else "cpu"
# extended print
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use the GPU.")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch cannot use the GPU.")
# regular  print
print(f"Using device: {device}")
print(f"Total device memory: {torch.cuda.get_device_properties(0).total_memory}")

batches = [500]

for batch in batches:
    print(f"Start with {batch}")
    # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-cls.pt").to(device)
    # train
    results = model.train(
        # data="../../datasets/cifar100",
        data="/home/silvan/Coding/mlmc-project-2025/datasets/cifar100/",
        epochs=500,
        batch=batch,
        imgsz=32,
    )
    # prints
    print(f"result for batch {batch} is:")
    print({results})

print("Training done")
