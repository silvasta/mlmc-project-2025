import torch
from ultralytics import YOLO

# Check for CUDA device and set it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Total device memory: {torch.cuda.get_device_properties(0).total_memory}")
# 16770007040

### Load a model
## load a pretrained model (recommended for training)

# # used in experiments {1,2,}
# model = YOLO("yolo11n-cls.pt").to(device)
# results = model.train(data="cifar100", epochs=300,  imgsz=32)

# load an empty model
# used in experiments {3,4,9=5}
# model = YOLO("yolo12-cls.yaml")  # build new model
# model = YOLO(
#     "/home/silvan/Coding/mlmc-project-2025/runs/classify/train4/weights/last.pt"
# )
# results = model.train(data="cifar100", epochs=100, batch=256, imgsz=32)


# # # used in experiments {10-}
# model = YOLO("yolo11n-cls.pt").to(device)
# results = model.train(data="cifar100", epochs=100, batch=1000, imgsz=32)

# # used in experiments {14-}
model = YOLO("yolo11n-cls.pt").to(device)
results = model.train(data="cifar100", epochs=100, batch=500, imgsz=32)

# prints
print(model.device)

print("results is:")
print({results})

print("Training done")

# ### export not working! only for detection ouput
# # export
# model.export(format="imx")
