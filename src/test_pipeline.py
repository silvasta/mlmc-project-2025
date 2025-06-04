from ultralytics import YOLO

# Load a YOLO11n PyTorch model
# model = YOLO("yolo11n.pt")
model = YOLO("test_yolo/african-wildlife/train_s_to_convergence/weights/best.pt")

# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category


# Export the model
# model.export(
#     format="imx",
#     # data="coco8.yaml"
# )  # exports with PTQ quantization by default


# Load the exported model
# imx_model = YOLO("yolo11n_imx_model")


# Run inference
# results = imx_model("https://ultralytics.com/images/bus.jpg")
