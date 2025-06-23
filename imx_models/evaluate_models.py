import sys
import time

import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

models = [
    "n_640/network.rpk",
    "n_to_convergence_all-2025-06-23_03-34-40/network.rpk",
    "n_to_convergence_all-2025-06-23_03-38-14/network.rpk",
    "s_to_convergence-2025-06-23_03-41-47/network.rpk",
    "s_to_convergence_all-2025-06-23_03-49-35/network.rpk",
    "yolo_n_1/network.rpk",
]


def get_inference():
    picam2.start()
    times = []
    for _ in range(50):
        start_time = time.time()
        metadata = picam2.capture_metadata()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    avg_time = np.mean(times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    result = {"avg_inference_time_ms": avg_time, "fps": fps}
    picam2.stop()
    return result


if __name__ == "__main__":
    # This must be called before instantiation of Picamera2
    results = {}
    for model in models:
        imx500 = IMX500(model)
        intrinsics = imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            print("Network is not an object detection task", file=sys.stderr)
            exit()
        picam2 = Picamera2(imx500.camera_num)
        result = get_inference()
        results[model] = result
        picam2.close()
    for model, metrics in results.items():
        print(
            f"Model {model}: Avg Inference Time = {metrics['avg_inference_time_ms']:.2f} ms, FPS = {metrics['fps']:.2f}"
        )
