import tensorflow as tf
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Test 1
print("### Test 1")
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))

# Test 2
print("### Test 2")
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    print("We got a GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("Sorry, no GPU for you...")
