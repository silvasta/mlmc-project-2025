# Manual for computer preparation and environment installation

## Requirements for GPU usage (Ubuntu)

Check NVIDIA driver version

```bash
silvan@silvan-OMEN-u2504:~$ nvidia-smi
Thu May  1 23:17:26 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 570.133.07     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   37C    P5             14W /   40W |      13MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4160      G   /usr/bin/gnome-shell                      3MiB |
+-----------------------------------------------------------------------------------------+
```

Check NVIDIA cuda version

```bash
nvcc --version
```

```bash
sudo apt install nvidia-cuda-toolkit
```

```bash
Installing:
  nvidia-cuda-toolkit

Installing dependencies:
  ca-certificates-java     libcublaslt12    libhsa-runtime64-1  libnppisu12            libtbb12               nvidia-cuda-dev
  cpp-12                   libcudart12      libhsakmt1          libnppitc12            libtbbbind-2-5         nvidia-cuda-gdb
  fonts-dejavu-extra       libcufft11       libhwloc-plugins    libnpps12              libtbbmalloc2          nvidia-cuda-toolkit-doc
  g++-12                   libcufftw11      libhwloc15          libnvblas12            libthrust-dev          nvidia-opencl-dev
  gcc-12                   libcuinj64-12.2  libibmad5           libnvidia-ml-dev       libucx0                nvidia-profiler
  gcc-12-base              libcupti-dev     libibumad3          libnvjitlink12         libvdpau-dev           nvidia-visual-profiler
  java-common              libcupti-doc     libllvm17t64        libnvjpeg12            libx11-dev             ocl-icd-opencl-dev
  libaccinj64-12.2         libcupti12       libnppc12           libnvrtc-builtins12.2  libxau-dev             opencl-c-headers
  libamd-comgr2            libcurand10      libnppial12         libnvrtc12             libxcb1-dev            opencl-clhpp-headers
  libamdhip64-5            libcusolver11    libnppicc12         libnvtoolsext1         libxdmcp-dev           openjdk-8-jre
  libatk-wrapper-java      libcusolvermg11  libnppidei12        libnvvm4               node-html5shiv         openjdk-8-jre-headless
  libatk-wrapper-java-jni  libcusparse12    libnppif12          libpfm4                nsight-compute         x11proto-dev
  libcu++-dev              libgcc-12-dev    libnppig12          librdmacm1t64          nsight-compute-target  xorg-sgml-doctools
  libcub-dev               libgl-dev        libnppim12          libstdc++-12-dev       nsight-systems         xtrans-dev
  libcublas12              libglx-dev       libnppist12         libtbb-dev             nsight-systems-target

Suggested packages:
  gcc-12-locales   gcc-12-multilib           libtbb-doc    nvidia-cuda-samples       fonts-ipafont-mincho
  cpp-12-doc       default-jre               libvdpau-doc  opencl-clhpp-headers-doc  fonts-wqy-microhei
  g++-12-multilib  libhwloc-contrib-plugins  libx11-doc    fonts-nanum               fonts-wqy-zenhei
  gcc-12-doc       libstdc++-12-doc          libxcb-doc    fonts-ipafont-gothic      fonts-indic

Recommended packages:
  libnvcuvid1

Summary:
  Upgrading: 0, Installing: 90, Removing: 0, Not Upgrading: 0
  Download size: 2,459 MB
  Space needed: 6,690 MB / 91.9 GB available
```

```bash
silvan@silvan-OMEN-u2504:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
```

```bash
silvan@silvan-OMEN-u2504:~$ sudo apt install nvidia-cudnn
Installing:
  nvidia-cudnn

Installing dependencies:
  nvidia-cuda-toolkit-gcc
```

## Tensorflow

```bash
conda create -n tensorflow_0501 python=3.12
conda activate tensorflow_0501
pip install tensorflow[and-cuda]
```

```bash
conda activate tensorflow_0501
```

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

```python
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
print("TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
```

## PyTorch

```bash
export NAME=pytorch_0501
python3 -m venv $HOME/Environments/$NAME
source $HOME/Environments/$NAME/bin/activate
pip install torch
```

```bash
source $HOME/Environments/pytorch_0501/bin/activate
```

```bash
pip install numpy
```

```python
import torch
if torch.cuda.is_available():
   print("CUDA is available! PyTorch can use the GPU.")
   print(f"Device count: {torch.cuda.device_count()}")
   print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
   print("CUDA is not available. PyTorch cannot use the GPU.")
```

## Ultralytics

```bash
export NAME=ultra_0502
python3 -m venv $HOME/Environments/$NAME
source $HOME/Environments/$NAME/bin/activate
pip install ultralytics
```
