<!-- markdownlint-configure-file { "MD013": { "line_length": 200} } -->

# Install Pi

## Raspberry Pi Imager - Setup OS

### Settings GUI

Raspberry Pi Device: RASPBERRY PI 5

OS: PI OS (64-BIT) Bookworm

Card: SanDisk Extreme, 128 GB, 190/70 MB/s (read/write), adapter marked with green

### Settings OS Customisation

#### General

- [x] Set hostname: `rpi-bw518`.local

- [x] Set username and Password

  Username: `silvan`

  Password: `XXXXXX`
  <!-- Password: `mlmc25` -->

- [x] Configure WLAN

  SSID: `WN-0286DF`

#### Services

- [x] Enable SSH

  - Allow public-key authentication

    Set authorized_keys for 'silvan':

    ```
    ssh-rsa XXXXXXXXXXXXXXXXXXXXX= silvan@silvan-OMEN-u2504
    ```

See in the [appendix](#setup-ssh-keypair) for further information about ssh-keypair.

## Connection

Test connection with:

```bash
ping rpi-bw518.local
```

### WLAN

If the installation worked fine, the Pi should directly connect to the WLAN.
Unfortunately just to the one specified at the beginning.
Not tested with the ETH eduroam.

Connected Pi with eduroam, connection to laptop works,
outgoing connection is broken.

#### WIFI Trouble Shooting

- Modify connection

```bash
# turn wifi on
rfkill block wifi
# turn wifi off
rfkill unblock wifi
# see current stats
rfkill
```

(Did not solve the problem)

- Modify connections from command line UI

```bash
sudo nmtui
```

(not sure how good that works)

- Next approach

```bash
# check connection, like `ifconfig` but only wifi
iwconfig
# if wlan down, f.e. from visual connection
sudo ip link set wlan0 up
# scan all
sudo iwlist wlan0 scan
# filter for important statements
sudo iwlist wlan0 scan | grep ESSID
# filter but in modern
sudo iwlist wlan0 scan | rg ESSID
# filter but in modern
sudo iwlist wlan0 scan | rg -e ESSID -e Freq
```

The network connection didn't worked at the end...

..here (at the end) the commands:

<https://en.ubunlog.com/nmtui-or-nmcli-establishes-Wi-Fi-connection-from-terminal/>

- This one creates **very good results**

```bash
# check if wifi is enabled
nmcli radio wifi
# otherwise turn on (needs sudo on pi-os)
sudo nmcli radio wifi on
##very nice terminal output, table with all information
nmcli dev wifi list # use this! #TODO: attach to main manual
# one of both
sudo nmcli dev wifi connect {network-ssid}
sudo nmcli dev wifi connect {network-ssid} password {network-password}
# or this one
sudo nmcli dev wifi connect network-ssid -a
```

```bash
silvan@rpi-bw518:~ $ sudo nmcli dev wifi connect silvasta14 -a
Password: ••••••••
Device 'wlan0' successfully activated with '56aafde7-a957-47d6-8dc9-0577e793f7c5'.
silvan@rpi-bw518:~ $
```

That one worked impressively well! Thanks to the 2 guys from:

<https://unix.stackexchange.com/questions/675099/how-to-connect-to-wifi-with-nmcli>

### Ethernet

Plug the Ethernet cable to the Pi and to the Laptop.

Set"Wired Connection" on the Laptop to "Shared to other computers"

Works fine with the disadvantage of the cable...

### Setup Full Computer

Needs:

- [x] Good power supply
- [ ] Keyboard with USB-Connection
- [x] Mouse with USB-Connection
- [x] Screen with cable ending at MicroHDMI

### SSH

```bash
ssh [username]@[hostname].local
ssh [username]@[ip-adress]
```

To get the hostname and the IP:

```bash
hostname
hostname -I
```

(but one needs already a connection, or a physical display)

If the hostname is properly set, one can use this approach:

```bash
ssh silvan@rpi-bw518.local
```

Choose `yes` if it asks for fingerprint,
and it will add you to known hosts.

#### Copy files over SSH

Both commands executed from laptop

```bash
# copy file from pi to laptop
scp silvan@rpi-bw518:/home/silvan/test/bla.txt /home/silvan/Desktop
```

```bash
# copy file from laptop to pi
scp /home/silvan/test.py silvan@rpi-bw518.local:/home/silvan/test
```

#### Visual connection

The basic VNC installation of the RPi failed.
Tigervnc didn't produce good results.
Some workarounds failed or endangered other parts of the installation.

RDP worked somehow but the effort to connect from everywhere was to big.

### Rpi-Connect

Finally a stable visual connection trough the internet was established with Raspberry Pi Connect.

If not installed already, use this before:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install rpi-connect
```

### Setup rpi-connect

```bash
rpi-connect on
rpi-connect signin
```

Follow the link, create an account and connect the device.

## Camera

```bash
sudo apt install imx500-all
sudo reboot
```

### Basic tests

```bash
# standard test
rpicam-hello
# pictures with some duration, probably settings, calibration
rpicam-still -o test_image.jpg
# adjust size
rpicam-still -o image-small.jpg --width 640 --height 480
# create a video
rpicam-vid -o ~/Desktop/video.mp4
# only with visual connection
vlc ~/Desktop/video.mp4
```

### Test pre-installed Camera Models

```bash
# again the standard test
rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30
# output with format .264?
rpicam-vid -t 10s -o mobilenet.264 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30

# output with format .mp4
rpicam-vid -t 10s -o mobilenet.mp4 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30
```

```bash
# copy to laptop (in case cloud-sync not available)
scp silvan@rpi-bw518:/home/silvan/PolyBox/test-pictures/mobilenet.mp4  /home/silvan/PolyBox/pi/test-pictures/mobilenet.mp4
```

## Model

### Prepare Custom Model

Execute this scrips, on laptop or on pi

```bash
#!/bin/bash

### transforms output from model compress to camera format

# setup environment
source /home/silvan/mlmc/.venv/bin/activate

# input
IMX_IN_FILE=${2:-/home/silvan/mlmc/experiments/african-wildlife/train_n_to_convergence_all/weights/best_imx_model/best_imx.onnx}

# output
IMX_OUT_NAME=${1:-"model_$(date +%F)"}
IMX_OUT_DIR=${3:-/home/silvan/mlmc/imx_models}
IMX_OUT_PATH="$IMX_OUT_DIR/$IMX_OUT_NAME"

# test directories
echo "$IMX_IN_FILE"
echo "$IMX_OUT_PATH"

# finally, do the conversion
imxconv-pt -i "$IMX_IN_FILE" -o "$IMX_OUT_PATH" --no-input-persistency --overwrite-output
```

(in case the model is not automatically converted by the Sony model compression toolkit)

### Load Custom Model

```bash
# needs to be installed
sudo apt install imx500-tools
```

```bash
# generic
imx500-package -i <path to packerOut.zip> -o <output folder>
# in case you are in the folder with the zip and wants the model there
imx500-package -i packerOut.zip  -o .
```

```bash
# result should look like this
silvan@rpi-bw518:~/mlmc/imx_models/yolo_n_1 $ ls
best_imx.pbtxt  best_imx_MemoryReport.json  dnnParams.xml  network.rpk  packer  packerOut.zip
```

```bash
python imx500_object_detection_demo_mp.py --model yolo_n_1/network.rpk --fps 17 --bbox-normalization --labels labels.txt
```

## Appendix

### Failure imx500-converter

```bash
silvan@rpi-bw518:~ $ pip install imx500-converter[pt] --break-system-packages
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Collecting imx500-converter[pt]
  Using cached imx500_converter-3.16.1-py3-none-any.whl (21 kB)
Collecting sdspconv~=3.16.1
  Using cached sdspconv-3.16.1-py3-none-any.whl (54.7 MB)
Collecting uni-pytorch==3.16.1
  Using cached uni_pytorch-3.16.1-py3-none-any.whl (265 kB)
Collecting uni-model==9.0.10
  Using cached uni_model-9.0.10-py3-none-any.whl (171 kB)
Collecting networkx~=3.0.0
  Using cached https://www.piwheels.org/simple/networkx/networkx-3.0-py3-none-any.whl (2.0 MB)
Requirement already satisfied: numpy<2 in /usr/lib/python3/dist-packages (from uni-pytorch==3.16.1->imx500-converter[pt]) (1.24.2)
Collecting mct-quantizers~=1.5.0
  Using cached mct_quantizers-1.5.2-py3-none-any.whl (104 kB)
Collecting packaging
  Using cached packaging-25.0-py3-none-any.whl (66 kB)
Collecting sony-custom-layers~=0.3.0
  Using cached sony_custom_layers-0.3.0-py3-none-any.whl (36 kB)
Collecting onnx==1.16.1
  Using cached onnx-1.16.1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (15.8 MB)
Collecting onnxruntime~=1.19.2
  Using cached onnxruntime-1.19.2-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl (11.5 MB)
Collecting onnxruntime-extensions~=0.13.0
  Using cached onnxruntime_extensions-0.13.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (3.3 MB)
Collecting protobuf>=3.20.2
  Using cached protobuf-6.31.0-cp39-abi3-manylinux2014_aarch64.whl (321 kB)
  Using cached protobuf-4.25.5-cp37-abi3-manylinux2014_aarch64.whl (293 kB)
Collecting stringcase
  Using cached https://www.piwheels.org/simple/stringcase/stringcase-1.2.0-py3-none-any.whl (4.1 kB)
Collecting coloredlogs
  Using cached https://www.piwheels.org/simple/coloredlogs/coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
Collecting flatbuffers
  Using cached https://www.piwheels.org/simple/flatbuffers/flatbuffers-20181003210633-py2.py3-none-any.whl (14 kB)
Collecting sympy
  Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
Collecting humanfriendly>=9.1
  Using cached https://www.piwheels.org/simple/humanfriendly/humanfriendly-10.0-py2.py3-none-any.whl (89 kB)
Collecting mpmath<1.4,>=1.1.0
  Using cached https://www.piwheels.org/simple/mpmath/mpmath-1.3.0-py3-none-any.whl (536 kB)
Installing collected packages: stringcase, sdspconv, onnxruntime-extensions, mpmath, flatbuffers, sympy, protobuf, packaging, networkx, imx500-converter, humanfriendly, uni-model, sony-custom-layers, onnx, mct-quantizers, coloredlogs, onnxruntime, uni-pytorch
  WARNING: The script sdspconv is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script isympy is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts imxconv-pt and imxconv-tf are installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script humanfriendly is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The scripts backend-test-tools, check-model and check-node are installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script coloredlogs is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script onnxruntime_test is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
  WARNING: The script uni-pytorch is installed in '/home/silvan/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
Successfully installed coloredlogs-15.0.1 flatbuffers-20181003210633 humanfriendly-10.0 imx500-converter-3.16.1 mct-quantizers-1.5.2 mpmath-1.3.0 networkx-3.0 onnx-1.16.1 onnxruntime-1.19.2 onnxruntime-extensions-0.13.0 packaging-25.0 protobuf-4.25.5 sdspconv-3.16.1 sony-custom-layers-0.3.0 stringcase-1.2.0 sympy-1.14.0 uni-model-9.0.10 uni-pytorch-3.16.1
```

### Setup ssh-keypair

```bash
silvan@silvan-OMEN-u2504:~/.ssh$ ssh-keygen -t rsa
Generating public/private rsa key pair.
Enter file in which to save the key (/home/silvan/.ssh/id_rsa): rpi_rsa
silvan@silvan-OMEN-u2504:~/.ssh$ cat rpi_rsa
-----BEGIN OPENSSH PRIVATE KEY-----
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
-----END OPENSSH PRIVATE KEY-----
silvan@silvan-OMEN-u2504:~/.ssh$ cat rpi_rsa.pub
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDSxEt/R6RtzOh9EBCsM7R8767vd6u4kiKbTO3JCq72pYkiJvCoJmizfqmoxJHeBmEM+2zLxtBUZStbtUzzlCAyASrIU3rAqI0SZpOgN39RcZAvJLfFWDUB42V3BsVq6NXHZwUjxQHIoH9vmxgRcvCR+hSqGutbNZ+V6DF2Yu/WlspBiSQxX9YaZ4QLaH0io9dXA+6v+jEUDshtKODtKoIS4rDqSVuMLlyLV/9XWUOopwd/stDDHgek5Oei8Cd+rwEuLXHo2Qs72GUuHSh9ZRrIzEBTkudB/g7Da0bmyAM8dYCLeXxFjMvEuHJnWjIsO0yovAPgcsSB6StG2kKVQ/B56bLs4XEwkN1knSbV+bkwZG6tjDBQlWEY8yjRkkQKmvhW7yamevktJ3Wl6GhzMdWbPRsulQJ1/rO01q+f4vPpfuElqqVrXBWRxb3MNfuAYCLbSAUsC+8vDV2I/kQKJ2dASQWfYg3RwQKFbY9nzW7XdAQE3hPO9F77rtXURZ9+oBc= silvan@silvan-OMEN-u2504
silvan@silvan-OMEN-u2504:~$ ssh-add .ssh/rpi_rsa
Identity added: .ssh/rpi_rsa (silvan@silvan-OMEN-u2504)
```
