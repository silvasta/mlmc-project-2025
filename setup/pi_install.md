# Install Pi

<!-- markdownlint-configure-file { "MD013": { "line_length": 400} } -->

## Raspberry Pi Imager

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

## Run the Raspberry Pi

### Connection

Test connection with:

```bash
ping rpi-bw518.local
```

#### Type of Connection

##### WLAN

If the installation worked fine, the Pi should directly connect to the WLAN.
Unfortunatelly just to the one specified at the beginning.
Not tested with the ETH eduroam.

Connected Pi with Eduroam, connection to laptop works,
outgoing connection is broken.

```bash
# turn wifi on and off
rfkill block wifi
rfkill unblock wifi
# see current stats
rfkill
```

(Did not solve the problem)

##### Ethernet

Plug the Ethernet cable to the Pi and to the Laptop.
Set"Wired Connection" on the Laptop to "Shared to other computers"

##### Setup as full computer

- [x] Good power supply
- [ ] Keyboard with USB-Connection
- [x] Mouse with USB-Connection
- [x] Screen with cable ending at MicroHDMI

#### SSH

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

##### Copy files over SSH

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

Finally a stable connection with Raspberry Pi Connect was established.

If not installed already, use this before:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install rpi-connect
```

##### Setup rpi-connect

```bash
rpi-connect on
rpi-connect signin
```

Follow the link, create an account and connect the device.

### Camera

```bash
sudo apt install imx500-all
sudo reboot
```

#### Basic tests

```bash
rpicam-hello

rpicam-still -o test_image.jpg
rpicam-still -o ~/Desktop/image-small.jpg --width 640 --height 480

rpicam-vid -o ~/Desktop/video.mp4
vlc ~/Desktop/video.mp4
```

On Laptop:

```bash
scp silvan@rpi-bw54:/home/silvan/Desktop/video.mp4 /home/silvan/Desktop/video.mp4
vlc ~/Desktop/video.mp4
```

#### Testing camera features

```bash
rpicam-hello -t 0s --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080 --framerate 30

rpicam-vid -t 10s -o output.264 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30

rpicam-vid -t 10s -o output.mp4 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30
```

```bash
scp silvan@rpi-bw54:/home/silvan/Desktop/output.mp4  /home/silvan/Desktop/output.mp4
```

#### Prepare for ultralytics

```bash
scp ultralytics/yolo11n.onnx silvan@rpi-bw518.local:/home/silvan/Desktop/
```

```bash
python -m venv ptv1
pip install imx500-converter[pt]
```

```bash
imxconv-pt -i /home/silvan/Desktop/yolo11n.onnx -o /home/Desktop/ --no-input-persistency
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
