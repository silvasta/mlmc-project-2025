# Install Pi

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

Choose `yes` if it asks for fingerprint, and it will add you to known hosts.

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

rpicam-still -o ~/Desktop/image.jpg
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
