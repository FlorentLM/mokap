<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="mokap/gui/icons/mokap.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Mokap</h3>

  <p>
    An easy to use but powerful multi-camera acquisition software
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
![Screenshot](screenshot.png)

Mokap is an easy to use multi-camera acquisition software developed for animal behaviour recording using hardware-triggered (synchronised) machine vision cameras.

### Features
* Cross platform (Linux, Windows, macOS)
* Supports synchronised cameras (only using a Raspberry Pi for now, but other modes will come soon)
* Supports encoding to individual frames or straight to video (with or without GPU encoding)
* (Coming soon) Live camera calibration for 3D triangulation 

<p>(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

#### ffmpeg (optional)
If you wish to use straight-to-video encoding, you will need [ffmpeg](https://www.ffmpeg.org/download.html) installed on your machine.

* Linux (most Debian-based distros):
   ```sh
   sudo apt install ffmpeg
   ```
* Windows:
   ```sh
   winget install --id Gyan.FFmpeg
   ```
* macOS:
   ```sh
   brew install ffmpeg
   ```
If you do not want to use ffmpeg, you can still use Mokap in image mode (videos will be written as individual frames)

#### uv:

We recommend using uv to manage Python environments and install Mokap easily.
* If you don't have uv installed, see [here](https://github.com/astral-sh/uv).

### Installation

#### Basler Pylon SDK

* Download the installer package for your system: https://www2.baslerweb.com/en/downloads/software-downloads/

###### Linux-specific post-install
* You need to increase the limit on file descriptors and USB memory.
    Basler provides a script to do so automatically, but it may not completely work on all distros.

    Run `sudo chmod +x /opt/pylon/share/pylon/setup-usb.sh` and `sudo /opt/pylon/share/pylon/setup-usb.sh` (assuming you installed the Pylon SDK to the default `/opt/pylon` directory)

* **Note:** Basler's default increase on USB memory is 1000 Mib. This is, in our case, **not enough** for more than 3 USB cameras. 
  You can increase it even further by modifying the `/sys/module/usbcore/parameters/usbfs_memory_mb` file.
  A value of `2048` is enough for our 5 cameras.
* **Note:** On Arch-based systems, you need to manually add the line `DefaultLimitNOFILE=2048` to `/etc/systemd/user.conf` (or `/etc/systemd/system.conf` if you want to apply it system-wide)
* On systems that do not use GRUB, if you want to the USB memory setting to be persistent, Basler's script won't work. You need to change your bootloader options manually.
    
    For instance, EndeavourOS uses systemd-boot: edit `/efi/loader/entries/YOURDISTRO.conf` (replace `YOURDISTRO` by the name of the entry for your system, typically the machine-id in the case of EndeavourOS) and add `usbcore.usbfs_memory_mb=2048` to the `options` line.

#### Mokap

1. Clone this repository:
   ```sh
   git clone https://github.com/FlorentLM/mokap.git
   ```
2. Create environment:
   ```sh
   cd mokap && uv sync
   ```
<p>(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Customise `config_example.yaml` and rename it to `config.yaml` (or whatever you want)

Starting example for 5 cameras (replace the xxxxx by your cameras' serial numbers):
```yaml
# General parameters
base_path: D:/            # Where the recordings will be saved
save_format: 'mp4'        # or jpg, bmp, tif, png
save_quality: 80          # 0 - 100%
gpu: True                 # Only used by the video encoder (i.e. if you use mp4 in save_format)

# Add/remove sources below
sources:
    strawberry:
        type: basler
        serial: 401xxxxx
        color: da141d
    avocado:
        type: basler
        serial: 401xxxxx
        color: 7a9c21
    banana:
        type: basler
        serial: 401xxxxx
        color: f3d586
    blueberry:
        type: basler
        serial: 401xxxxx
        color: 443e93
    coconut:
        type: basler
        serial: 401xxxxx
        color: efeee7
```

### Start GUI

1. Activate the uv environment within mokap. On Linux this is `source .venv/bin/activate`
2. Run `./mokap.py`

*Note: There are some default values hardcoded in `mokap.py`, but they can be changed with the GUI*

### Hardware Trigger

**Important**: The default in `mokap.py` is to use a hardware trigger (Raspberry Pi). For this, you **_MUST_** have three environment variables defined.
The recommended way is to create a file named `.env` that contains the three variables:

For example (replace with your trigger's IP or hostname, username and passsword):
```dotenv
TRIGGER_HOST=192.168.0.10
TRIGGER_USER=pi
TRIGGER_PASS=hunter2
```

You can disable the hardware trigger by editing the `mokap.py` file line 6:
```python
mc = MultiCam(config='./config.yaml', triggered=False, silent=False)
```
(or the name you chose for the config file)

You must enable the GPIO interface on the PI using:
```
sudo raspi-config
```

Make sure that `Dhcpcd` and `pigpiod` GPIO pin trigger services are running on PI using:
```
sudo systemctl status <service>
```
Make sure that  systemd-networkd service is running on Linux Desktop if connected via wired connection.
Make sure you can ping between devices. You made need to explicitly set IP addresses with subnet.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Remarks

* If you plan on recording high framerate from many cameras, you probably want to use the GPU, as the software encoders and the image encoding are both slower


<!-- ROADMAP -->
## Roadmap

- [x] Allow GPU video encoding
- [x] Replace Tk with Qt as the GUI framework
- [ ] Finish calibration mode
- [ ] Add support for other camera brands (FLIR, etc)
- [ ] Add support for other kinds of triggers (primary/secondary cameras, Arduino, etc)
- [ ] Remember settings set with the GUI instead of using hardcoded values in `mokap.py`

<p>(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

### Linux

    permission denied: ./mokap.py

**Fix**: make the file executable `chmod u+x ./mokap.py`

---

    Failed to open device xxxxx for XML file download. Error: 'The device cannot be operated on an USB 2.0 port. The device requires an USB 3.0 compatible port.'

**Fix**: Unplug and plug the camera(s) again

---

    Warning: Cannot change group of xxxx to 'video'.

**Fix**:  Add the local user to the *video* group: `sudo usermod -a -G video $USER`

---

    Error: 'Insufficient system resources exist to complete the API.'
    
or
    
    Too many open files. Reached open files limit

**Fix**:  Increase the number of open file descriptors: `ulimit -n 2048` (or more)

_Note_: mokap normally does this automatically

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p>(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Florent Le Moel - [@optic_flo](https://twitter.com/optic_flo)

Project Link: [https://github.com/FlorentLM/mokap](https://github.com/github_username/mokap)

<p>(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/FlorentLM/mokap.svg?style=for-the-badge
[forks-url]: https://github.com/FlorentLM/mokap/network/members
[stars-shield]: https://img.shields.io/github/stars/FlorentLM/mokap.svg?style=for-the-badge
[stars-url]: https://github.com/FlorentLM/mokap/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[product-screenshot]: screenshot.png
