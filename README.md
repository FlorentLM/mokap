# Installation

## Basler Pylon SDK

* Download the installer package for your system: https://www2.baslerweb.com/en/downloads/software-downloads/

###### Linux-specific post-install
* You need to increase the limit on file descriptors and USB memory.
    Basler provides a script to do so automatically, but it may not completely work on all distros.

    Run `sudo chmod +x /opt/pylon/share/pylon/setup-usb.sh` and `sudo /opt/pylon/share/pylon/setup-usb.sh` (assuming you installed the Pylon SDK to the default `/opt/pylon` directory)

* **Note:** Basler's default increase on USB memory is 1000 Mib. This is, in our case, **not enough** for more than 3 USB cameras. 
  You can increase it even further by modifying the `/sys/module/usbcore/parameters/usbfs_memory_mb` file.
  A value of `2048` is enough for our 5 cameras.
* **Note:** On Arch-based systems, Basler's script is not sufficient. You need to manually add the line `DefaultLimitNOFILE=8192` to `/etc/systemd/user.conf`
* On systems that do not use GRUB, if you want to the USB memory setting to be persistent, Basler's script won't work. You need to change your bootloader options manually.
    
    For instance, EndeavourOS uses systemd-boot: edit `/efi/loader/entries/YOURDISTRO.conf` (replace `YOURDISTRO` by the name of the entry for your system, typically the machine-id in the case of EndeavourOS) and add `usbcore.usbfs_memory_mb=2048` to the `options` line.

## Mokap

### Windows, macOS and Linux

#### Miniconda:
* If you don't have Miniconda installed, see [here](https://docs.conda.io/projects/miniconda/en/latest/)
* Clone this repository `git clone https://github.com/FlorentLM/mokap`
* Create environment `conda env create --file=environment.yml`

###### Linux-specific optional dependencies
  * (Optional) Install [uhubctl](https://github.com/mvp/uhubctl) and follow their [post-install instructions](https://github.com/mvp/uhubctl#linux-usb-permissions).

###### Notes regarding disk write performance
In most situations, the bottleneck for acquiring high framerate videos from multiple cameras (in our case 5 cameras filming at 220 fps in 1440x1080 px) was disk IO.
We got very good performance using BTRFS on Linux, but your experience may vary.

On Windows, it is recommended to disable cache writing, otherwise the OS will try to optimise writing by using the available RAM, and it will crash quickly.
Although there are BTRFS drivers for Windows that work very well, they do not support no-cache writing (yet?), so it is not recommended to use BTRFS on Windows with Mokap.


# Usage

* Activate the conda environment `conda activate mokap`
* (Optional) Customise the example `main.py` file with your favourite text/code editor
* Run `./main.py`

# Troubleshooting

### Linux

    permission denied: ./main.py

**Fix**: make the file executable `chmod u+x ./main.py`

---

    /bin/sh: 1: uhubctl: not found

**Fix**: Install [uhubctl](https://github.com/mvp/uhubctl)

_Note_: This dependency is optional

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

**Fix**:  Increase the number of file `nofile` file descriptors: `ulimit -n 8192` (or more)

_Note_: mokap normally does this automatically

---