import numpy as np
import os
from subprocess import  check_output
import platform


def setup_ulimit(wanted_value=8192, silent=True):
    """
    Sets up the maximum number of open file descriptors for nofile processes
    It is required to run multiple (i.e. more than 4) Basler cameras at a time
    """

    if 'Linux' not in platform.system():
        return

    out = os.popen('ulimit')
    ret = out.read().strip('\n')

    if ret == 'unlimited':
        hard_limit = np.inf
    else:
        hard_limit = int(ret)

    out = os.popen('ulimit -n')
    current_limit = int(out.read().strip('\n'))

    if current_limit < wanted_value:
        if not silent:
            print(f'[WARN] Current file descriptors limit is too small (n={current_limit}), '
                  f'increasing it to {wanted_value} (max={hard_limit}).')
        os.popen(f'ulimit -n {wanted_value}')
    else:
        if not silent:
            print(f'[INFO] Current file descriptors limit seems fine (n={current_limit})')


# TODO: Use this in a GUI helper to populate settings
def get_encoders(ffmpeg_path='ffmpeg', codec='hevc'):
    """
    Get available encoders for the given codec. This is tailored for h264 and hevc (h265)
    but this may work with others
    """
    if '265' in codec:
        codec = 'hevc'
    elif '264' in codec:
        codec = 'h264'

    r = check_output([ffmpeg_path, '-hide_banner', '-codecs'], stderr=False).decode('UTF-8').splitlines()
    codec_line = list(filter(lambda x: codec in x, r))[0]

    all_encoders = codec_line.split('encoders: ')[1].strip(')').split()

    encoders_names = []

    for encoder in all_encoders:
        r = check_output([ffmpeg_path, '-hide_banner', '-h', f'encoder={encoder}'], stderr=False).decode('UTF-8').splitlines()
        name_line = list(filter(lambda x: f'Encoder {encoder}' in x, r))[0]
        true_name = name_line.split(f'Encoder {encoder} [')[1][:-2]
        encoders_names.append(true_name)

    unique_encoders = {}
    for a, b in zip(all_encoders, encoders_names):
        if b not in unique_encoders:
            unique_encoders[b] = a
    return list(unique_encoders.values())

# TODO: We should also query the supported presets and options and auto adjust ...



# TODO: These probably should be exposed as an advanced tool in the GUI
def enable_usb(hub_number):
    """
    Uses uhubctl on Linux to enable the USB bus
    """
    if 'Linux' in platform.system():
        out = os.popen(f'uhubctl -l {hub_number} -a 1')
        ret = out.read()


def disable_usb(hub_number):
    """
    Uses uhubctl on Linux to disable the USB bus (effectively switches off the cameras connected to it
    so they don't overheat, without having to be physically unplugged)
    """
    if 'Linux' in platform.system():
        out = os.popen(f'uhubctl -l {hub_number} -a 0')
        ret = out.read()

