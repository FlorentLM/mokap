import subprocess
from pathlib import Path
from typing import Union

import numpy as np
from subprocess import  check_output
import os
import time
import errno
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


def is_locked(path: str) -> bool:
    path = str(path)
    if platform.system().startswith('Windows'):
        import win32file, win32con
        GENERIC_READ = win32con.GENERIC_READ
        OPEN_EXISTING = win32con.OPEN_EXISTING
        FILE_SHARE_READ = 0
        FILE_SHARE_WRITE = 0
        try:
            handle = win32file.CreateFile(
                path,
                GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                None,
                OPEN_EXISTING,
                0,
                None
            )
        except Exception:
            return True
        else:
            win32file.CloseHandle(handle)
            return False
    else:
        import fcntl
        try:
            with open(path, 'a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            return False
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise


def wait_until_unlocked(path: str, timeout: float = 5.0, poll: float = 0.1) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if not is_locked(path):
            return True
        time.sleep(poll)
    return False


def wait_for_size_stable(path: str, checks: int = 3, pause: float = 0.2) -> bool:
    try:
        prev = os.path.getsize(path)
    except OSError:
        return False
    for _ in range(checks):
        time.sleep(pause)
        try:
            curr = os.path.getsize(path)
        except OSError:
            return False
        if curr != prev:
            prev = curr
            return False
    return True


def safe_replace(src: str, dst: str, *,
                 lock_timeout: float = 5.0,
                 size_checks: int = 3,
                 size_pause: float = 0.2,
                 replace_timeout: float = 5.0) -> bool:

    # wait until both files are unlocked
    for p in (src, dst):
        if os.path.exists(p):
            if not wait_until_unlocked(p, timeout=lock_timeout):
                return False

    # wait until src looks finished
    if not wait_for_size_stable(src, checks=size_checks, pause=size_pause):
        # might still be writing
        return False

    # try replace in loop
    start = time.time()
    while True:
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if time.time() - start > replace_timeout:
                return False
            time.sleep(0.1)


def get_size(path: Union[Path, str]) -> int:
    if os.path.isfile(path):
        return os.stat(path).st_size    # single file: one stat call

    # directory: recursive call
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            try:
                if entry.is_dir(follow_symlinks=False):
                    total += get_size(entry.path)
                else:
                    # DirEntry.stat() is cached, no extra syscall if already fetched
                    total += entry.stat(follow_symlinks=False).st_size
            except:
                pass
    return total