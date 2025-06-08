import platform

# Height (in pixels) of the OS taskbar
# TODO: this probably should be improved
if platform.system() == 'Windows':
    TASKBAR_H = 48
    TOPBAR_H = 23
else:
    TASKBAR_H = 48
    TOPBAR_H = 23

SPACING = 5