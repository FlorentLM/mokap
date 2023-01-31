import pypylon.pylon as py
from pypylon import genicam
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum, unique
from mokap.hardware import SSHTrigger
from mokap import utils

##

# Open the connection to the camera
tlf = py.TlFactory.GetInstance()
cam = py.InstantCamera(tlf.CreateFirstDevice())

cam.GrabCameraEvents = True
cam.Open()

trigger = SSHTrigger()

##

# (Re)load the camera's Default powerup state
cam.UserSetSelector.SetValue("Default")
cam.UserSetLoad.Execute()

##

# Configure the GPIO Line 4 as input (Line 4 is on pin 3, see ./pins.png)
cam.LineSelector = "Line4"
cam.LineMode = "Input"

# Enable Hardware trigger using Line 4
cam.TriggerSelector = "FrameStart"
cam.TriggerMode = "On"
cam.TriggerSource = "Line4"

# Trigger activation value:
#           ________           ________
#          |        |         |        |
# _________|        |_________|        |_________
#          ↑        ↓
#        Rising   Falling
#
# FallingEdge apparently induces slightly less delay than RisingEdge:
# https://docs.baslerweb.com/io-timing-characteristics-%28ace-ace-2-boost%29.html
cam.TriggerActivation.Value = 'FallingEdge'

# Exposure Time mode:
#       'Standard'   mode:  minimum exposure time = 21 or 20 μs (8-bit or 12-bit pixel format, resp.), maximum = 10 sec
#       'UltraShort' mode:  minimum exposure time = 1 μs, maximum = 13 μs
# Exposure trigger mode:
#       'TriggerWidth'  mode:   exposure time = width of hardware trigger cycle
#       'Timed'         mode:   exposure time = set time
cam.ExposureTimeMode.SetValue('Standard')
cam.ExposureAuto = 'Off'
cam.ExposureMode = 'Timed'

##

# Pixel format
# https://docs.baslerweb.com/pixel-format
# acA1440-220um has three available formats:
#       - Mono 8 ('Mono8')
#       - Mono 12 ('Mono12')
#       - Mono 12 Packed ('Mono12p')
# Default 8-bits allows for quickest framerate, and should be sufficient in terms of detail.
# cam.PixelFormat.Value

##

#
# Delays and timing
#
# https://docs.baslerweb.com/acquisition-timing-information.html


# Optimal timings at camera's max native fps for acA1440-220um:
T_shortest_exposure = 21                # in μs
T_longest_exposure_maxfps = 4331        # in μs

# Get camera's max default fps
max_native_fps = cam.ResultingFrameRate.Value

T_max_total_frame = 1 / max_native_fps * 1e6
print(f"\nMax native framerate:\n  {max_native_fps:.2f} fps")
print(f"This is equal to cycles of maximum:\n  {T_max_total_frame} μs")


# The following table is true for both Overlapping and Non-Overlapping modes :
#  ________________________________________________________________________________________________
# |  Camera Model  	 |    Exposure Start Delay             |    Exposure Start Delay               |
# |                  |    (Standard exposure time mode)    |    (Ultra Short exposure time mode)   |
# |__________________|_____________________________________|_______________________________________|
# |  acA1440-220um 	 |    8-bit pixel format: 13.5 µs      |    8-bit pixel format: 29.5 µs        |
# |                  |    12-bit pixel format: 17.5 µs     |    12-bit pixel format: 35.5 µs       |
# |__________________|_____________________________________|_______________________________________|
#
# Propagation delay ( = hardware response time) with GPIO Falling Edge: <0.5 μs
#
# Total Start Delay = Exposure Start Delay, 13.5 µs (see table above)
#                       + Propagation delay (response time), 0.5 µs
#                       + Line Debouncer Time, e.g. 5 µs
#                       + Trigger Delay, e.g. 10 µs
#                   = 29.5 μs + 0.5 μs + 5 μs + 10 μs = 45 μs

T_propagation_delay = 0.5                   # in μs
T_exposure_time_offset = 13.5               # in μs

T_additional_exposure_start_delay = 0.0     # in μs
T_line_debouncing = 5.0                     # in μs

T_trigger_delays = T_additional_exposure_start_delay + T_line_debouncing
cam.TriggerDelay.Value = T_additional_exposure_start_delay
cam.LineDebouncerTime.Value = T_line_debouncing

T_exposure_start_delay = T_propagation_delay + T_exposure_time_offset


##

# Define the wanted exposure time
T_exposure = 4318                      # in µs
cam.ExposureTime.SetValue(T_exposure)

# And get the resulting timings
resulting_fps = cam.ResultingFrameRate.Value
resulting_cycle_length = 1/resulting_fps*1e6

# acA1440-220um has automatic Overlapping Image Acquisition.
# https://docs.baslerweb.com/overlapping-image-acquisition.html#non-overlapping-image-acquisition
# This means that the readout time is in effect shorter than the non-overlapping one
T_readout_non_overlapping = cam.SensorReadoutTime.Value


T_effective_readout = resulting_cycle_length - T_exposure - T_exposure_start_delay

bottom_text = f"\nCycle length..................: {resulting_cycle_length} µs\n"\
              f"Resulting framerate...........: {resulting_fps:.2f} fps\n"\  
                f"\nTimings details:\n"\
                f"    Exposure start delay......: {T_exposure_start_delay} µs\n"\
                f"    Exposure of image.........: {T_exposure} µs\n"\
                f"    Real sensor readout.......: {T_readout_non_overlapping} µs\n"\
                f"    Effective sensor readout..: {T_effective_readout} µs\n"\
                f"    Trigger Delay.............: {cam.TriggerDelay.Value} µs\n"\
                f"    Line Debouncer Time.......: {cam.LineDebouncerTime.Value} µs"

print(bottom_text)

##

def plot_timeline():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))

    ax1.axvline(x=0,
               color='k', linestyle='-', linewidth=1, alpha=0.3)
    ax1.broken_barh([(0,  # start
                     resulting_cycle_length)],  # length
                   (1.5, 0.75),  # position, thickness
                   facecolor='black', alpha=0.1, label='Total acquisition')

    ax1.broken_barh([(0,  # start
                     T_exposure_start_delay)],  # length
                   (2.5, 0.75),  # position, thickness
                   facecolor='red', alpha=0.5, label='Exposure start delay')
    ax1.axvline(x=T_exposure_start_delay,
               color='k', linestyle='dotted', linewidth=1, alpha=0.5)

    ax1.broken_barh([(T_exposure_start_delay,  # start
                     T_exposure)],  # length
                   (3.5, 0.75),  # position, thickness
                   facecolor='orange', alpha=0.35, label='Exposure')
    ax1.axvline(x=T_exposure_start_delay + T_exposure,
               color='k', linestyle='dotted', linewidth=1, alpha=0.5)

    ax1.broken_barh([(resulting_cycle_length - T_readout_non_overlapping,  # start
                     T_readout_non_overlapping)],  # length
                   (4.5, 0.75),  # position, thickness
                   facecolor='purple', alpha=0.1)

    ax1.broken_barh([(T_exposure_start_delay + T_exposure,  # start
                     T_effective_readout)],  # length
                   (4.5, 0.75),  # position, thickness
                   facecolor='purple', alpha=0.7, label='Sensor readout')

    ax1.axvline(x=T_exposure_start_delay + T_exposure + T_effective_readout,
               color='k', linestyle='dotted', linewidth=1, alpha=0.5)
    ax1.axvline(x=resulting_cycle_length,
               color='k', linestyle='-', linewidth=1, alpha=0.3)

    high_length = 0.5
    square_wave_x = np.arange(int(1.2 * resulting_cycle_length)) - 0.1*resulting_cycle_length
    square_wave_y = np.array(((square_wave_x >= 0)
                              & (square_wave_x < resulting_cycle_length*high_length)
                              | (square_wave_x > resulting_cycle_length)))

    ax1.plot(square_wave_x, square_wave_y, color='k', label='Trigger signal')

    ax1.legend()
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
              ncol=3, fancybox=True, shadow=False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set(yticklabels=[])
    ax1.tick_params(left=False)
    ax1.set_xlabel("Time (µs)")
    ax2.set_axis_off()
    ax2.text(0, 1.5, bottom_text, va='top', ha='left', fontfamily='monospace')
    plt.tight_layout()
    plt.show()

plot_timeline()

##

# Define which events we want to grab
@unique
class EventID(Enum):
    FRAMESTART = 0
    EXPOSUREEND = 1
    OVERTRIGGER = 2

nodes_mapping = {EventID.FRAMESTART: 'FrameStart',
                 EventID.EXPOSUREEND: 'ExposureEnd',
                 EventID.OVERTRIGGER: 'FrameStartOvertrigger'
                 }

# Custom camera event handler
class EventHandler(py.CameraEventHandler):
    # Only very short processing tasks should be performed by this method.
    # Otherwise, the event notification will block the processing of images.
    def __init__(self, frames=100, print=False):
        super().__init__()

        self.events_per_frame = len(EventID)

        dt = np.dtype([(nodes_mapping[e], '<u8') for e in EventID])
        self.timestamps = np.zeros(frames, dtype=dt)

        self._print = print
        self.nodes = cam.GetNodeMap()

        self.counter = 0

    def OnCameraEvent(self, camera, event_id, node):
        event = EventID(event_id)
        ts = self.nodes.GetNode(f'Event{nodes_mapping[event]}Timestamp').Value
        image_id = self.nodes.GetNode(f'Event{nodes_mapping[event]}FrameID').Value

        self.timestamps[image_id][nodes_mapping[event]] = ts

        if self._print:
            print(f"\n    {nodes_mapping[event]} event\n"
                  f"        Reported frameID.....: {image_id}\n"
                  f"        Timestamp............: {ts}")


##

h, w = cam.Height.Value, cam.Width.Value

# Custom image event handler
class TriggeredImage(py.ImageEventHandler):
    def __init__(self, frames=100, print=False):
        super().__init__()
        self.timestamps = np.zeros(frames, dtype='<u8')
        self.images = np.zeros((frames, h, w), dtype='<u1')
        self.max = frames
        self.counter = 0
        self._print = print

    def OnImageGrabbed(self, camera, grabResult):
        self.timestamps[self.counter] = grabResult.TimeStamp

        if grabResult.GrabSucceeded():
            self.images[self.counter] = grabResult.GetArray()
            if self._print:
                print(f"Done frame {self.counter+1}/{self.max}.")
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())
        self.counter += 1

##

max_images = 100

image_handler = TriggeredImage(frames=max_images, print=True)

event_handler = EventHandler(frames=max_images, print=True)


# Register the image handler
cam.RegisterImageEventHandler(image_handler,
                              py.RegistrationMode_ReplaceAll,   # Remove any existing handlers
                              py.Cleanup_None)

# Register the event handlers
for event in EventID:
    cam.RegisterCameraEventHandler(event_handler,
                                   f"Event{nodes_mapping[event]}Data",
                                   event.value,
                                   py.RegistrationMode_Append,  # Notice the 'Append'
                                   py.Cleanup_None)


##

# Check if the device supports events.
if not genicam.IsAvailable(cam.EventSelector):
    raise genicam.RuntimeException("The device doesn't support events.")

# Enable event reporting
for event in EventID:
    cam.EventSelector = nodes_mapping[event]
    cam.EventNotification = 'On'

##


# Set the trigger parameters accordingly and start the trigger on the RPi
frequency = 227         # in Hz or fps
duration = 30           # in seconds
interval, count = utils.to_ticks(frequency, duration)

trigger.start(frequency, duration)


# Start grabbing in a background loop
print(f'\nGrabbing {max_images} frames...')
cam.StartGrabbingMax(max_images)

while cam.IsGrabbing():
    # Execute the software trigger. Wait up to 1000 ms for the camera to be ready for trigger.
    # if cam.WaitForFrameTriggerReady(1000, py.TimeoutHandling_ThrowException):
    #     cam.ExecuteSoftwareTrigger()

    # Retrieve grab results and notify the camera event and image event handlers.
    grabResult = cam.RetrieveResult(5000)   # timeout 5 sec
cam.StopGrabbing()
print('Done.')

##

print(f'\nEvents recorded:\n   - ' + "\n   - ".join(event_handler.timestamps.dtype.names))

framestart_times = event_handler.timestamps['FrameStart'].astype(np.int64)
exposureend_times = event_handler.timestamps['ExposureEnd'].astype(np.int64)
overtrigger_times = event_handler.timestamps['FrameStartOvertrigger'].astype(np.int64)

frames_durations = np.diff(framestart_times)
frames_durations = frames_durations[frames_durations >= 0]
print(f"\nWanted framerate..............: {frequency} fps\n"
      f"Framerate recorded............: {1/(frames_durations.mean()/1e9):.3f} fps")

exposure_durations = exposureend_times - np.roll(framestart_times, 1)
exposure_durations = exposure_durations[exposure_durations >= 0]
print(f"\nWanted exposure time..........: {T_exposure:.2f} μs\n"
      f"Mean exposure time recorded...: {exposure_durations.mean()/1000:.3f} μs")


##

# Image acquisition precision
# We assume the camera's main oscillator has a much higher precision than the Raspberry Pi's PWM output
# For acA1440-220um, tick frequency is 1 GHz (= 1 000 000 000 ticks per second, 1 tick = 1 ns)

def plot_acquisition_delays(timestamps_data):
    wanted_delta_ms = interval * 1000
    recorded_delta_ms = (np.diff(timestamps_data)/1e6).astype(np.float64)

    frames = recorded_delta_ms.shape[0]
    time = np.cumsum(recorded_delta_ms/1000)

    mean_recorded_delta_ms = np.mean(recorded_delta_ms)

    frames_delays_ms = recorded_delta_ms - wanted_delta_ms
    mean_frames_delay = mean_recorded_delta_ms - wanted_delta_ms

    recorded_fps = 1/recorded_delta_ms * 1000
    mean_recorded_fps = 1/mean_recorded_delta_ms * 1000

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(time, recorded_fps, c='purple', alpha=0.75, label='Recorded fps')
    ax1.axhline(mean_recorded_fps, linestyle='dotted', c='purple', alpha=0.5)
    ax1.axhline(frequency, c='g', alpha=0.5, label='Wanted fps')

    ax1.set_ylabel("Acquisition rate (fps)")
    ax1.set_xlabel("Time (s)")
    ax1.ticklabel_format(useOffset=False)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.legend()

    ax2.hist(frames_delays_ms - mean_frames_delay, color='purple', alpha=0.75, bins=min([100, frames]))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_ylabel("Amount of frames")
    ax2.set_xlabel("Time difference between frames (ms)")

    plt.tight_layout()
    plt.show()
    print(f"Wanted fps:       {frequency}\n"
          f"Mean recorded fps:  {mean_recorded_fps}")

# Plot the results
plot_acquisition_delays(image_handler.timestamps)


##

# Plot some example images

def show_random_frames():
    fig = plt.figure(figsize=(6, 5))
    spec = fig.add_gridspec(3, 2)

    ax00 = fig.add_subplot(spec[0, 0])
    ax01 = fig.add_subplot(spec[0, 1])
    ax10 = fig.add_subplot(spec[1, 0])
    ax11 = fig.add_subplot(spec[1, 1])
    a = [ax00, ax01, ax10, ax11]

    ax2 = fig.add_subplot(spec[2, :])

    for i, img in enumerate(np.random.randint(0, max_images, len(a))):
        a[i].imshow(image_handler.images[img], cmap='Greys_r')
        a[i].set_axis_off()
        a[i].set_title(f"Frame {img}")

    ax2.text(0, 1, f'Framerate.......: {frequency} fps\nExposure time...: {T_exposure} µs', va='top', ha='left', fontfamily='monospace')
    ax2.set_axis_off()
    fig.suptitle(f'Random image captures')
    plt.tight_layout()

    plt.show()

show_random_frames()


##

cam.Close()
