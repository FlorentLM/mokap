import pypylon.pylon as py
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
from mokap.hardware import SSHTrigger

##

# Open the connection to the camera
tlf = py.TlFactory.GetInstance()
cam = py.InstantCamera(tlf.CreateFirstDevice())
cam.Open()


trigger = SSHTrigger()

##

# Allow sampling the I/O at very high frame rate (i.e. 1000 fps) by minimizing the image ROI:
# https://docs.baslerweb.com/image-roi.html
# The smallest ROI is 4x1 pixels on our acA1440-220um cameras.
# Pixel binning can also be used in addition (https://docs.baslerweb.com/binning.html),
# but it should not be necessary here.
cam.Height = cam.Height.Min     # 1 px
cam.Width = cam.Width.Min       # 4 px
cam.ExposureTime = cam.ExposureTime.Min

# Sample for 1 sec at 1 kHz
sampling_time = 1               # in seconds
sampling_rate = 1000            # in Hz

cam.AcquisitionFrameRateEnable = True
cam.AcquisitionFrameRate = sampling_rate

nb_samples = sampling_rate * sampling_time

##

# Captured image is sent in a 'chunk' of data. Additional 'data chunks' can be appended following the image.
# https://docs.baslerweb.com/data-chunks

# List available data chunks for the current camera
# cam.ChunkSelector.Symbolics

# Enable the chunk that samples all I/O lines on every FrameStart
cam.ChunkModeActive = True
cam.ChunkSelector = "LineStatusAll"
cam.ChunkEnable = True

##

io_data = np.empty(nb_samples,
                   dtype={
                       'bitval': ('u1', 0),   # Camera returns LineStatusAll as '<u8' but this is overkill, 'u1' is fine
                       'timestamp': ('<u8', 1)  # For the timestamp, uint64 precision is needed though
                   })


# Start the trigger on the RPi for 30 seconds at 10 pulses per second
trigger.start(10, 30)

time.sleep(0.5)

i = 0
cam.StartGrabbingMax(nb_samples)
while cam.IsGrabbing():
    with cam.RetrieveResult(nb_samples) as res:
        timestamp_chunk = res.TimeStamp
        linestatus_chunk = res.ChunkLineStatusAll.Value

        io_data[i] = (linestatus_chunk, timestamp_chunk)
        i += 1
cam.StopGrabbing()

##

def plot_logic_analyzer(io_data):

    x_vals = io_data['timestamp']

    y_vals = np.unpackbits(io_data['bitval']).reshape(-1, 8)    # Unpack to N samples x 8 bits
    y_vals = np.flip(y_vals[:, -4:])                            # The 4 small bits encode the 4 GPIO lines

    # Make the timestamps relative to the first one
    x_vals -= x_vals[0]

    x_vals_sec = x_vals / 1e9
    mean_acq_rate = np.diff(x_vals_sec).mean()
    time_total = mean_acq_rate*len(x_vals_sec)      # in secs

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # for each bit plot the graph
    yticklabels = []
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))

    bottom_text = ''
    for bit in range(4):

        name = f'Line {bit+1}'

        thisline_vals = y_vals[:, bit]

        low = []
        high = []
        for current, pulse in itertools.groupby(thisline_vals):
            if current == True:
                high.append(sum(1 for _ in pulse))
            else:
                low.append(sum(1 for _ in pulse))
        nb_cycles = len(high)

        if nb_cycles > 1:  # 0 or 1 means either always low or always high
            high_length = np.max(high)
            low_length = np.max(low)
            mean_H_length = high_length * mean_acq_rate   # in seconds
            mean_L_length = low_length * mean_acq_rate    # in seconds
            bottom_text += f'\n{name}:'\
                    f'\n    {nb_cycles} cycles → {nb_cycles/time_total:,.3f} fps'\
                    f'\n       ⌁ High width: {mean_H_length:,.3f} s = {mean_H_length * 1000:,.3f} ms = {mean_H_length * 1e6:,.3f} µs (avg.)'\
                    f'\n       ⌁ Low width: {mean_L_length:,.3f} s = {mean_L_length * 1000:,.3f} ms = {mean_L_length * 1e6:,.3f} µs (avg.)'

        yticklabels.append(name)
        c = next(colors)
        plt.plot(x_vals_sec, thisline_vals/2 + bit + 0.25, c=c, label=name)
        plt.plot(x_vals_sec, thisline_vals*0 + bit + 0.25, c='k', linestyle='dotted', alpha=0.3)

        plt.plot((-0.025, -0.025), (bit + 0.25, bit + 0.75), linewidth=1, color='k')  # volts vertical axes
        plt.plot((-0.025, -0.035), (bit + 0.25, bit + 0.25), linewidth=1, color='k')   # min volt tick
        plt.plot((-0.025, -0.035), (bit + 0.75, bit + 0.75), linewidth=1, color='k')   # max volt tick
        plt.text(-0.05, bit + 0.15, '0.0', ha='right')
        plt.text(-0.05, bit + 0.65, '5.0', ha='right')

    plt.xlim([-0.125, 1.05])
    plt.ylim([-0.2, 6])

    ax1.set_ylabel("Amplitude (V)")
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_bounds(0, 1)
    ax1.spines['bottom'].set_bounds(0.0, 1.0)

    locs, labels = plt.yticks()
    ax2.set_yticks(np.arange(4) + 0.5, yticklabels, rotation=-45)
    ax2.set_ylabel("I/O Lines")
    ax1.set_xlabel("Time (s)")
    plt.box(False)

    plt.text(0.1, -1.5, f'R = {1/mean_acq_rate:,.3f} Hz\n{bottom_text}', va='top', ha='left', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# Plot the results
plot_logic_analyzer(io_data)

##

cam.Close()
