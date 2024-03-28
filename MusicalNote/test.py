import queue
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

device = 0  # id of the audio device by default
window = 1000  # window for the data
downsample = 1  # how much samples to drop
channels = [1]  # a list of audio channels
interval = 30  # update interval in milliseconds for plot
sample_duration = 2  # duration of the sample window in seconds

# Create a queue to store audio data
q = queue.Queue()

# Query device info
device_info = sd.query_devices(device, 'input')
samplerate = device_info['default_samplerate']
length = int(window * samplerate / (1000 * downsample))

# Initialize plotdata
plotdata = np.zeros((length, len(channels)))
plotdata_fft = np.zeros(int(sample_duration*samplerate))

# Create figure and axis for plotting
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))
ax1.set_title("Time Domaine")
lines = ax1.plot(plotdata)
lines2 = ax2.plot(plotdata_fft)
ax2.set_title("Frequency Domain")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude")
ax2.set_xlim(27.5, 4187)

def FFT(f_time, sampling_rate):

    F_fft = np.fft.rfft(f_time, norm="forward")
    F_amp = np.abs(F_fft)*2
    F_amp[0] = F_amp[0]/2
    freqs = np.fft.rfftfreq(len(f_time), 1/sampling_rate)

    return [F_amp, freqs]


# Function to update plot with audio data
def update_plot(frame):
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    
    # Only plot the last 2 seconds of data
    plotdata_to_plot = plotdata[-int(sample_duration * samplerate):, :]
    
    for column, line in enumerate(lines):
        line.set_ydata(plotdata_to_plot[:, column])
    
    # Calculate FFT of last 2 seconds of data
    audio_data = plotdata_to_plot[:, 0]  # Taking only the first channel for FFT
    fft_amp, freqs = FFT(audio_data, samplerate)
    
    # Update FFT plot
    lines2[0].set_data(freqs, fft_amp)

    return lines + lines2  # Return both sets of lines
# Add grid to plot
ax1.set_yticks([0])
ax1.yaxis.grid(True)


# Function for audio callback
def audio_callback(indata, frames, time, status):
    global device, downsample
    q.put(indata[::downsample, [0]])

# Set up input stream
stream = sd.InputStream(device=device, channels=max(channels), samplerate=samplerate, callback=audio_callback)

# Start animation
ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)

# Start input stream and show plot with FFT on the last 2 seconds of data
with stream:
    plt.show()
