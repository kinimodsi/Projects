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
sample_duration = 1  # duration of the sample window in seconds

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
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 4))
ax1.set_title("Time Domaine")
lines = ax1.plot(plotdata)
lines2 = ax2.semilogx(plotdata_fft)
ax2.set_title("Frequency Domain")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude")
ax2.set_xlim(20, 4187)
#ax2.set_ylim(0, 0.2)
min_freq = 27

note_frequencies = {
    "C8": 4186.01,
    "B7": 3951.07,
    "A#7/Bb7": 3729.31,
    "A7": 3520,
    "G#7/Ab7": 3322.44,
    "G7": 3135.96,
    "F#7/Gb7": 2959.96,
    "F7": 2793.83,
    "E7": 2637.02,
    "D#7/Eb7": 2489.02,
    "D7": 2349.32,
    "C#7/Db7": 2217.46,
    "C7": 2093,
    "B6": 1975.53,
    "A#6/Bb6": 1864.66,
    "A6": 1760,
    "G#6/Ab6": 1661.22,
    "G6": 1567.98,
    "F#6/Gb6": 1479.98,
    "F6": 1396.91,
    "E6": 1318.51,
    "D#6/Eb6": 1244.51,
    "D6": 1174.66,
    "C#6/Db6": 1108.73,
    "C6": 1046.5,
    "B5": 987.767,
    "A#5/Bb5": 932.328,
    "A5": 880,
    "G#5/Ab5": 830.609,
    "G5": 783.991,
    "F#5/Gb5": 739.989,
    "F5": 698.456,
    "E5": 659.255,
    "D#5/Eb5": 622.254,
    "D5": 587.33,
    "C#5/Db5": 554.365,
    "C5": 523.251,
    "B4": 493.883,
    "A#4/Bb4": 466.164,
    "A4": 440,
    "G#4/Ab4": 415.305,
    "G4": 391.995,
    "F#4/Gb4": 369.994,
    "F4": 349.228,
    "E4": 329.628,
    "D#4/Eb4": 311.127,
    "D4": 293.665,
    "C#4/Db4": 277.183,
    "C4": 261.626,
    "B3": 246.942,
    "A#3/Bb3": 233.082,
    "A3": 220,
    "G#3/Ab3": 207.652,
    "G3": 195.998,
    "F#3/Gb3": 184.997,
    "F3": 174.614,
    "E3": 164.814,
    "D#3/Eb3": 155.563,
    "D3": 146.832,
    "C#3/Db3": 138.591,
    "C3": 130.813,
    "B2": 123.471,
    "A#2/Bb2": 116.541,
    "A2": 110,
    "G#2/Ab2": 103.826,
    "G2": 97.9989,
    "F#2/Gb2": 92.4986,
    "F2": 87.3071,
    "E2": 82.4069,
    "D#2/Eb2": 77.7817,
    "D2": 73.4162,
    "C#2/Db2": 69.2957,
    "C2": 65.4064,
    "B1": 61.7354,
    "A#1/Bb1": 58.2705,
    "A1": 55,
    "G#1/Ab1": 51.9131,
    "G1": 48.9994,
    "F#1/Gb1": 46.2493,
    "F1": 43.6535,
    "E1": 41.2034,
    "D#1/Eb1": 38.8909,
    "D1": 36.7081,
    "C#1/Db1": 34.6478,
    "C1": 32.7032,
    "B0": 30.8677,
    "A#0/Bb0": 29.1352,
    "A0": 27.5,
}

def FFT(f_time, sampling_rate):

    F_fft = np.fft.rfft(f_time, norm="forward")
    F_amp = np.abs(F_fft)*2
    F_amp[0] = F_amp[0]/2
    freqs = np.fft.rfftfreq(len(f_time), 1/sampling_rate)
    min_freq_index = np.argmax(freqs >= min_freq)
    F_amp[:min_freq_index] = 0
    F_amp = F_amp*(10/freqs)
    return [F_amp, freqs]

def get_note_freq(frequency):
    min_diff = float('inf')
    closest_note = None
    
    for note, freq in note_frequencies.items():
        abs_diff = abs(frequency - freq)
        if abs_diff < min_diff:
            min_diff = abs_diff
            closest_note = note
    
    return closest_note

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

    # Get the current note frequency
    current_note_freq = freqs[np.argmax(fft_amp)]
    current_note = get_note_freq(current_note_freq)
    
    # Update the text showing the current note
    text.set_text(f'Current Note: {current_note}, Frequency: {current_note_freq:.2f} Hz')
    
    return lines + lines2 + [text]  # Return both sets of lines and the text object

# Add text showing the current note
text = ax2.text(0, 0.9, '', transform=ax2.transAxes)

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
