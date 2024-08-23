import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Directory to save data files
DATA_DIR = 'data'

def get_today_filename():
    """Get the filename for today's data file."""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(DATA_DIR, f"data_{today}.pickle")

def load_data(filename):
    """Load data from a pickle file."""
    data = []
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            while True:
                try:
                    data.append(pickle.load(file))
                except EOFError:
                    break
    else:
        print(f"No data file found at {filename}")
    return data

def process_data(data):
    """Process and extract data for analysis."""
    timestamps = {0: [], 1: []}
    raw_data = {0: [], 1: []}
    fft_results = {0: [], 1: []}
    sample_rate = None
    buffer_size = None
    time_axis = None
    frequency_axis = None

    for entry in data:
        if 'sample_rate' in entry:
            sample_rate = entry['sample_rate']
            buffer_size = entry['buffer_size']
            time_axis = entry['time_axis']
            frequency_axis = entry['frequency_axis']
        if 'timestamps' in entry:
            for ch in range(2):
                timestamps[ch].extend(entry['timestamps'][ch])
        if 'raw_data' in entry:
            for ch in range(2):
                raw_data[ch].extend(entry['raw_data'][ch])
        if 'fft_results' in entry:
            for ch in range(2):
                fft_results[ch].extend(entry['fft_results'][ch])

    return timestamps, raw_data, fft_results, sample_rate, buffer_size, time_axis, frequency_axis

def plot_data(timestamps, raw_data, fft_results, sample_rate, buffer_size, time_axis, frequency_axis):
    """Plot the raw and FFT data."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Acquired Data', fontsize=16)

    # Plot raw data
    for ch in range(2):
        axs[0, 0].plot(np.arange(len(raw_data[ch])) / sample_rate, raw_data[ch], label=f'Channel {ch}')
    axs[0, 0].set_title('Raw Data')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()

    # Plot FFT results
    for ch in range(2):
        avg_frequencies = np.mean([result['frequencies'] for result in fft_results[ch]], axis=0)
        avg_magnitudes = np.mean([result['magnitudes'] for result in fft_results[ch]], axis=0)
        axs[ch, 1].loglog(avg_frequencies, avg_magnitudes, label=f'Channel {ch}')
    axs[0, 1].set_title('Average FFT Results')
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    # Get filename for today's data
    filename = get_today_filename()
    data = load_data(filename)

    if not data:
        print("No data to process.")
        return

    # Process the loaded data
    timestamps, raw_data, fft_results, sample_rate, buffer_size, time_axis, frequency_axis = process_data(data)

    # Plot the data
    plot_data(timestamps, raw_data, fft_results, sample_rate, buffer_size, time_axis, frequency_axis)

if __name__ == "__main__":
    main()
