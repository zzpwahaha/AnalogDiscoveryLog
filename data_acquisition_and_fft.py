import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import time
from AnalogDiscovery2FFT import AnalogDiscovery2FFT

# Directory to save data files
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def get_today_filename():
    """Get the filename for today's data file."""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(DATA_DIR, f"data_{today}.pickle")

def save_data(data, filename):
    """Save data to a pickle file with the given filename."""
    with open(filename, 'ab') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def acquire_and_save_data(fft_analyzer, num_acquisitions, interval, save_raw_data):
    """Acquire data, perform FFT, and save results."""
    filename = get_today_filename()

    # Prepare data to be saved
    data = {
        'timestamps': {0: [], 1: []},
        'fft_results': {0: [], 1: []},
        'sample_rate': fft_analyzer.sample_rate,
        'buffer_size': fft_analyzer.buffer_size,
        'time_axis': np.arange(fft_analyzer.buffer_size) / fft_analyzer.sample_rate,
        'frequency_axis': np.fft.fftfreq(fft_analyzer.buffer_size, 1 / fft_analyzer.sample_rate)
    }
    if save_raw_data:
        data['raw_data'] = {0: [], 1: []}

    # Loop to acquire multiple sets of data
    raw_datas = fft_analyzer.acquire_data(num_acquisitions)
    avg_magnitudes = fft_analyzer.perform_fft(raw_datas)
    for ch, (freqs, magnitudes) in enumerate(avg_magnitudes):
        data['timestamps'][ch].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if save_raw_data:
            data['raw_data'][ch].append(raw_datas[ch])
        data['fft_results'][ch].append({
            'frequencies': freqs,
            'magnitudes': magnitudes
        })


    # Save data to file
    save_data(data, filename)

    # Wait for the specified interval before acquiring data again
    time.sleep(interval)

def main():
    # Number of acquisitions per cycle and interval between acquisitions
    num_acquisitions = 10  # Number of acquisitions to perform in one cycle
    interval = 10  # Interval in seconds (2 minutes)
    save_raw_data = False  # Set to True to save raw data, False otherwise

    # Create an instance of the AnalogDiscovery2FFT class
    fft_analyzer = AnalogDiscovery2FFT()

    # Initialize the device with a selected configuration index
    config_index = 1  # Change as needed
    fft_analyzer.initialize_device(config_index)

    # Continuous data acquisition and saving
    while True:
        acquire_and_save_data(fft_analyzer, num_acquisitions, interval, save_raw_data)

if __name__ == "__main__":
    main()
