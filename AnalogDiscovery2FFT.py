from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Load the DWF library based on the OS
if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

# Constants
DWF_STATE_DONE = 2

class AnalogDiscovery2FFT:
    def __init__(self, sample_rate=20e6, buffer_size=16384, range=5):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.range = range
        self.hdwf = c_int()
        self.num_channels = 0
        self.rgd_samples = []  # List to hold data for each channel
        self.version = create_string_buffer(16)
        self.config_index = 0  # Default config index

    def initialize_device(self, config_index):
        # Get version
        dwf.FDwfGetVersion(self.version)
        print("DWF Version: " + str(self.version.value, 'utf-8'))

        # Open device
        print("Opening first device")
        dwf.FDwfDeviceConfigOpen(c_int(-1), c_int(config_index), byref(self.hdwf))

        if self.hdwf.value == 0:
            szerr = create_string_buffer(512)
            dwf.FDwfGetLastErrorMsg(szerr)
            print(szerr.value.decode('utf-8'))
            print("Failed to open device")
            quit()

        # Apply the selected configuration
        self.apply_config(config_index)

        # Get number of channels
        c_channels = c_int()
        dwf.FDwfAnalogInChannelCount(self.hdwf, byref(c_channels))
        self.num_channels = c_channels.value
        print(f"Number of channels: {self.num_channels}")

        # Initialize sample buffers for each channel
        self.rgd_samples = [(c_double * self.buffer_size)() for _ in range(self.num_channels)]

        # Set up acquisition for each channel
        for ch in range(self.num_channels):
            self.set_analog_in_channel_enable(ch, True)
            self.set_analog_in_channel_range(ch, self.range)
            self.set_analog_in_channel_filter(ch, filter_type=c_int(0))  # filterDecimate

        # Set common acquisition settings
        self.set_analog_in_frequency(self.sample_rate)
        self.set_analog_in_buffer_size(self.buffer_size)

        dbl0, dbl1, dbl2 = c_double(), c_double(), c_double()
        dwf.FDwfAnalogInChannelRangeInfo(self.hdwf, byref(dbl0), byref(dbl1), byref(dbl2))
        print("Range from "+str(dbl0.value)+" to "+str(dbl1.value)+" in "+str(dbl2.value)+" steps")

        dwf.FDwfAnalogInChannelOffsetInfo(self.hdwf, byref(dbl0), byref(dbl1), byref(dbl2))
        print("Offset from "+str(dbl0.value)+" to "+str(dbl1.value)+" in "+str(dbl2.value)+" steps")

        # Wait for offset to stabilize
        time.sleep(2)

    def apply_config(self, config_index):
        # Apply the selected configuration
        dwf.FDwfEnumConfig(c_int(0), None)
        print(f"Applied configuration index: {config_index}")

    def set_analog_in_frequency(self, frequency):
        dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(frequency))
        actual_freq = c_double()
        dwf.FDwfAnalogInFrequencyGet(self.hdwf, byref(actual_freq))
        if abs(actual_freq.value - frequency) > 1e-6:
            print(f"Warning: Frequency set to {actual_freq.value} Hz, expected {frequency} Hz")

    def set_analog_in_buffer_size(self, size):
        dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(size))
        actual_size = c_int()
        dwf.FDwfAnalogInBufferSizeGet(self.hdwf, byref(actual_size))
        if actual_size.value != size:
            print(f"Warning: Buffer size set to {actual_size.value}, expected {size}")

    def set_analog_in_channel_enable(self, channel, enable):
        dwf.FDwfAnalogInChannelEnableSet(self.hdwf, c_int(channel), c_bool(enable))
        # Check if enabled
        enabled = c_bool()
        dwf.FDwfAnalogInChannelEnableGet(self.hdwf, c_int(channel), byref(enabled))
        if enabled.value != enable:
            print(f"Warning: Channel {channel} enable state set to {enabled.value}, expected {enable}")

    def set_analog_in_channel_range(self, channel, range):
        dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(channel), c_double(range))
        actual_range = c_double()
        dwf.FDwfAnalogInChannelRangeGet(self.hdwf, c_int(channel), byref(actual_range))
        if abs(actual_range.value - range) > 1e-6:
            print(f"Warning: Channel {channel} range set to {actual_range.value} V, expected {range} V")

    def set_analog_in_channel_filter(self, channel, filter_type):
        dwf.FDwfAnalogInChannelFilterSet(self.hdwf, c_int(channel), filter_type)
        actual_filter = c_int()
        dwf.FDwfAnalogInChannelFilterGet(self.hdwf, c_int(channel), byref(actual_filter))
        if actual_filter.value != filter_type.value:
            print(f"Warning: Channel {channel} filter type set to {actual_filter.value}, expected {filter_type}")

    def acquire_data(self, num_acquisitions):
        """
        Acquire data multiple times and return the raw data for later processing.
        """
        all_samples = [[] for _ in range(self.num_channels)]

        for i in range(num_acquisitions):
            print(f"Starting acquisition {i + 1}")
            dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1))

            # Wait for acquisition to complete
            sts = c_byte()
            while True:
                dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))
                if sts.value == DWF_STATE_DONE:
                    break
                time.sleep(0.1)
            print(f"Acquisition {i + 1} done")

            # Retrieve data for each channel
            for ch in range(self.num_channels):
                dwf.FDwfAnalogInStatusData(self.hdwf, c_int(ch), self.rgd_samples[ch], self.buffer_size)
                data = np.fromiter(self.rgd_samples[ch], dtype=np.float64)
                all_samples[ch].append(data)

        return all_samples

    def perform_fft(self, samples):
        """
        Perform FFT on the acquired samples and return the average FFT result.
        """
        avg_magnitudes = []

        for ch_samples in samples:
            fft_results = np.fft.fft(np.array(ch_samples), axis=1, norm='forward')
            freqs = np.fft.fftfreq(self.buffer_size, 1 / self.sample_rate)
            avg_magnitude = np.sqrt(np.mean(np.abs(fft_results)**2, axis=0))

            # Normalize the FFT result
            normalization_factor = np.sqrt(self.buffer_size / self.sample_rate)
            avg_magnitude *= normalization_factor

            # Only keep positive frequencies
            positive_freqs = freqs[:len(freqs) // 2]
            positive_magnitudes = avg_magnitude[:len(avg_magnitude) // 2]

            avg_magnitudes.append([positive_freqs, positive_magnitudes])

        return avg_magnitudes

    def plot_time_domain(self, channel):
        """
        Plot the time-domain data for the specified channel.
        """
        # Convert ctypes array to numpy array for the specified channel
        data = np.fromiter(self.rgd_samples[channel], dtype=np.float64)

        # Create time vector
        time_vector = np.arange(len(data)) / self.sample_rate

        # Plot the time-domain data
        fig, ax = plt.subplots()
        ax.plot(time_vector, data)
        ax.set_title(f'Time Domain Data - Channel {channel}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.grid(True)
        plt.show()

    def plot_fft(self, avg_magnitudes):
        """
        Plot the averaged FFT data for all channels.
        """
        for ch, (freqs, magnitudes) in enumerate(avg_magnitudes):
            fig, ax = plt.subplots()
            ax.loglog(freqs, magnitudes)
            ax.set_title(f'Average FFT Result - Channel {ch}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude (V/√Hz)')
            ax.grid(True)
            plt.show()

    def close(self):
        # Ensure the device is closed
        dwf.FDwfDeviceCloseAll()

    def __del__(self):
        self.close()

    @staticmethod
    def list_available_configs():
        """List available configurations."""
        szerr = create_string_buffer(512)
        dwf.FDwfGetLastErrorMsg(szerr)
        if szerr[0] != b'\0':
            print(str(szerr.value))

        # Check library loading errors
        IsInUse = c_bool()
        cDevice = c_int()
        cConfig = c_int()
        cInfo = c_int()
        iDevId = c_int()
        iDevRev = c_int()
        devicename = create_string_buffer(64)
        serialnum = create_string_buffer(16)

        # Print DWF version
        version = create_string_buffer(16)
        dwf.FDwfGetVersion(version)
        print("DWF Version: " + str(version.value, 'utf-8'))

        # Enumerate and print device information
        dwf.FDwfEnum(c_int(0), byref(cDevice))
        print("Number of Devices: " + str(cDevice.value))

        for iDev in range(cDevice.value):
            dwf.FDwfEnumDeviceName(c_int(iDev), devicename)
            dwf.FDwfEnumSN(c_int(iDev), serialnum)
            dwf.FDwfEnumDeviceType(c_int(iDev), byref(iDevId), byref(iDevRev))
            print("------------------------------")
            print(f"Device {iDev} : ")
            print(f"\tName: '{devicename.value.decode('utf-8')}' {serialnum.value.decode('utf-8')}")
            print(f"\tID: {iDevId.value} rev: {iDevRev.value}")

            print("\tConfigurations:")
            dwf.FDwfEnumConfig(c_int(iDev), byref(cConfig))
            for iCfg in range(cConfig.value):
                sz = f"\t{iCfg}."
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(1), byref(cInfo))  # DECIAnalogInChannelCount
                sz += f" AnalogIn: {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(7), byref(cInfo))  # DECIAnalogInBufferSize
                sz += f" x {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(2), byref(cInfo))  # DECIAnalogOutChannelCount
                sz += f" \tAnalogOut: {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(8), byref(cInfo))  # DECIAnalogOutBufferSize
                sz += f" x {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(4), byref(cInfo))  # DECIDigitalInChannelCount
                sz += f" \tDigitalIn: {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(9), byref(cInfo))  # DECIDigitalInBufferSize
                sz += f" x {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(5), byref(cInfo))  # DECIDigitalOutChannelCount
                sz += f" \tDigitalOut: {cInfo.value}"
                dwf.FDwfEnumConfigInfo(c_int(iCfg), c_int(10), byref(cInfo))  # DECIDigitalOutBufferSize
                sz += f" x {cInfo.value}"
                print(sz)

if __name__ == "__main__":
    # List available configurations
    AnalogDiscovery2FFT.list_available_configs()

    # Prompt user to select a configuration index
    config_index = 1 #int(input("Enter the configuration index to use: "))
    num_acquisitions = 30 #int(input("Enter the number of acquisitions: "))

    # Create an instance of the class
    fft_analyzer = AnalogDiscovery2FFT()
    
    # Initialize the device with the selected configuration
    fft_analyzer.initialize_device(config_index)
    
    # Acquire data
    raw_data = fft_analyzer.acquire_data(num_acquisitions)
    
    # Perform FFT and get averaged results
    avg_magnitudes = fft_analyzer.perform_fft(raw_data)
    
    # Plot data for both channels
    for ch in range(fft_analyzer.num_channels):
        fft_analyzer.plot_time_domain(ch)
    fft_analyzer.plot_fft(avg_magnitudes)
    
    # Close the device
    fft_analyzer.close()
