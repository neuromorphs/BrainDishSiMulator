import numpy as np
import torch 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import utils_data, utils_spikes, utils_events


def load_file(chip_id, chip_session, data_path) :
    # Loading data and getting the spikes
    full_data = utils_data.read_maxwell_h5(data_path)

    data = utils_data.Get_channel_spiketimes(full_data)
    data['session'] = data['session'].map(str)
    data['id'] = data[['chip_id', 'session', 'date']].agg('-'.join, axis=1)
    data["session"] = pd.to_numeric(data["session"])

    subset = data[(data['chip_id'] == str(chip_id)) & (data['session'] == chip_session)]

    # Loading events
    filename = utils_events. find_file(data_path, str(chip_id), str(chip_session))
    events = utils_events.load_event_txt(filename)
    
    return subset, events


def get_spiketimes(data, array_size) :
    # Extracting the spike times into a list and converting to seconds
    spiketimes_list = []
    for i in range(array_size) :
        spiketimes = data['spike_times'].iloc[0]['channel_'+str(i)]
        spiketimes_list.append(spiketimes.to_numpy()/fs)
        
    return np.asarray(spiketimes_list, dtype=object)


def get_electrode_regions(data, spiketimes, do_plot=False):
    # Extracting the electrode mapping 
    xs = data['x_coordinates'].iloc[0]
    ys = data['y_coordinates'].iloc[0]

    # Convert electrode names to channel indices
    chan_idx = [int(x.split('_')[1]) for x in xs.keys()]

    # Get x and y coordinates of each channel
    chan_xs = np.asarray(list(xs.values()))
    chan_ys = np.asarray(list(ys.values()))

    # Create array with channel index, x and y coordinates
    coo_array = np.vstack((chan_idx, chan_xs, chan_ys)).T

    # Identify rows with missing coordinates
    rows_with_nan = np.isnan(coo_array).any(axis=1)

    # Filter out channels with missing coordinates
    coo_array = coo_array[~rows_with_nan]
    spiketimes = spiketimes[~rows_with_nan]

    # Define the halfway point on the y-axis
    y_halfway = np.max(coo_array[:,2])/2

    # Split channels into sensory and motor based on y-coordinate
    sensory_idx = coo_array[:, 2] <= y_halfway
    motor_idx = coo_array[:, 2] > y_halfway

    # Get coordinates and spike times for sensory and motor channels
    sensory_coo, sensory_spikes = coo_array[sensory_idx], spiketimes[sensory_idx]
    motor_coo, motor_spikes = coo_array[motor_idx], spiketimes[motor_idx]

    # Split channels into left and right based on x-coordinate
    x_halfway_motor = np.max(motor_coo[:, 1])/2
    left_idx, right_idx = motor_coo[:, 1] <= x_halfway_motor, motor_coo[:, 1] > x_halfway_motor
    left_coo, left_spikes = motor_coo[left_idx], motor_spikes[left_idx]
    right_coo, right_spikes = motor_coo[right_idx], motor_spikes[right_idx]

    # Split left region into up and down based on x-coordinate
    x_halfway_left = (np.min(left_coo[:, 1]) + np.max(left_coo[:, 1]))/2
    up1_idx, down1_idx = left_coo[:, 1] <= x_halfway_left, left_coo[:, 1] > x_halfway_left
    up1_coo, up1_spikes = left_coo[up1_idx], left_spikes[up1_idx]
    down1_coo, down1_spikes = left_coo[down1_idx], left_spikes[down1_idx]

    # Split right region into up and down based on x-coordinate
    x_halfway_right = (np.min(right_coo[:, 1]) + np.max(right_coo[:, 1]))/2
    up2_idx, down2_idx = right_coo[:, 1] <= x_halfway_right, right_coo[:, 1] > x_halfway_right
    up2_coo, up2_spikes = right_coo[up2_idx], right_spikes[up2_idx]
    down2_coo, down2_spikes = right_coo[down2_idx], right_spikes[down2_idx]
    
    # Optionally plot the electrode locations for sanity check
    if do_plot:
        fig, ax = plt.subplots(figsize = (5,5))
        map_up = plt.cm.gray(np.linspace(.4, .8, 2))
        ax.scatter(up2_coo[:,1]  , up2_coo[:,2], color = map_up[0], label = 'up2')
        ax.scatter(up1_coo[:,1]  , up1_coo[:,2], color = map_up[1], label = 'up1')
        # same but for down coo 
        map_down = plt.cm.magma(np.linspace(.2, .6, 2))
        ax.scatter(down2_coo[:,1]  , down2_coo[:,2], color = map_down[0], label = 'down2')
        ax.scatter(down1_coo[:,1]  , down1_coo[:,2], color = map_down[1], label = 'down1')
        ax.scatter(sensory_coo[:,1]  , sensory_coo[:,2], color = 'b', label = 'sensory')
        plt.legend()
        plt.title('Sanity check of the extracted electrode mapping')
        plt.show()

    return sensory_spikes, up1_spikes, up2_spikes, down1_spikes, down2_spikes


def spike_times_to_bins(spike_times_list, bin_width_ms, max_time_ms, spike_tag):
    bin_edges = np.arange(0, max_time_ms + bin_width_ms, bin_width_ms)
    binned_spikes = []
    for spikes in tqdm(spike_times_list, 'Binning %s channels' % spike_tag):
        # Convert spike times to milliseconds and compute histogram
        spikes_ms = spikes * 1000
        counts, _ = np.histogram(spikes_ms, bins=bin_edges)
        binned_spikes.append(counts)
    # Convert to tensor
    return torch.tensor(binned_spikes)


def check_binary(spike_tensor, name) :
    if torch.max(spike_tensor) > 1:
        raise ValueError('The tensor {} is not binary'.format(name))
    else:
        return True