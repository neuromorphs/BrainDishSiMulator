import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import h5py
import os 
from typing import Dict, List, Any, Tuple

import utils_events, utils_tensor 

def read_maxwell_h5(data_path: str) -> pd.DataFrame:
    """
    Extracts frame, channel, amplitude, electrode id, x, y, chip_id, date, and session from MAxwell H5 files.

    Args:
        data_path: directory of the spikeword (.npy) files

    Returns:
        data: dataframe including all of the information above about each H5 file
    """
    data = pd.DataFrame()

    for filename in tqdm(os.listdir(data_path), desc = 'Loading data...', total = len(os.listdir(data_path))):
        if filename.endswith('.h5'):

            full_filename = os.path.join(data_path, filename)

            with h5py.File(full_filename, 'r') as f:
                proc0 = f['proc0']
                spike_times = np.array(proc0['spikeTimes'])

                if len(spike_times) > 0:
                    file_data = h5_to_pd(full_filename)

                    file_data['count'] = 1
                    file_data['chip_id'] = filename.split('.')[0]
                    file_data['date'] = filename.split('.')[1]
                    
                    # Extract session if it exists in filename
                    split_filename = filename.split('.')
                    if len(split_filename) >= 3:
                        try:
                            file_data['session'] = int(split_filename[2])
                        except ValueError:
                            print(f"No valid session number found in {filename}. 'session' will be NaN for this file.")
                            file_data['session'] = np.nan
                    else:
                        file_data['session'] = np.nan

                    if data.empty:
                        data = file_data
                    else:
                        data = pd.concat([data, file_data], ignore_index=True)

    return data


def h5_to_pd(filename: str) -> pd.DataFrame:
    """Converts a H5 file to a pandas dataframe.

    Args:
        filename: The name of the file to convert

    Returns:
        pd.DataFrame: The converted dataframe
    """
    with h5py.File(filename, 'r') as file:
        maps = pd.DataFrame(np.array(file['mapping']).tolist(), columns=['channel', 'electrode', 'x', 'y'])

        ar = np.array(file['proc0']['spikeTimes'])
        spikes = pd.DataFrame(ar.tolist(), columns=['frame', 'channel', 'amplitude'])

        df = spikes.merge(maps, on='channel', how='left')
        df['frame'] -= df['frame'].iloc[0]  # Normalizing frame values

    return df


def group_by_channels(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group dataframe by 'channel' and return a dictionary."""
    channels = {}
    for i, g in df.groupby('channel'):
        channels.update({f'channel_{i}': g.reset_index(drop=True)})
    return channels


def process_channels(channels: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.Series], Dict[str, float], Dict[str, float]]:
    """Process channels dictionary and return a dictionary with 'frame' series and 'x' and 'y' coordinates for each channel."""
    processed_channels = {}
    x_coordinates = {}
    y_coordinates = {}
    for channel_id, df in channels.items():
        processed_channels[channel_id] = df['frame']
        x_coordinates[channel_id] = df['x'].iloc[0]
        y_coordinates[channel_id] = df['y'].iloc[0]
    return processed_channels, x_coordinates, y_coordinates



def extract_attributes(channels: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Extract attributes from the first channel's dataframe and return them in a dictionary."""
    first_channel_df = list(channels.values())[0]
    return {
        'chip_id': first_channel_df['chip_id'].unique()[0],
        'session': first_channel_df['session'].unique()[0],
        'date': first_channel_df['date'].unique()[0],
    }


def Get_channel_spiketimes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Extract channel spike times from a dataframe."""
    # Early return if keys are not found
    if not set(['session', 'chip_id', 'date']).issubset(dataframe.columns):
        datas = group_by_channels(dataframe)
        processed_datas, x_coordinates, y_coordinates = process_channels(datas)
        return pd.DataFrame({'spike_times': [processed_datas]})

    trials = {}
    for kk, (i, g) in enumerate(dataframe.groupby(['session', 'chip_id', 'date'])):
        trials.update({f'trials_{kk}': g.reset_index(drop=True)})

    list_datas = []
    for trial_df in trials.values():
        list_datas.append(group_by_channels(trial_df))

    new_list = [process_channels(datas)[0] for datas in list_datas]  # Only keep spike times

    spike_times = []
    chip_id = []
    session = []
    date = []
    all_x_coordinates = []
    all_y_coordinates = []

    for datas in list_datas:
        attributes = extract_attributes(datas)
        chip_id.append(attributes['chip_id'])
        session.append(attributes['session'])
        date.append(attributes['date'])
        spike_times.append(new_list.pop(0))
        _, x_coordinates, y_coordinates = process_channels(datas)
        all_x_coordinates.append(x_coordinates)
        all_y_coordinates.append(y_coordinates)

    return pd.DataFrame({
        'chip_id': chip_id,
        'session': session,
        'date': date,
        'spike_times': spike_times,
        'x_coordinates': all_x_coordinates,
        'y_coordinates': all_y_coordinates,
    })
    
    

def load_file(chip_id, chip_session, data_path) :
    # Loading data and getting the spikes
    full_data = read_maxwell_h5(data_path)

    data = Get_channel_spiketimes(full_data)
    data['session'] = data['session'].map(str)
    data['id'] = data[['chip_id', 'session', 'date']].agg('-'.join, axis=1)
    data["session"] = pd.to_numeric(data["session"])

    subset = data[(data['chip_id'] == str(chip_id)) & (data['session'] == chip_session)]

    # Loading events
    filename = utils_events. find_file(data_path, str(chip_id), str(chip_session))
    events = utils_events.load_event_txt(filename)
    
    return subset, events


def get_spiketimes(data, array_size,fs) :
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


def get_all_spikes_time_window(start_time_window, end_time_window, binsize, chip_ids, chip_sessions, data_path, array_size, fs) :
    all_sensory_spikes = []
    all_motor_spikes = []
    all_up_spikes, all_down_spikes = [], []

    for i_chipid, chip_id in enumerate(chip_ids):
        for i_chip_session, chip_session in enumerate(chip_sessions):
            print('Loading for chip {}, session {}'.format(chip_id, chip_session))
            try:
                data_subset, events = load_file(chip_id, chip_session, data_path)
            except:
                print(f'>>Could not load chip {chip_id}, session {chip_session}<<')
                print('------------------------\n')
                continue
            spiketimes = get_spiketimes(data_subset, array_size,fs)
            sensory_spikes, up1_spikes, up2_spikes, down1_spikes, down2_spikes = get_electrode_regions(data_subset, spiketimes, do_plot = False)

            all_spikes = [sensory_spikes, up1_spikes, up2_spikes, down1_spikes, down2_spikes]
            max_time_ms = max(max(max(spikes) for spikes in spike_list)*1000 for spike_list in all_spikes)

            sensory_spikes_binned = utils_tensor.spike_times_to_bins(sensory_spikes, binsize, max_time_ms, spike_tag = 'sensory')
            up1_spikes_binned = utils_tensor.spike_times_to_bins(up1_spikes, binsize, max_time_ms, spike_tag = 'up1')
            down1_spikes_binned = utils_tensor.spike_times_to_bins(down1_spikes, binsize, max_time_ms, spike_tag='down1')
            up2_spikes_binned = utils_tensor.spike_times_to_bins(up2_spikes, binsize, max_time_ms, spike_tag = 'up2')
            down2_spikes_binned = utils_tensor.spike_times_to_bins(down2_spikes, binsize, max_time_ms, spike_tag = 'down2')

            # Determine how many bins correspond to the desired time window
            start_window_bins = int(start_time_window / (binsize/1000))
            end_window_bins = int(end_time_window / (binsize/1000))
            
            # Slice the tensors
            sensory_spikes_binned = sensory_spikes_binned[:,start_window_bins:end_window_bins]
            motor_spikes_binned = torch.cat([up1_spikes_binned[:,start_window_bins:end_window_bins], 
                                            down1_spikes_binned[:,start_window_bins:end_window_bins], 
                                            up2_spikes_binned[:,start_window_bins:end_window_bins], 
                                            down2_spikes_binned[:,start_window_bins:end_window_bins]], dim = 0)
            up_spikes_binned = torch.cat([up1_spikes_binned[:,start_window_bins:end_window_bins],
                                        up2_spikes_binned[:,start_window_bins:end_window_bins]], dim = 0)
            down_spikes_binned = torch.cat([down1_spikes_binned[:,start_window_bins:end_window_bins],
                                        down2_spikes_binned[:,start_window_bins:end_window_bins]], dim = 0)

            # Add the binned spikes to their respective lists
            all_sensory_spikes.append(sensory_spikes_binned)
            all_motor_spikes.append(motor_spikes_binned)
            all_up_spikes.append(up_spikes_binned)
            all_down_spikes.append(down_spikes_binned)
            
            print('------------------------\n')

    # Concatenate all sensory and motor binned spikes into two separate tensors
    sensory_spikes_binned = torch.cat(all_sensory_spikes, dim=1)
    motor_spikes_binned = torch.cat(all_motor_spikes, dim=1)
    up_spikes_binned = torch.cat(all_up_spikes, dim=1) # these two only come in handy later
    down_spikes_binned = torch.cat(all_down_spikes, dim=1)
    
    return sensory_spikes_binned, motor_spikes_binned, up_spikes_binned, down_spikes_binned
