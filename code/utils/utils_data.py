import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import h5py
import os 
from typing import Dict, List, Any, Tuple

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