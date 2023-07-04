import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy import stats, optimize, interpolate
from scipy.stats import alexandergovern, f_oneway

from typing import Dict, List, Any, Tuple

def firing_stats(dataframe, bin_size_sec, total_exp_dur_sec, sampling_freq):
    """ 
    Calculate firing rate and other statistics for each channel.
        
    Args:
        dataframe: dataframe containing spike times for each recording
        bin_size_sec: length of time bins for calculating firing rate
        total_exp_dur_sec: total length of recordings in seconds
        sampling_freq: sampling frequency of the recordings in Hz

    Returns:
        dataframe: a copy of the input dataframe with additional columns for each calculated statistic
    """
    
    # Get list of spike times dictionaries for each recording
    spike_times_list = dataframe['spike_times']
    
    # For each spike times dictionary, remove spike times that exceed the total experiment duration
    for spike_times in spike_times_list:    
        for channel in spike_times.keys():
            spike_times[channel] = spike_times[channel][spike_times[channel] < total_exp_dur_sec * sampling_freq]
    
    # Calculate number of bins for experiment
    bin_size = sampling_freq*bin_size_sec
    experiment_len = int(total_exp_dur_sec * sampling_freq/bin_size) + 1 

    # Initialize lists to store results
    all_firing_counts = []
    all_firing_rates = []
    channels_mean_firing = []
    culture_mean_firing = []
    culture_max_firing = []
    channels_var_firing = []
    culture_var_firing = []
    all_channel_ISI = []
    all_channel_ISI_means = []
    culture_ISI_mean = []
    
    # For each spike times dictionary in the list
    for spike_times in spike_times_list:
        channels_firings = []
        channels_firing_rates = []
        channel_ISI = []
        channel_ISI_mean = []
        
        # For each channel in the dictionary
        for channel in spike_times.keys():
            # Initialize list of firing counts for each bin
            firing_counts = [0]*experiment_len
            
            # Increment firing count for each bin where a spike occurred
            for spike_time in spike_times[channel]/bin_size:
                firing_counts[int(spike_time)] += 1
            
            # Calculate firing rates for each bin
            firing_rates = np.array(firing_counts) / bin_size_sec
            
            # Append firing counts and rates to lists
            channels_firings.append(firing_counts)
            channels_firing_rates.append(firing_rates)
            
            # Calculate interspike intervals and mean ISI
            ISIs = np.diff(spike_times[channel]) / sampling_freq
            channel_ISI.append(ISIs)
            channel_ISI_mean.append(np.mean(ISIs))
        
        # Append results to lists
        all_channel_ISI.append(channel_ISI)
        all_channel_ISI_means.append(channel_ISI_mean)
        culture_ISI_mean.append(np.nanmean(channel_ISI_mean))
        all_firing_counts.append(channels_firings)
        all_firing_rates.append(channels_firing_rates)
        channels_mean_firing.append(np.mean(channels_firing_rates, axis=1))
        culture_mean_firing.append(np.mean(channels_firing_rates))
        channels_var_firing.append(np.var(channels_firing_rates, axis=1))
        culture_var_firing.append(np.var(channels_firing_rates))
        culture_max_firing.append(np.max(channels_firing_rates))

    # Create a copy of the input dataframe and add new columns
    dataframe_copy = dataframe.copy()
    dataframe_copy['firing_counts/spikewords'] =  all_firing_counts
    dataframe_copy['firing_rates'] =  all_firing_rates 
    dataframe_copy['channel_mean_firing_rates'] =  channels_mean_firing 
    dataframe_copy['culture_mean_firing_rates'] =  culture_mean_firing
    dataframe_copy['channel_var_firing_rates'] =  channels_var_firing 
    dataframe_copy['culture_var_firing_rates'] =  culture_var_firing 
    dataframe_copy['culture_max_firing_rates'] =  culture_max_firing 
    dataframe_copy['channel_ISI'] =  all_channel_ISI 
    dataframe_copy['channel_ISI_mean'] =  all_channel_ISI_means 
    dataframe_copy['culture_ISI_mean'] =  culture_ISI_mean 

    return dataframe_copy
