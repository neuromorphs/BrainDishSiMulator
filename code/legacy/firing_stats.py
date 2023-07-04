from scipy import stats, optimize, interpolate
from scipy.stats import alexandergovern
from scipy.stats import f_oneway
import pandas as pd
import numpy as np

def firing_stats(dataframe,bin_size_sec,total_exp_dur_sec,sampling_freq):
    
    """ Calculating the firing rate of each channel.
        
        Args:

             dataframe: dataframe containing the 'spike_times' field for each recording. 'spike_times' is a dictionary which 
             stores each channel id ('channel_i') and the vector of the corresponding spike times. e.g. 
             {'channel_0': [2398, 8263, 19925],'channel_3': [3487,6009,10001,12321]}.
             
             bin_size_sec: A float specifying the length of the time bins in seconds for calculating the firing rate.
             
             total_exp_dur_sec: Integer specifying the entire length of recordings in seconds.
             
             sampling_freq: Integer specifying the sampling frequency of the recordings in Hz.
             
        
        Output:
             data_frame_out: dataframe containing:
             1. 'firing_counts/spikewords': An N x M array for each recording containing the calculated firing count in each time
             bin of specified length (bin_size_sec) for each channel.
             N: number of channels, M: number of time bins with lenght 'bin_size_sec'.
 
             2. 'firing_rates': An N x M array for each recording containing the calculated firing rate in each time bin of 
             specified length (bin_size_sec) for each channel.
             
             3. 'channel_mean_firing_rates': Average firing rate associated with each of the N channels in every recording.
             
             4. 'culture_mean_firing_rates': Average firing rate of all the channels in each recording.
             
             5. 'channel_var_firing_rates': Variance of firing rates associated with each of the N channels in every recording.
             
             6. 'culture_var_firing_rates': Variance of firing rates in all the channels in each recording.
             
             7. 'culture_max_firing_rates': Maximum firing rate among all channels in the entire duration of the recording.
             
             8. 'channel_ISI': An array of shape (N,) containing the interspike intervals calculated for each of the N channels.
             
             9. 'channel_ISI_mean': An array of length N containing the average interspike interval for each channel.
             
             10. 'culture_ISI_mean': Average interspike interval of all the channels in each culture recording. 
             
        
        
    """
    
    new_list = dataframe['spike_times']
    for i in range(len(new_list)):    
        for channel in new_list[i].keys():
            new_list[i][channel] = new_list[i][channel] [new_list[i][channel] < total_exp_dur_sec * sampling_freq]
            

    bin_size = sampling_freq*bin_size_sec
    experiment_len = int(total_exp_dur_sec * sampling_freq/bin_size)+1 # e.g. 6000 = 10 min * 60s * 20kHz / bin_size

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
    for i in range(len(new_list)):
        channels_firings = []
        channels_firing_rates = []
        channel_ISI = []
        channel_ISI_mean = []
        for channel in new_list[i].keys():
            firing_temp = [0]*experiment_len
            for item in new_list[i][channel]/bin_size:
                firing_temp[int(item)]+=1
            channels_firings.append(firing_temp)
            channels_firing_rates.append(np.array(firing_temp)/bin_size_sec)
            
            temp = [t - s for s, t in zip(new_list[i][channel], new_list[i][channel][1:])]
            channel_ISI.append(np.array(temp)/sampling_freq)
            channel_ISI_mean.append(np.mean(np.array(temp)/sampling_freq))

        all_channel_ISI.append(channel_ISI) 
        all_channel_ISI_means.append(channel_ISI_mean)
        culture_ISI_mean.append(np.nanmean(np.array(channel_ISI_mean)))
        all_firing_counts.append(channels_firings)
        all_firing_rates.append(channels_firing_rates)
        channels_mean_firing.append(np.mean(channels_firing_rates,axis =1))
        culture_mean_firing.append(np.mean(channels_firing_rates))
        channels_var_firing.append(np.var(channels_firing_rates,axis =1))
        culture_var_firing.append(np.var(channels_firing_rates)) 
        culture_max_firing.append(np.max(channels_firing_rates))



    data_frame_out = dataframe.copy()
    data_frame_out['firing_counts/spikewords'] =  all_firing_counts
    data_frame_out['firing_rates'] =  all_firing_rates 
    data_frame_out['channel_mean_firing_rates'] =  channels_mean_firing 
    data_frame_out['culture_mean_firing_rates'] =  culture_mean_firing
    data_frame_out['channel_var_firing_rates'] =  channels_var_firing 
    data_frame_out['culture_var_firing_rates'] =  culture_var_firing 
    data_frame_out['culture_max_firing_rates'] =  culture_max_firing 
    
    data_frame_out['channel_ISI'] =  all_channel_ISI 
    data_frame_out['channel_ISI_mean'] =  all_channel_ISI_means 
    data_frame_out['culture_ISI_mean'] =  culture_ISI_mean 


    
    return data_frame_out





