import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import utils_data, utils_spikes, utils_events

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
    
    
def events_to_bins(events, event_types, bin_width, max_time_ms):
    num_bins = int(np.ceil(max_time_ms / bin_width))
    event_bins = np.zeros((len(event_types), num_bins))
    event_index = 0  # pointer to track current event
    
    # sort events based on 'norm_timestamp'
    events = sorted(events, key=lambda e: e['norm_timestamp'])

    for bin_index in range(num_bins):
        current_bin_time = bin_width * bin_index
        next_bin_time = bin_width * (bin_index + 1)

        # move the event pointer until an event is in the future
        while event_index < len(events) and events[event_index]['norm_timestamp'] < next_bin_time:
            curr_event = events[event_index]['event']
            curr_event_type_index = event_types.index(curr_event)
            event_bins[curr_event_type_index, bin_index] = 1
            event_index += 1

    return event_bins


def transform_data(labels, spikes, len_trial) :
    all_onsets = []
    for irowlabel in range(labels.shape[0]) :
        onsets = np.where(labels[irowlabel,:]==1)[0]
        all_onsets.append([(x, irowlabel) for x in onsets])
        
    sorted_onsets = sorted([x for sublist in all_onsets for x in sublist], key=lambda x: x[0])

    transformed_data = np.zeros((len_trial, len(sorted_onsets), spikes.shape[0]))
    transformed_labels = np.zeros((len(sorted_onsets)))
    for ionset, onset in enumerate(sorted_onsets) :
        onset_time = onset[0]
        onset_label = onset[1]
        onset_spikes = spikes[:,onset_time:onset_time+len_trial].swapaxes(0,1)
        transformed_data[:,ionset,:] = onset_spikes
        transformed_labels[ionset] = onset_label
        
    return transformed_data, transformed_labels


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[:, idx, :], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)