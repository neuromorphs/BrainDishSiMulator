import numpy as np
import torch 
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