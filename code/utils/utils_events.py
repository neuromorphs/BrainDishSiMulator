import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import h5py
import os 
from typing import Dict, List, Any, Tuple
import fnmatch

def find_file(directory, chip_id, session_id):
    # Define the filename pattern
    pattern = f'{chip_id}.*.{session_id}.events.txt'

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # If this file matches the pattern, return it
        if fnmatch.fnmatch(filename, pattern):
            return filename

    # If no matching file was found, return None
    return None

def load_event_txt(filename):
    event_list = []
    timestamps = []
    # Open and read the file
    with open('../data/cortical_labs_data/%s' % filename, 'r') as file:
        for line in file:
            # Skip all 'info' lines except the stimulation mode
            if line.startswith('info:'):
                if 'stimulation mode' in line:
                    _, _, stim_mode = line.partition('stimulation mode:')
                    print(f'Stimulation mode: {stim_mode.strip()}')
                continue

            # Process event lines
            timestamp, _, event = line.partition(':')
            timestamps.append(timestamp)
            event_list.append({'norm_timestamp': int(timestamp) - int(timestamps[0]), 'event': event.strip()})
            
    return event_list