
import numpy as np
import pandas as pd
import os
import sys
import h5py
import pickle




def read_Maxwell_H5(data_path):
    
    """Extacts frame,channel,amplitude,electrode id, x, y, chip_id, date, and session from MAxwell H5 files

        Args:

             data_path: directory of the spikeword (.npy) files
        
        Output:
             data: dataframe including all of the information above about each H5 file
            
        
        
    """
    dir = data_path
    data = pd.DataFrame()

    for f in os.listdir(dir):

        if '.h5' in f:

            print(f)

            fileName = dir+f

            check = h5py.File(fileName, 'r')

            proc0 = check['proc0']

            ar = np.array(proc0['spikeTimes'])

            if len(ar) > 0:

                filedata = h5_to_pd(fileName)

#                 filedata['tag'] = grabTag(f)

                filedata['count'] = 1

                filedata['chip_id'] = f.split('.')[0]

                filedata['date'] = f.split('.')[1]

                filedata['session'] = f.split('.')[2]

                if data.shape == (0, 0):

                    data = filedata

                else:

                    data = data.append(filedata, ignore_index=True, sort = False)
    return data




def h5_to_pd(filename):

    f = h5py.File(filename, 'r')

    maps = np.array(f['mapping'])

    maps=pd.DataFrame(maps.tolist())

    #print(len(maps))

    maps.columns=['channel', 'electrode', 'x', 'y']

    proc0 = f['proc0']

    #print(len(proc0))

    ar = np.array(proc0['spikeTimes'])

    #print(len(ar))

    spikes=pd.DataFrame(ar.tolist())

    #print(len(spikes))

    spikes.columns=['frame', 'channel', 'amplitude']

    df = spikes.merge(maps, on = 'channel', how = 'left')

    df['frame'] = df['frame'] - df['frame'].iloc[0]

    return df


def grabTag(name):
    name = name.rsplit('.', 1)[0]
    name = name + '.spikes.bin'
    print(name)
    chip = New_tags_filenames[name]
    tagName = chip.get('tag')
    return tagName