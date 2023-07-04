
import numpy as np
import pandas as pd
import os
import sys
import h5py
import pickle

def Get_channel_spiketimes(dataframe):
    
    trials = {}
    kk=0
    
    if (  ('session' in  dataframe.keys() ) & ('chip_id' in dataframe.keys()) & ('date' in dataframe.keys()) ):
        for i, g in dataframe.groupby(['session','chip_id','date']):
            #print 'data_' + str(i)
            #print g
            trials.update({'trials_' + str(kk) : g.reset_index(drop=True)})
            kk= kk+1
      
        
        num_trials = len(trials)
        list_datas = list()
        for j in range(num_trials):
            datas = {}
            for i, g in trials['trials_' + str(j)].groupby('channel'):
                #print 'data_' + str(i)
                #print g        
                datas.update({'channel_' + str(i) : g.reset_index(drop=True)})
            list_datas.append(datas)

        new_list = list_datas
        for i in range(len(list_datas)):
            for channel_id in list_datas[i].keys():
                   new_list[i][channel_id]=new_list[i][channel_id]['frame']

        list_datas = list()
        for j in range(num_trials):
            datas = {}
            for i, g in trials['trials_' + str(j)].groupby('channel'):
                #print 'data_' + str(i)
                #print g        
                datas.update({'channel_' + str(i) : g.reset_index(drop=True)})
            list_datas.append(datas)


        df_all = pd.DataFrame()
        chip_id = []
        date = []
        session = []
        spike_times = []
        tag = []

        for i in range (len(list_datas)):
            list_keys = list(list_datas[i].keys())
            chip_id.append(list_datas[i][list_keys[0]]['chip_id'].unique()[0])
            session.append(list_datas[i][list_keys[0]]['session'].unique().astype(int)[0])
            date.append(list_datas[i][list_keys[0]]['date'].unique()[0])  
    #         tag.append(list_datas[i]['channel_0']['tag'].unique()[0])
            spike_times.append(new_list[i])


        df_all['chip_id']= chip_id
        df_all['session'] = session
        df_all['date'] = date
    #     df_all['tag'] = tag
        df_all['spike_times'] = spike_times
    
    
    else:
        list_datas = list()
        datas = {}
        for i, g in dataframe.groupby('channel'):
            #print 'data_' + str(i)
            #print g        
            datas.update({'channel_' + str(i) : g.reset_index(drop=True)})
        list_datas.append(datas)

        new_list = list_datas
        for i in range(len(list_datas)):
            for channel_id in list_datas[i].keys():
                   new_list[i][channel_id]=new_list[i][channel_id]['frame']

        list_datas = list()

        datas = {}
        for i, g in dataframe.groupby('channel'):
            #print 'data_' + str(i)
            #print g        
            datas.update({'channel_' + str(i) : g.reset_index(drop=True)})
        list_datas.append(datas)


        df_all = pd.DataFrame()
        spike_times = []


        for i in range (len(list_datas)):
            spike_times.append(new_list[i])


        df_all['spike_times'] = spike_times
 
        
    
    return df_all


           