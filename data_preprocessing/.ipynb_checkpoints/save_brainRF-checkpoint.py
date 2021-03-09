import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from glob import glob

    
col_centers = sio.loadmat('../col_centers_112px.mat')
row_centers= sio.loadmat('../row_centers_112px.mat')

def get_RF(brain_region, size, col_centers, row_centers):

    V1_col_nan= col_centers[brain_region]
    V1_col = V1_col_nan[~np.isnan(V1_col_nan)]

    V1_row_nan = row_centers[brain_region]
    V1_row = V1_row_nan[~np.isnan(V1_row_nan)]

    centers = np.stack((V1_col , V1_row))
    
    unitA = 1/size
    channels = np.zeros((len(V1_col),size,size))

    for elect_i, coord in enumerate(centers.transpose(1,0)):
        channels[elect_i, int((coord[0] + 0.5 ) / unitA), int((coord[1] + 0.5 ) / unitA)] = 1

    return channels




def save_RFs_and_signals(brain_region, set_t, col_centers, row_centers):
    '''
    
    Please specify brain region as a strng:
    
    - V1
    - V3
    
    '''
    

    roimask = sio.loadmat('roimask.mat')
    
    roi_dorsal = roimask[f'{brain_region}d'].astype('bool')
    roi_ventral = roimask[f'{brain_region}v'].astype('bool')

    v1dorsalRF = get_RF(f'{brain_region}d', 96, col_centers, row_centers).astype('float32')
    v1ventralRF = get_RF(f'{brain_region}v', 96, col_centers, row_centers).astype('float32')
        
    dir_list = glob(f'{set_t[:-3]}_brain_signals/{set_t[:-3]}_*.npy')
    print(f'lendir:{len(dir_list)}')
    for run_nr in range(1, len(dir_list)+ 1):
       
        allbrain_signal = np.load(f'{set_t[:-3]}_brain_signals/{set_t[:-3]}_brain_signals_run{run_nr}.npy')
        if set_t == 'training':
            allbrain_signal_z = stats.zscore(allbrain_signal, 0)
            allbrain_signal_z[np.isnan(allbrain_signal_z)] = 0
            allbrain_signal = allbrain_signal_z
            
        ventral_signals_nan = allbrain_signal[:,roi_ventral]
        ventral_signals = ventral_signals_nan[:,~np.isnan(col_centers[f'{brain_region}v'][0])]

        dorsal_signal_nan = allbrain_signal[:,roi_dorsal]
        dorsal_signals = dorsal_signal_nan[:,~np.isnan(col_centers[f'{brain_region}d'][0])]
        print(f'run:{run_nr}')
        os.makedirs(f'{set_t}_signalsRF/{brain_region}/', exist_ok=True)
        np.save(f'{set_t}_signalsRF/{brain_region}/dorsal_{run_nr}_{set_t[:-3]}.npy', dorsal_signals)
        np.save(f'{set_t}_signalsRF/{brain_region}/ventral_{run_nr}_{set_t[:-3]}.npy', ventral_signals)

    np.save(f'RF_locations/RFlocs_{brain_region}_dorsal.npy', v1dorsalRF)
    np.save(f'RF_locations/RFlocs_{brain_region}_ventral.npy', v1ventralRF)

    
    
    
def save_RFs_and_signals_MTFFA(brain_region, set_t, col_centers, row_centers):
    
    '''
    
    For data without ventral / dorsal (MT & FFA)
    
    
    '''
    

    roimask = sio.loadmat('../roimask.mat')
    roi = roimask[f'{brain_region}'].astype('bool')
    
    
    ROI_RF = get_RF(f'{brain_region}', 96, col_centers, row_centers).astype('float32')
        
    dir_list = glob(f'../{set_t[:-3]}_brain_signals/{set_t[:-3]}_*.npy')
    print(f'lendir:{len(dir_list)}')
    for run_nr in range(1, len(dir_list)+ 1):
       
        allbrain_signal = np.load(f'../{set_t[:-3]}_brain_signals/{set_t[:-3]}_brain_signals_run{run_nr}.npy')
        if set_t == 'training':
            allbrain_signal_z = stats.zscore(allbrain_signal, 0)
            allbrain_signal_z[np.isnan(allbrain_signal_z)] = 0
            allbrain_signal = allbrain_signal_z
            
        signals_nan = allbrain_signal[:,roi]
        signals = signals_nan[:,~np.isnan(col_centers[f'{brain_region}'][0])]
        
        print(f'run:{run_nr}')
        os.makedirs(f'../{set_t}_signalsRF/{brain_region}/', exist_ok=True)
        np.save(f'../{set_t}_signalsRF/{brain_region}/{run_nr}_{set_t[:-3]}.npy', signals)

    np.save(f'../RF_locations/RFlocs_{brain_region}.npy', ROI_RF)
    

# save_RFs_and_signals('V1', 'testing')
# save_RFs_and_signals('V1', 'training')
# save_RFs_and_signals('V3', 'testing', col_centers, row_centers)
# save_RFs_and_signals('V3', 'training', col_centers, row_centers)
# save_RFs_and_signals('V2', 'testing', col_centers, row_centers)
# save_RFs_and_signals('V2', 'training', col_centers, row_centers)
# save_RFs_and_signals_MTFFA('MT', 'testing', col_centers, row_centers)
save_RFs_and_signals_MTFFA('MT', 'training', col_centers, row_centers)
save_RFs_and_signals_MTFFA('FFA', 'testing', col_centers, row_centers)
save_RFs_and_signals_MTFFA('FFA', 'training', col_centers, row_centers)

