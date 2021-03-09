
import numpy as np
from glob import glob
from tqdm import tqdm


rel_dir = '../'

def get_signals_from_run(set_t, roi, run_i, signal_selection):
    signals_run_i = np.load(f'{rel_dir}used_data/{set_t}_signalsRF/{roi}/{run_i}_{set_t[:-3]}.npy')
    signals_run_i = signals_run_i[signal_selection,:]

    return signals_run_i

def save_stacked_signals(set_t, roi, time_range=5, shift=4 ):
    '''
    Creates input for the iterator
    
    time_range = is how many frames you want as input in the time dimension
    shift = how many frames delay 
    
    
    '''
    
    
    number_of_run_files = len(glob(f'../used_data/{set_t}_webm/{set_t[:-3]}_video_run*'))
    targets_list = []
    signals_list = []


    for run_i in tqdm(range(1,number_of_run_files+1)):
        if set_t == 'training':
            run_str = f'{run_i:0>3}'
        else:
            run_str = f'{run_i}'
        seen_img =  np.load(f'../used_data/{set_t}_webm/{set_t[:-3]}_video_run{run_str}.npy')
        n_frames = seen_img.shape[0]
        signal_selection = np.arange(time_range)[np.newaxis,:] + np.arange(shift,shift+n_frames)[:,np.newaxis]
        signals_run_i = get_signals_from_run(set_t, roi, run_i, signal_selection)
        signals_list.append(signals_run_i)

    signals = np.concatenate(signals_list).astype('float32')
    signals = signals[:,:,:,np.newaxis,np.newaxis].astype(np.float32)
    signals.shape   

    np.save(f'{set_t[:-3]}/{roi}-shift={shift},time_range={time_range}', signals)

save_stacked_signals('testing', 'MT', time_range=5, shift=4)
save_stacked_signals('training', 'MT', time_range=5, shift=4)
save_stacked_signals('testing', 'FFA', time_range=5, shift=4)
save_stacked_signals('training', 'FFA', time_range=5, shift=4)