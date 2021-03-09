import numpy as np
import mxnet as mx
from glob import glob 
from tqdm import tqdm
from retinawarp.retina.retina import warp_image

rel_dir = '../../MonkeyProject/git/models/whoRF/'

def get_RFs(roi, context):
    '''
    roi (str): region of interest, one of the following:
        - V1
        - V2
        - V3
        
    this loads the speficied ROI RF locations, to become  and the overlapping RFs are divided 
    '''
    
    RFlocs = np.concatenate([
        np.load(f'{rel_dir}used_data/RF_locations/RFlocs_{roi}_dorsal.npy'), 
        np.load(f'{rel_dir}used_data/RF_locations/RFlocs_{roi}_ventral.npy')
    ])

    RFlocs_sum = np.sum(RFlocs, axis = 0)
    RF_not_null_mask = RFlocs_sum!=0

    RFlocs[:, RF_not_null_mask]= RFlocs[:, RF_not_null_mask]/RFlocs_sum[RF_not_null_mask]
    RFlocs_overlapped_avg = mx.nd.array(RFlocs).expand_dims(0).expand_dims(0)
    
    return RFlocs_overlapped_avg.as_in_context(context)


def leclip(x_xlip):
    x_clip0 = x_xlip - x_xlip.min()
    return x_clip0/x_clip0.max()

def get_inputsROI(signals, RF_overlapped):
    
    '''
    
    Multiplying the signals with the RF centers (obtained from get_RFs() )together to become RFSimages. 
    Epects both to be mxnet.ndarrays.
    
    '''
    channels = mx.nd.multiply( RF_overlapped, signals)
    inputs = channels.sum(axis=2)
    return inputs


def get_signals_from_run(roi, set_t, run_i, signal_selection):
    signalsD_run_i = np.load(f'{rel_dir}used_data/{set_t}_signalsRF/{roi}/dorsal_{run_i}_{set_t[:-3]}.npy')
    signalsD_run_i = signalsD_run_i[signal_selection,:]

    signalsV_run_i = np.load(f'{rel_dir}used_data/{set_t}_signalsRF/{roi}/ventral_{run_i}_{set_t[:-3]}.npy')
    signalsV_run_i = signalsV_run_i[signal_selection,:]

    signals_run_i = np.concatenate([signalsD_run_i, signalsV_run_i], axis=2)
    
    return signals_run_i


def make_iterator(set_t, *roi, shift=4, time_range=5, batch_size=16, shuffle=False):
    
    
    '''
    
    This iterator returns signals and targets.
    
    '''
    
    number_of_run_files = len(glob(f'{rel_dir}used_data/{set_t}_webm/{set_t[:-3]}_video_run*'))
    
    targets_list = []
    signals_list_per_roi = [[] for _ in roi]
    
    for run_i in tqdm(range(1,number_of_run_files+1)):
        if set_t == 'training':
            run_str = f'{run_i:0>3}'
        else:
            run_str = f'{run_i}'
            
        seen_img =  np.load(f'{rel_dir}used_data/{set_t}_webm/{set_t[:-3]}_video_run{run_str}.npy')
        
        if len(seen_img) < 1:
            raise FileNotFoundError('WHERE ARE THE FILES? THANKU')
        else:
            print(len(seen_img))
            
        n_frames = seen_img.shape[0]
        targets_list.append(seen_img/255)

        signal_selection = np.arange(time_range)[np.newaxis,:] + np.arange(shift,shift+n_frames)[:,np.newaxis]

        for i in range(len(roi)):
            signals_run_i = get_signals_from_run(roi[i], set_t, run_i, signal_selection)
            signals_list_per_roi[i].append(signals_run_i)

    targets = np.concatenate(targets_list).astype('float32')
    
    
    ## R E T I N A W A R P
    print(targets.shape)
    for fi in tqdm(range(targets.shape[3])): 
        img = np.transpose(targets[:,:,:,fi], [1,2,0])
        img = warp_image(img)
        targets[:,:,:,fi] = np.transpose(img, [2,0,1])
    
    signals_per_roi = []
    for signal_list in signals_list_per_roi:
        signals = np.concatenate(signal_list)
        signals = signals[:,:,:,np.newaxis,np.newaxis].astype(np.float32)
        signals_per_roi.append(signals)
    
    dataset = mx.gluon.data.ArrayDataset(*signals_per_roi, targets)
    return mx.gluon.data.DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)


def make_synthetic_iterator(RFlocs, set_t, batch_size, shuffle):
        
        number_of_run_files = len(glob(f'{set_t}_webm/{set_t[:-3]}_video_run*'))
        targets_list = []
        signals_list = []
        for run_i in range(1,number_of_run_files+1):
            if set_t == 'training':
                run_str = f'{run_i:0>3}'
            else:
                run_str = f'{run_i}'
            seen_img =  np.load(f'{set_t}_webm/{set_t[:-3]}_video_run{run_str}.npy')
            n_frames = seen_img.shape[0]
            targets_list.append(seen_img/255)            
        
        targets = np.concatenate(targets_list).astype('float32')
        
        targets_colsum = targets.sum(3)
        dotnumber_list = []
        
        for i in range(RFlocs.shape[0]):
            dotnumber = targets_colsum[:,RFlocs[i,:,:].astype('bool')]
            dotnumber_list.append(dotnumber)
        dot_numbers = np.concatenate(dotnumber_list, axis=1)
        
        dot_numbers = dot_numbers[:,np.newaxis,:,np.newaxis,np.newaxis]
        
        dataset = mx.gluon.data.ArrayDataset(dot_numbers, targets)

        return mx.gluon.data.DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)

    
def make_iterator_preprocessed(set_t, *roi, shift=4, time_range=5, batch_size=16, shuffle=False, synthetic=False):
    
    targets = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/targets_warped.npy').transpose((0,3,1,2))
    signals_per_roi = []
    if synthetic:
        for roi_i in roi:
            signals = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/synthetic-{roi_i}.npy')
            signals_per_roi.append(signals)
    else:
        for roi_i in roi:
            signals = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/{roi_i}-shift={shift},time_range={time_range}.npy')
            signals_per_roi.append(signals)
            
    dataset = mx.gluon.data.ArrayDataset(*signals_per_roi, targets)
    return mx.gluon.data.DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)
