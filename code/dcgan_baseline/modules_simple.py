import numpy as np
import mxnet as mx
from glob import glob 
from tqdm import tqdm
# from retinawarp.retina.retina import warp_image

rel_dir = '../../MonkeyProject/git/models/whoRF/'


def get_signals_from_run(roi, set_t, run_i, signal_selection):
    signalsD_run_i = np.load(f'{set_t}_signalsRF/{roi}/dorsal_{run_i}_{set_t[:-3]}.npy')
    signalsD_run_i = signalsD_run_i[signal_selection,:]

    signalsV_run_i = np.load(f'{set_t}_signalsRF/{roi}/ventral_{run_i}_{set_t[:-3]}.npy')
    signalsV_run_i = signalsV_run_i[signal_selection,:]

    signals_run_i = np.concatenate([signalsD_run_i, signalsV_run_i], axis=2)
    
    return signals_run_i


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


def make_iterator(set_t, *roi, shift=4, time_range=5, batch_size=16, shuffle=False):
    
    
    '''
    
    This iterator returns signals and targets.
    
    '''
    
    number_of_run_files = len(glob(f'{set_t}_webm/{set_t[:-3]}_video_run*'))
    
    targets_list = []
    signals_list_per_roi = [[] for _ in roi]
    
    for run_i in tqdm(range(1,number_of_run_files+1)):
        if set_t == 'training':
            run_str = f'{run_i:0>3}'
        else:
            run_str = f'{run_i}'
            
        seen_img =  np.load(f'{set_t}_webm/{set_t[:-3]}_video_run{run_str}.npy')
        n_frames = seen_img.shape[0]
        targets_list.append(seen_img/255)

        signal_selection = np.arange(time_range)[np.newaxis,:] + np.arange(shift,shift+n_frames)[:,np.newaxis]

        for i in range(len(roi)):
            signals_run_i = get_signals_from_run(roi[i], set_t, run_i, signal_selection)
            signals_list_per_roi[i].append(signals_run_i)

    targets = np.concatenate(targets_list).astype('float32')
    
    
    ## R E T I N A W A R P
    print(targets.shape)
    for fi in range(targets.shape[3]): 
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

def get_inputsROI(RF_ROI, RF_overlapped, context):
    
    channels = mx.nd.multiply(
    RF_overlapped.as_in_context(context),
    RF_ROI.as_in_context(context)
    )
    inputs = channels.sum(axis=2)
    return inputs

def leclip(x_xlip):
    x_clip0 = x_xlip - x_xlip.min()
    return x_clip0/x_clip0.max()

def make_iterator_preprocessed(set_t, *roi, shift=4, time_range=5, batch_size=16, shuffle=False, synthetic=False, fraction_train = 1):
    
    targets = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/targets_warped.npy').transpose((0,3,1,2))
    signals_per_roi = []
    if synthetic:
        for roi_i in roi:
            signals = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/synthetic-{roi_i}.npy')
            signals_per_roi.append(signals)
    else:
        for roi_i in roi:
            signals = np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/{roi_i}-shift={shift},time_range={time_range}.npy')
            
            signals_per_roi.append(signals[0 : int(len(signals) * fraction_train)])
            
    dataset = mx.gluon.data.ArrayDataset(*signals_per_roi, targets[0 : int(len(targets) * fraction_train)])
    return mx.gluon.data.DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle)



class ConcatDataset(mx.gluon.data.Dataset):
    def __init__(self, dataset_list):
        self._dataset_list = dataset_list
        self._lens_cumsum = np.cumsum([0] + [len(x) for x in dataset_list])
    
    def __getitem__(self, idx):
        is_lower_then_idx = self._lens_cumsum <= idx
        dataset_idx = np.argmax(self._lens_cumsum[is_lower_then_idx])
        dataset = self._dataset_list[dataset_idx]
        
        dataset_start_idx = self._lens_cumsum[dataset_idx]
        idx_for_dataset = idx - dataset_start_idx
        return dataset[idx_for_dataset]

    def __len__(self):
        return self._lens_cumsum[-1]
    
def make_video_iterator(set_t, *roi, batch_size=16,shift=4, time_range=5, shuffle=True):
    recons = np.load('recons.npy')

    len_list = np.concatenate([[0],np.load(f'length_list_{set_t[:-3]}.npy')])
    lens_cumsum = len_list.cumsum()

    roi_array_list = [
        np.load(f'{rel_dir}pre_processed_data/{set_t[:-3]}/{roi_i}-shift={shift},time_range={time_range}.npy')
        for roi_i in roi
    ]

    run_datasets = []
    if set_t == 'testing':
        dirs = sorted(glob('../video_output/testing_webm/**.npy'))
    else:
        dirs = sorted(glob('../video_output/training_webm/**.npy'))

    for i, dir_i in enumerate(dirs):
        if i == 15:
            break
            
        if set_t == 'testing':
            run_i = dir_i[len('../video_output/testing_webm/test_video_run'):-len('.npy')]
        else:
            run_i = dir_i[len('../video_output/training_webm/train_video_run'):-len('.npy')]

        targets_i = np.load(f'../video_output/{set_t[:-3]}_6frames_targets/{set_t[:-3]}_movie_target_run{run_i}.npy').astype('float32')

        signals_per_roi_i = []

        for signals in roi_array_list:
            signals_i = signals[lens_cumsum[i]:lens_cumsum[i]+targets_i.shape[0]]
            
            signals_per_roi_i.append(signals_i)
            
        recon_i = recons[lens_cumsum[i]:lens_cumsum[i]+targets_i.shape[0]]
        dataset_i = mx.gluon.data.ArrayDataset(*signals_per_roi_i, targets_i, recon_i)
        run_datasets.append(dataset_i)
        
    dataset = ConcatDataset(run_datasets)
    dl =  mx.gluon.data.DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=shuffle)
    return dl