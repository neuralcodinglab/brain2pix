# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


# ### 1. Downsample targets temporally to match TR & and desired img dimension

target_dir = '/mnt/sdb/Ubuntu/whoRF/doctor_who_frames'

# R A W   D A T A 
raw_dir =  f'{target_dir}/raw_frames'

list_testvids = os.listdir(f'{raw_dir}/test')
list_trainvids = os.listdir(f'{raw_dir}/train')

# D O W N S A M P L E D

downsamp_dir = f'{target_dir}/in_between_processing'

os.makedirs(f'{downsamp_dir}/testing_webm', exist_ok=True)
os.makedirs(f'{downsamp_dir}/training_webm', exist_ok=True)
os.makedirs(f'{downsamp_dir}/testing_npy', exist_ok=True)
os.makedirs(f'{downsamp_dir}/training_npy', exist_ok=True)

###### T E S T I N G
for i in range(1, len(list_testvids)+1):
    get_ipython().system('ffmpeg -i /mnt/sdb/Ubuntu/whoRF/doctor_who_frames/raw_frames/test/run_{i}.webm -an -vf "scale=96:96,fps=1.42857142857" -y /mnt/sdb/Ubuntu/whoRF/doctor_who_frames/in_between_processing/testing_webm/run_{i}_07.webm')

###### T R A I N I N G
for i in range(1, len(list_trainvids)+1):
    i_z = str(i).zfill(3)
    get_ipython().system('ffmpeg -i /mnt/sdb/Ubuntu/whoRF/doctor_who_frames/raw_frames/train/run_{i_z}.webm -an -vf "scale=96:96,fps=1.42857142857" -y /mnt/sdb/Ubuntu/whoRF/doctor_who_frames/in_between_processing/training_webm/run_{i_z}_07.webm')


# ### 2. Rename and convert frames to numpy arrays
def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame[..., ::-1])

    return np.stack(frames)


def to_numpy_all_runs(set_t):
    
    '''saves each run into np format'''
    
    if set_t == 'testing':
        list_vid = os.listdir(f'{raw_dir}/test')

    if set_t == 'training':
        list_vid = os.listdir(f'{raw_dir}/train')

    for run_nr in tqdm(range(1, len(list_vid)+1)):
        if set_t == 'training':
            run_nr = str(run_nr).zfill(3)
        fps_run = read_frames(f'{downsamp_dir}/{set_t}_webm/run_{run_nr}_07.webm')
        
        np.save(f'{downsamp_dir}/{set_t}_npy/{set_t[:-3]}_video_run{run_nr}.npy', np.array(fps_run))

        
#save runs in np format
to_numpy_all_runs('training')
to_numpy_all_runs('testing')















