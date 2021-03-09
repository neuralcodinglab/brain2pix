# Brain2Pix: Supplementary materials

## Introduction
Welcome to the repository that contains supplementary materials and the source code for the paper "Brain2Pix: Fully convolutional naturalistic video reconstruction from brain activity".

The brain2pix model consists of 2 parts. 1) Making RFSimages, and 2) training the GAN-like model. For reproducing the experiment, first check out the [data_preprocessing](data_preprocessing/README.md) files for making the RFSimages and then the [experiment](experiment/README.md) files for training the model.

## Folders
data_preprocessing: this folder entails all the steps of transforming raw brain signals into RFSimages.

experiment: codes containing the model and training loop for the experiments.

visualizations: reconstruction videos in GIF format and figures in PFD format.

## Results

<b>Main results -- FixedRF (test set):</b>

fixed RFSimage | reconstruction | ground truth

![fixedRF_recons_of_all_frames_as_video_a](/additional_results/recons_fixed_of_all_frames_as_video_a.gif)


![fixedRF_recons_of_all_frames_as_video_b](/additional_results/recons_fixed_of_all_frames_as_video_b.gif)


![fixedRF_recons_of_all_frames_as_video_c](/additional_results/recons_fixed_of_all_frames_as_video_c.gif)


<b>Main results -- LearnedRF (test set):</b>

learned RFSimage | reconstruction | ground truth

![learnedRF_recons_of_all_frames_as_video_a](/additional_results/recons_of_all_frames_as_video_a.gif)


![learned_RF_recons_of_all_frames_as_video_b](/additional_results/recons_of_all_frames_as_video_b.gif)


![learned_RF_recons_of_all_frames_as_video_c](/additional_results/recons_of_all_frames_as_video_c.gif)


<b>Additional results (test set):</b>

![recons_of_all_frames_as_video_additional_a](/additional_results/recons_of_all_frames_as_video_additional_a.gif)
![recons_of_all_frames_as_video_additional_b](/additional_results/recons_of_all_frames_as_video_additional_b.gif)
    
<b>Codes: </b>

More information on the code in the README inside the "code" folder.

- Main experiment:
    - To replicate the main experiment, please see the "code/learnableRF" and "code/fixedRF" folders.

- Baseline experiments:    
    - To replicate the ~n-hrs fMRI data experiment, please see the "code/split_data" folder.

    - To replicate the Shen et. al., 2019 experiment, please see the "code/dcgan_baseline" folder.

    - To replicate the vim2 data, please see the 'code/vim2' folder.
    

    
    
<b>Datasets:</b>

<b> Dr. Who: </b> The Dr. Who dataset is publicly available. The first author of the dataset paper (Seeliger et. al., 2019) mentioned (on http://disq.us/p/23bj45d) that the link will be activated soon. For now it is available by contacting them. 



<b> vim2: </b> This dataset was taken from http://crcns.org/, originally published by Nishimoto et. al, 2011.

<b> References: </b>

Nishimoto, S., Vu, A. T., Naselaris, T., Benjamini, Y., Yu, B., & Gallant, J. L. (2011). Reconstructing visual experiences from brain activity evoked by natural movies. Current Biology, 21(19), 1641-1646.


Seeliger, K., et al. "A large single-participant fMRI dataset for probing brain responses to naturalistic stimuli in space and time." bioRxiv (2019): 687681.


Shen, G., Dwivedi, K., Majima, K., Horikawa, T., & Kamitani, Y. (2019). End-to-end deep image reconstruction from human brain activity. Frontiers in Computational Neuroscience, 13, 21.




    
