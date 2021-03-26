# Data preprocessing

## Targets
The targets in our experiment are Doctor who frames. The following codes were made for the targets:
- generate_targets.py
- warping_targets.py

## Making RFSimages

### What are RFSimages?
RFSimages are brain signals responding to a specific frame, mapped onto the visual space based on its retinotopic mapping of that/those specific brain region(s). These RFSimages are essentially fmri voxels where spatial information is preserved as a 2x2 dimensional picture.

### How do we make RFSimages?
The codes provided in this directory contains everything needed to construct RFSimages from voxels and retinotopic mapping. These codes are to be used chronologically:

- save_brainRF.py
- stack.py

### What will these RFSimages be used for?
The GAN-like model from brain2pix reconstructs images by taking in these RFSimages of the brain. 


# Final Dataset
In the end, for each brain region, we want an input of 5 RFSimages, and an output of 1 Doctor Who frame. 