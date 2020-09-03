# -------------
# I M P O R T S
# -------------

# MXNET:
import mxnet as mx
from mxboard import SummaryWriter
from mxnet import Context, cpu, gpu
from mxnet.ndarray import concat, clip
from mxnet.io import NDArrayIter
from mxnet.ndarray import stack
import mxnet.gluon

# Other libraries:
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt 
import numpy as np
import discriminator as dis
import time
from glob import glob

# MODULES:
from discriminator import Discriminator
from generator_FC import Generator
import generator_FC as gen
import modules as modules
from util import Device

# -----------------------------------------------------------------------------------------------------------

# ----------------------
# Check before running:
# ----------------------
runname = f'baseline_FC_1'
device       = Device.GPU1
epochs       = 50
features     = 64
batch_size = 4
all_image_size = 96
in_chan = 15
MODEL_DIR = '.'
on_amazon = False

load_model = False
old_runname = 'baseline_FC0' # insert runname that will be loaded.
start_epoch = 29

# -----------------------------------------------------------------------------------------------------------
if on_amazon:
    # AWS Sagemaker:
    import boto3
    region = boto3.Session().region_name
    bucket = boto3.Session().resource('s3').Bucket('sagemaker-inf')
    MODEL_DIR = '/dev/shm/models'


if __name__ == "__main__":

    # -------------------------------
    # Context as needed to run on GPU
    # -------------------------------
    context = cpu() if device.value == -1 else gpu(device.value)

    # ----------------------------------------------------
    # SummaryWriter is for visualizing logs in tensorboard
    # ----------------------------------------------------
    if load_model:
        summaryWriter = SummaryWriter('logs/'+old_runname, flush_secs=5)
    else:
        summaryWriter = SummaryWriter('logs/'+runname, flush_secs=5)


    with Context(context):
        # ----------------------------------------------------
        # RF centers: overlapping 
        # ----------------------------------------------------
        RFlocs_V1_overlapped_avg = modules.get_RFs('V1', context)
        RFlocs_V2_overlapped_avg = modules.get_RFs('V2', context)
        RFlocs_V3_overlapped_avg = modules.get_RFs('V3', context)

        test_iter = modules.make_iterator_preprocessed('testing', 'V1', 'V2', 'V3', batch_size=batch_size, shuffle=False)

        RF_signals_lengths = []
        for *RFsignals, targets in test_iter:
            for s in RFsignals:
                RF_signals_lengths.append(s.shape[2])
            break

        discriminator = Discriminator(in_chan)

        generator = Generator(in_chan, context, RF_in_units=RF_signals_lengths, 
                              conv_input_shape=(96, 96),
                                  train_RF=True)
        if load_model:
            generator.network.load_parameters(f'saved_models/{old_runname}/netG_{start_epoch}.model', ctx=context) 
            generator.rf_mapper.load_parameters(f'saved_models/{old_runname}/RFlayers_{start_epoch}.model', allow_missing=True)
            discriminator.network.load_parameters(f'saved_models/{old_runname}/netD_{start_epoch}.model')

        gen_lossfun = gen.Lossfun(1, 100, 1, context)
        d = discriminator.network

        dis_lossfun = dis.Lossfun(1)
        g = generator.network
        

        train_iter = modules.make_iterator_preprocessed('training', 'V1', 'V2', 'V3', batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):

                loss_discriminator_train = []
                loss_generator_train = []

                # ====================
                # T R AI N I N G
                # ====================
                for *RFsignals, targets in tqdm(train_iter, total = len(train_iter)):
                    # -----
                    # Inputs
                    # -----
                    RFsignals = [x.as_in_context(context) for x in RFsignals]

                    # ------------------------------------
                    # T R A I N  D i s c r i m i n a t o r
                    # ------------------------------------
                    targets = targets.as_in_context(context)

                    loss_discriminator_train.append(
                        discriminator.train(g, 
                                            generator.rf_mapper(RFsignals), 
                                            targets)
                    )

                    # ----------------------------
                    # T R A I N  G e n e r a t o r
                    # ----------------------------
                    loss_generator_train.append(generator.train(d, RFsignals, targets))
                    
                os.makedirs('saved_models/'+runname, exist_ok=True)
                generator.network.save_parameters(f'saved_models/{runname}/netG_{epoch}.model')
                generator.rf_mapper.save_parameters(f'saved_models/{runname}/RFlayers_{epoch}.model')
                discriminator.network.save_parameters(f'saved_models/{runname}/netD_{epoch}.model')
