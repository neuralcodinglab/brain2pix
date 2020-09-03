

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

from discriminator import Discriminator
from generator_learnRF import Generator
import generator_learnRF as gen
import modules as modules
from util import Device


# ----------------------
# Check before running:
# ----------------------
runname = f'finalVIM2{int(time.time())}'
device       = Device.GPU1
epochs       = 50
features     = 64
batch_size = 4
all_image_size = 96
in_chan = 15


# In[3]:


# -------------------------------
# Context as needed to run on GPU
# -------------------------------
context = cpu() if device.value == -1 else gpu(device.value)

# ----------------------------------------------------
# SummaryWriter is for visualizing logs in tensorboard
# ----------------------------------------------------
summaryWriter = SummaryWriter('../../logs/'+runname, flush_secs=5)

with Context(context):

    test_iter = modules.make_iterator_preprocessed('testing', 'V1', 'V2', 'V3', batch_size=batch_size, shuffle=True)

    RF_signals_lengths = []
    for *RFsignals, targets in test_iter:
        for s in RFsignals:
            RF_signals_lengths.append(s.shape[2])
        break

    discriminator = Discriminator(in_chan)
    generator = Generator(in_chan, context, RF_in_units=RF_signals_lengths, 
                          conv_input_shape=(96, 96), train_RF=True)

    gen_lossfun = gen.Lossfun(1, 100, 1, context)
    d = discriminator.network

    dis_lossfun = dis.Lossfun(1)
    g = generator.network

    train_iter = modules.make_iterator_preprocessed('training', 'V1', 'V2', 'V3', batch_size=batch_size, shuffle=True)

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
        # ====================
        # T E S T I N G 
        # ====================
        loss_discriminator_test = []
        loss_generator_test = []

        for *RFsignals, targets in tqdm(test_iter, total = len(test_iter)):
            # -----
            # Inputs
            # -----
            RFsignals = [x.as_in_context(context) for x in RFsignals]

            learned_RF = generator.rf_mapper(RFsignals)

            # --------
            # Targets
            # --------         
            targets = targets.as_in_context(context)

            # -------------------------------------------------
            # sample randomly from history buffer (capacity 50) 
            # -------------------------------------------------
            y_hat = g(learned_RF)
            z = concat(learned_RF, y_hat, dim=1)

            dis_loss_test = 0.5 * (dis_lossfun(0, d(z)) + dis_lossfun(1,d(concat(learned_RF, targets,dim=1))))

            loss_discriminator_test.append(float(dis_loss_test.asscalar()))

            gen_loss_test = gen_lossfun(1, d(concat(learned_RF, y_hat, dim=1)), targets, y_hat)

            loss_generator_test.append(float(gen_loss_test.asscalar()))
        
            # ----------------------------
            # T R A I N  G e n e r a t o r
            # ----------------------------
#             loss_generator_train.append(generator.train(d, RFsignals, targets))
        os.makedirs('saved_models/'+runname, exist_ok=True)
        
        generator.network.save_parameters(f'saved_models/{runname}/netG_{epoch}.model')
        generator.rf_mapper.save_parameters(f'saved_models/{runname}/RFlayers_{epoch}.model')
        discriminator.network.save_parameters(f'saved_models/{runname}/netD_{epoch}.model')
            
        # ------------------------------------------------------------------
        # T R A I N I N G Losses
        # ------------------------------------------------------------------
        np.save(f'saved_models/{runname}/Gloss_train{epoch}', np.array(loss_generator_train))
        np.save(f'saved_models/{runname}/Dloss_train{epoch}', np.array(loss_discriminator_train))
        # ------------------------------------------------------------------
        # T E S T I N G Losses
        # ------------------------------------------------------------------
        np.save(f'saved_models/{runname}/Gloss_test{epoch}', np.array(loss_generator_test))
        np.save(f'saved_models/{runname}/Dloss_test{epoch}', np.array(loss_discriminator_test))
              