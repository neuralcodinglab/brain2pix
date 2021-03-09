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


# AWS Sagemaker:
import boto3
region = boto3.Session().region_name
bucket = boto3.Session().resource('s3').Bucket('sagemaker-inf')
MODEL_DIR = '/dev/shm/models'

# MODULES:
from discriminator import Discriminator
from generator_learnRF import Generator
import generator_learnRF as gen
import modules as modules
from util import Device
# -----------------------------------------------------------------------------------------------------------

# ----------------------
# Check before running:
# ----------------------
runname = f'learnableRF{int(time.time())}'
device       = Device.GPU0
epochs       = 1000
features     = 64
batch_size = 12
all_image_size = 96
in_chan = 15

# -----------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # -------------------------------
    # Context as needed to run on GPU
    # -------------------------------
    context = cpu() if device.value == -1 else gpu(device.value)
    
    # ----------------------------------------------------
    # SummaryWriter is for visualizing logs in tensorboard
    # ----------------------------------------------------
    summaryWriter = SummaryWriter('../../logs/'+runname, flush_secs=5)
    
with Context(context):
    # ----------------------------------------------------
    # RF centers: overlapping 
    # ----------------------------------------------------
    RFlocs_V1_overlapped_avg = modules.get_RFs('V1', context)
    RFlocs_V2_overlapped_avg = modules.get_RFs('V2', context)
    RFlocs_V3_overlapped_avg = modules.get_RFs('V3', context)
    
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
            brain_inputs = [modules.get_inputsROI(RFsignal, RFlocs_overlapped_avg)
                            for RFsignal, RFlocs_overlapped_avg
                            in zip(RFsignals, [RFlocs_V1_overlapped_avg,
                                               RFlocs_V2_overlapped_avg,
                                               RFlocs_V3_overlapped_avg])]
            
            brain_inputs = concat(*brain_inputs, dim=1)
            # ------------------------------------
            # T R A I N  D i s c r i m i n a t o r
            # ------------------------------------
            targets = targets.as_in_context(context)

            loss_discriminator_train.append(
                discriminator.train(g, 
                                    generator.rf_mapper(RFsignals)+brain_inputs, 
                                    targets)
            )

            # ----------------------------
            # T R A I N  G e n e r a t o r
            # ----------------------------
            loss_generator_train.append(generator.train(d, RFsignals, brain_inputs, targets))
            
        os.makedirs(MODEL_DIR+'/saved_models/'+runname, exist_ok=True)
        generator.network.save_parameters(f'{MODEL_DIR}/saved_models/{runname}/netG_{epoch}.model')
        generator.rf_mapper.save_parameters(f'{MODEL_DIR}/saved_models/{runname}/RFlayers_{epoch}.model')
        discriminator.network.save_parameters(f'{MODEL_DIR}/saved_models/{runname}/netD_{epoch}.model')
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/netG_{epoch}.model'     ,f'ec2_models/{runname}/netG_{epoch}.model' )
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/RFlayers_{epoch}.model' ,f'ec2_models/{runname}/RFlayers_{epoch}.model')
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/netD_{epoch}.model'     ,f'ec2_models/{runname}/netD_{epoch}.model')
        
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
            brain_inputs = [modules.get_inputsROI(RFsignal, RFlocs_overlapped_avg)
                            for RFsignal, RFlocs_overlapped_avg
                            in zip(RFsignals, [RFlocs_V1_overlapped_avg,
                                               RFlocs_V2_overlapped_avg,
                                               RFlocs_V3_overlapped_avg])]
            
            merged_inputs = concat(*brain_inputs, dim=1) + generator.rf_mapper(RFsignals)
            
            # --------
            # Targets
            # --------         
            targets = targets.as_in_context(context)

            # -------------------------------------------------
            # sample randomly from history buffer (capacity 50) 
            # -------------------------------------------------
            y_hat = g(merged_inputs)
            z = concat(merged_inputs, y_hat, dim=1)

            dis_loss_test = 0.5 * (dis_lossfun(0, d(z)) + dis_lossfun(1,d(concat(merged_inputs, targets,dim=1))))

            loss_discriminator_test.append(float(dis_loss_test.asscalar()))

            gen_loss_test = gen_lossfun(1, d(concat(merged_inputs, y_hat, dim=1)), targets, y_hat)

            loss_generator_test.append(float(gen_loss_test.asscalar()))
            
        # ------------------------------
        # Saving to logs for tensorboard
        # ------------------------------
        summaryWriter.add_image("input", modules.leclip(merged_inputs.expand_dims(2).sum(1)), epoch)
        summaryWriter.add_image("target", modules.leclip(targets), epoch)
        summaryWriter.add_image("pred", modules.leclip(g(merged_inputs)), epoch)
        summaryWriter.add_scalar("dis/loss_discriminator_train", sum(loss_discriminator_train) / len(loss_discriminator_train), epoch)
        summaryWriter.add_scalar("gen/loss_generator_train", sum(loss_generator_train) / len(loss_generator_train), epoch)
        summaryWriter.add_scalar("dis/loss_discriminator_test", sum(loss_discriminator_test) / len(loss_discriminator_test), epoch)
        summaryWriter.add_scalar("gen/loss_generator_test", sum(loss_generator_test) / len(loss_generator_test), epoch)

        # -----------
        # Save models
        #------------
        np.save(f'{MODEL_DIR}/saved_models/{runname}/Gloss_train', np.array(loss_generator_train))
        np.save(f'{MODEL_DIR}/saved_models/{runname}/Dloss_train', np.array(loss_discriminator_train))
        
        # ------------------------------------------------------------------
        # T E S T I N G Losses
        # ------------------------------------------------------------------
        np.save(f'{MODEL_DIR}/saved_models/{runname}/Gloss_test', np.array(loss_generator_test))
        np.save(f'{MODEL_DIR}/saved_models/{runname}/Dloss_test', np.array(loss_discriminator_test))

        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/Gloss_train.npy' ,f'ec2_models/{runname}/Gloss_train.npy')
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/Dloss_train.npy' ,f'ec2_models/{runname}/Dloss_train.npy')
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/Gloss_test.npy' ,f'ec2_models/{runname}/Gloss_test.npy')
        bucket.upload_file(f'{MODEL_DIR}/saved_models/{runname}/Dloss_test.npy' ,f'ec2_models/{runname}/Dloss_test.npy')