import mxnet as mx

from mxboard import SummaryWriter
from mxnet import Context, cpu, gpu
from mxnet.ndarray import concat, clip
from tqdm import tqdm
 
from discriminator_6frames import Discriminator
from generator_6frames_simple import Generator

from util import Device
import matplotlib.pyplot as plt 
import numpy as np
from mxnet.io import NDArrayIter
import os

import generator_6frames as gen
import generator_6frames_simple as dis
import time
from glob import glob
from mxnet.ndarray import stack
import mxnet.gluon
import modules_simple as  modules


def run_experiment(runname, batch_size, on_amazon = False, load_model = False, old_runname = None, start_epoch = None ):
    # -----------------------------------------------------------------------------------------------------------
    if on_amazon:
        # AWS Sagemaker:
        import boto3
        region = boto3.Session().region_name
        bucket = boto3.Session().resource('s3').Bucket('sagemaker-inf')
        MODEL_DIR = '/dev/shm/models'

    device       = Device.GPU2
    epochs       = 50
    features     = 64
    all_image_size = 96
    in_chan = 18
        
    context = cpu() if device.value == -1 else gpu(device.value)
    # ----------------------------------------------------
    if load_model:
        summaryWriter = SummaryWriter('logs/'+old_runname, flush_secs=5)
    else:
        summaryWriter = SummaryWriter('logs/'+runname, flush_secs=5)

    train_iter = modules.make_video_iterator('training','V1','V2', 'V3', batch_size=batch_size, shuffle=True)
    test_iter = modules.make_video_iterator('testing', 'V1','V2', 'V3', batch_size=batch_size, shuffle=True)

    RFlocs_V1_overlapped_avg = modules.get_RFs('V1', context)
    RFlocs_V2_overlapped_avg = modules.get_RFs('V2', context)
    RFlocs_V3_overlapped_avg = modules.get_RFs('V3', context)

    with Context(context):
        discriminator = Discriminator(in_chan)
        generator = Generator(in_chan, context)
        
        if load_model:
            if on_amazon:
                generator.network.load_parameters(f'{MODEL_DIR}/saved_models/{runname}/netG_{epoch}.model', ctx=context) 
                discriminator.network.load_parameters(f'{MODEL_DIR}/saved_models/{runname}/netG_{epoch}.model')

                
            else:
                generator.network.load_parameters(f'saved_models/{old_runname}/netG_{start_epoch}.model', ctx=context) 
                discriminator.network.load_parameters(f'saved_models/{old_runname}/netD_{start_epoch}.model')

        
#         gen_lossfun = gen.Lossfun(1, 100, 1, context)
        d = discriminator.network

#         dis_lossfun = dis.Lossfun(1)
        g = generator.network

        print( 'train_dataset_length:', len(train_iter._dataset))

        for epoch in range(epochs):

            loss_discriminator_train = []
            loss_generator_train = []

            # ====================
            # T R AI N I N G
            # ====================

            for RFsignalsV1,RFsignalsV2,RFsignalsV3, targets, recon in tqdm(train_iter, total = len(train_iter)):
                # -------
                # Inputs
                # -------
                inputs1 = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
                inputs2 = modules.get_inputsROI(RFsignalsV2, RFlocs_V2_overlapped_avg, context)
                inputs3 = modules.get_inputsROI(RFsignalsV3, RFlocs_V3_overlapped_avg, context)

                inputs = concat(inputs1, inputs2, inputs3, recon, dim=1)
                # ------------------------------------
                # T R A I N  D i s c r i m i n a t o r
                # ------------------------------------
                targets = targets.transpose((0,1,4,2,3)).reshape((-1,18, 96,96))

                loss_discriminator_train.append(discriminator.train(g, inputs, targets))

                # ----------------------------
                # T R A I N  G e n e r a t o r
                # ----------------------------
                loss_generator_train.append(generator.train(d, inputs, targets))

                # ====================
                # T E S T I N G 
                # ====================
                loss_discriminator_test = []
                loss_generator_test = []

            for  RFsignalsV1,RFsignalsV2,RFsignalsV3, targets, recon in test_iter:
                # -------
                # Inputs
                # -------
                inputs1 = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
                inputs2 = modules.get_inputsROI(RFsignalsV2, RFlocs_V2_overlapped_avg, context)
                inputs3 = modules.get_inputsROI(RFsignalsV3, RFlocs_V3_overlapped_avg, context)
                inputs = concat(inputs1, inputs2, inputs3, recon, dim=1)

                # -----
                # Targets
                # -----            
                targets = targets.transpose((0,1,4,2,3)).reshape((-1,18, 96,96))

                # ----
                # sample randomly from history buffer (capacity 50) 
                # ----
                y_hat = g(inputs)

                z = concat(inputs, y_hat, dim=1)

                dis_loss_test = 0.5 * (discriminator.lossfun(0, d(z)) + discriminator.lossfun(1,d(concat(inputs, targets,dim=1))))

                loss_discriminator_test.append(float(dis_loss_test.asscalar()))

                gen_loss_test = generator.lossfun(1, d(concat(inputs, y_hat, dim=1)), targets.reshape((-1,3,96,96)), y_hat.reshape((-1,3,96,96)))

                loss_generator_test.append(float(gen_loss_test.asscalar()))

#             os.makedirs('saved_models/'+old_runname, exist_ok=True)
#             generator.network.save_parameters(f'saved_models/{old_runname}/netG_{epoch+start_epoch+1}.model')
#             discriminator.network.save_parameters(f'saved_models/{old_runname}/netD_{epoch+start_epoch+1}.model')
            os.makedirs('saved_models/'+runname, exist_ok=True)
            generator.network.save_parameters(f'saved_models/{runname}/netG_{epoch}.model')
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



run_experiment(runname = 'RECON', batch_size = 2)
