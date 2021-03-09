import mxnet as mx

from mxboard import SummaryWriter
from mxnet import Context, cpu, gpu
from mxnet.ndarray import concat, clip
from tqdm import tqdm
 
from discriminator import Discriminator
from generator_vgg import Generator

from util import Device
import matplotlib.pyplot as plt 
import numpy as np
from mxnet.io import NDArrayIter
import os

import generator_vgg as gen
import discriminator as dis
import time
from glob import glob
from mxnet.ndarray import stack
import mxnet.gluon
import modules

runname = f'V3-voxelRF123{int(time.time())}'

device       = Device.GPU1
epochs       = 1000
features     = 64
batch_size = 8
all_image_size = 96
in_chan = 15


if __name__=='__main__':

    context = cpu() if device.value == -1 else gpu(device.value)
    summaryWriter = SummaryWriter('logs/'+runname, flush_secs=5)

    train_iter = modules.make_iterator_preprocessed('training','V1','V2', 'V3', batch_size=batch_size, shuffle=True)
    test_iter = modules.make_iterator_preprocessed('testing', 'V1','V2', 'V3', batch_size=batch_size, shuffle=True)

    RFlocs_V1_overlapped_avg = modules.get_RFs('V1')
    RFlocs_V2_overlapped_avg = modules.get_RFs('V2')
    RFlocs_V3_overlapped_avg = modules.get_RFs('V3')

    with Context(context):
        discriminator = Discriminator(in_chan)
        generator = Generator(in_chan, context)
        gen_lossfun = gen.Lossfun(1, 100, 1, context)
        d = discriminator.network

        dis_lossfun = dis.Lossfun(1)
        g = generator.network

        for epoch in range(epochs):

            loss_discriminator_train = []
            loss_generator_train = []

            # ====================
            # T R AI N I N G
            # ====================

            for RFsignalsV1,RFsignalsV2,RFsignalsV3, targets in tqdm(train_iter, total = len(train_iter)):
                # -------
                # Inputs
                # -------
                inputs1 = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
                inputs2 = modules.get_inputsROI(RFsignalsV2, RFlocs_V2_overlapped_avg, context)
                inputs3 = modules.get_inputsROI(RFsignalsV3, RFlocs_V3_overlapped_avg, context)
                inputs = concat(inputs1, inputs2, inputs3, dim=1)
                # ------------------------------------
                # T R A I N  D i s c r i m i n a t o r
                # ------------------------------------
                targets = targets.as_in_context(context).transpose((0,3,1,2))

                loss_discriminator_train.append(discriminator.train(g, inputs, targets))

                # ----------------------------
                # T R A I N  G e n e r a t o r
                # ----------------------------
                loss_generator_train.append(generator.train(d, inputs, targets))

            os.makedirs('saved_models/'+runname, exist_ok=True)
            generator.network.save_parameters(f'saved_models/{runname}/netG_{epoch}.model')
            discriminator.network.save_parameters(f'saved_models/{runname}/netD_{epoch}.model')

            # ====================
            # T E S T I N G 
            # ====================
            loss_discriminator_test = []
            loss_generator_test = []

            for  RFsignalsV1,RFsignalsV2,RFsignalsV3, targets in test_iter:
                # -------
                # Inputs
                # -------
                inputs1 = modules.get_inputsROI(RFsignalsV1, RFlocs_V1_overlapped_avg, context)
                inputs2 = modules.get_inputsROI(RFsignalsV2, RFlocs_V2_overlapped_avg, context)
                inputs3 = modules.get_inputsROI(RFsignalsV3, RFlocs_V3_overlapped_avg, context)
                inputs = concat(inputs1, inputs2, inputs3, dim=1)
                
                # -----
                # Targets
                # -----            
                targets = targets.as_in_context(context).transpose((0,3,1,2))


                # ----
                # sample randomly from history buffer (capacity 50) 
                # ----

                z = concat(inputs, g(inputs), dim=1)

                dis_loss_test = 0.5 * (dis_lossfun(0, d(z)) + dis_lossfun(1,d(concat(inputs, targets,dim=1))))

                loss_discriminator_test.append(float(dis_loss_test.asscalar()))

                gen_loss_test = (lambda y_hat: gen_lossfun(1, d(concat(inputs, y_hat, dim=1)), targets, y_hat))(generator.network(inputs))

                loss_generator_test.append(float(gen_loss_test.asscalar()))

            summaryWriter.add_image("input", modules.leclip(inputs.expand_dims(2).sum(1)), epoch)
            summaryWriter.add_image("target", modules.leclip(targets), epoch)
            summaryWriter.add_image("pred", modules.leclip(g(inputs)), epoch)
            summaryWriter.add_scalar("dis/loss_discriminator_train", sum(loss_discriminator_train) / len(loss_discriminator_train), epoch)
            summaryWriter.add_scalar("gen/loss_generator_train", sum(loss_generator_train) / len(loss_generator_train), epoch)

            summaryWriter.add_scalar("dis/loss_discriminator_test", sum(loss_discriminator_test) / len(loss_discriminator_test), epoch)
            summaryWriter.add_scalar("gen/loss_generator_test", sum(loss_generator_test) / len(loss_generator_test), epoch)

            # ------------------------------------------------------------------
            # T R A I N I N G Losses
            # ------------------------------------------------------------------
            np.save(f'saved_models/{runname}/Gloss_train', np.array(loss_generator_train))
            np.save(f'saved_models/{runname}/Dloss_train', np.array(loss_discriminator_train))
            # ------------------------------------------------------------------
            # T E S T I N G Losses
            # ------------------------------------------------------------------
            np.save(f'saved_models/{runname}/Gloss_test', np.array(loss_generator_test))
            np.save(f'saved_models/{runname}/Dloss_test', np.array(loss_discriminator_test))

