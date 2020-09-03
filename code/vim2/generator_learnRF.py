from typing import Any, List

from mxnet import autograd
from mxnet.gluon import Trainer, ParameterDict
from mxnet.gluon.loss import L1Loss, Loss, SigmoidBCELoss
from mxnet.gluon.nn import Activation, BatchNorm, Conv2D, Conv2DTranspose, Dropout, HybridBlock, HybridSequential, LeakyReLU, Dense
from mxnet.ndarray import NDArray, concat, full, mean, random_normal, zeros, stack, reshape
from mxnet.gluon.model_zoo import vision
import mxnet as mx

class Lossfun:

    def __init__(self, alpha: float, beta_vgg:float, beta_pix: float, context=None) -> None:
        self._alpha = alpha
        self._bce = SigmoidBCELoss()
        self._beta_vgg = beta_vgg
        self._beta_pix = beta_pix
        self._l1 = L1Loss()
        self._vgg = VggLoss(context)
        
    def __call__(self, p: float, p_hat: NDArray, y: NDArray, y_hat: NDArray) -> NDArray:
        
        dis_loss = self._alpha * mean(self._bce(p_hat, full(p_hat.shape, p))) 
        
        gen_loss_vgg = self._beta_vgg * mean(self._vgg(y_hat, y))
        gen_loss_pix = self._beta_pix * mean(self._l1(y_hat, y))
        
        return dis_loss + gen_loss_vgg + gen_loss_pix
                                             
    
    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def bce(self) -> Loss:
        return self._bce

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def l1(self) -> Loss:
        return self._l1
    
    
class VggLoss():
    
    def __init__(self, context) -> None:        
        self.vgg19=vision.vgg19(pretrained=True, ctx = context)
        self.vgg_layer = 22
        self._l1 = L1Loss()

        
    def __call__(self, y_hat, y):
        target_224 = mx.nd.contrib.BilinearResize2D(y, height=224, width=224)
        g_out_224 = mx.nd.contrib.BilinearResize2D(y_hat, height=224, width=224)

        feat_target = self.vgg19.features[:self.vgg_layer](target_224)
        feat_out = self.vgg19.features[:self.vgg_layer](g_out_224)
                      
        return self._l1(feat_out, feat_target)  
        

class Layer(HybridBlock):
    def __init__(self) -> None:
        super(Layer, self).__init__()

    @property
    def count(self) -> int:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        raise NotImplementedError

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        raise NotImplementedError


class Identity(Layer):
    def __init__(self, count: int, depth: int) -> None:
        super(Identity, self).__init__()

        self._count = count
        self._depth = depth

    @property
    def count(self) -> int:
        return self._count

    @property
    def depth(self) -> int:
        return self._depth

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        return x


class Skip(Layer):
    def __init__(self, count: int, depth: int, layer: Layer) -> None:
        super(Skip, self).__init__()

        with self.name_scope():
            self._block = HybridSequential()

            self._block.add(Conv2D(layer.depth, 4, 2, 1, use_bias=False, in_channels=depth))
            self._block.add(BatchNorm(momentum=0.1, in_channels=layer.depth))
            self._block.add(LeakyReLU(0.2))
            self._block.add(layer)
            self._block.add(Conv2DTranspose(count, 4, 2, 1, use_bias=False, in_channels=layer.count))
            self._block.add(BatchNorm(momentum=0.1, in_channels=count))

        self._count = count
        self._depth = depth
        self._layer = layer

    @property
    def block(self) -> HybridSequential:
        return self._block

    @property
    def count(self) -> int:
        return self._count + self._depth

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def layer(self) -> Layer:
        return self._layer

    def hybrid_forward(self, f: Any, x: NDArray, **kwargs) -> NDArray:
        return f.relu(f.concat(x, self._block(x), dim=1))


class RFLayer(HybridBlock):
    def __init__(self, RF_in_units, output_shape=(96,96)) -> None:
        super(RFLayer, self).__init__()
        
        self.output_shape = output_shape
        
        self._RFmappers = []
        with self.name_scope():
            for in_units in RF_in_units:
                RFmapper = Dense(output_shape[0]*output_shape[1], in_units=in_units)
                self.register_child(RFmapper)
                self._RFmappers.append(RFmapper)
        
        for param in self.collect_params().values():
            param.initialize()
            if "bias" in param.name:
                param.set_data(zeros(param.data().shape))
            elif "gamma" in param.name:
                param.set_data(random_normal(1, 0.02, param.data().shape))
            elif "weight" in param.name:
                param.set_data(random_normal(0, 0.02, param.data().shape))
    
    def forward(self, signals) -> NDArray:
        outputs = []
        for signal, RFmapper in zip(signals, self._RFmappers):
            for i in range(signal.shape[1]):
                outputs.append(RFmapper(signal[:,i,:]))
        outputs = stack(*outputs, axis=1)
        outputs = reshape(outputs, (outputs.shape[0], outputs.shape[1], *self.output_shape))
        return outputs
            
class Network(HybridSequential):
    def __init__(self, count: int, depth: int) -> None:
        super(Network, self).__init__()

        self._count = count
        self._depth = depth

        with self.name_scope():           
            self.add(Conv2D(64, 4, 2, 1, in_channels=depth))
            self.add(LeakyReLU(alpha=0.2))

            layer = Identity(512, 512)
            layer = Skip(512, 512, layer)

            for _ in range(0):
                layer = Skip(512, 512, layer)

                layer.block.add(Dropout(0.5))

            layer = Skip(256, 256, layer)
            layer = Skip(128, 128, layer)
            layer = Skip(64, 64, layer)

            self.add(layer)
            self.add(Conv2DTranspose(count, 4, 2, 1, in_channels=128))
            self.add(Activation("sigmoid"))

        for param in self.collect_params().values():
            param.initialize()
            if "bias" in param.name:
                param.set_data(zeros(param.data().shape))
            elif "gamma" in param.name:
                param.set_data(random_normal(1, 0.02, param.data().shape))
            elif "weight" in param.name:
                param.set_data(random_normal(0, 0.02, param.data().shape))

    @property
    def count(self) -> int:
        return self._count

    @property
    def depth(self) -> int:
        return self._depth

class Generator_prediction:
    def __init__(self, input_channels, context, RF_in_units, conv_input_shape=(96,96)) -> None:
        self.rf_mapper = RFLayer(RF_in_units, conv_input_shape)
        self._network = Network(3, input_channels)

    @property
    def network(self) -> HybridSequential:
        return self._network

    
class Generator:
    def __init__(self, input_channels, context, RF_in_units, conv_input_shape=(96,96), train_RF=False) -> None:
        self._train_RF = train_RF
        self._lossfun = Lossfun(alpha= 1, beta_vgg=100, beta_pix= 1, context=context)
        
        self._rf_mapper = RFLayer(RF_in_units, conv_input_shape)
        
        self._network = Network(3, input_channels)
        params_to_train = ParameterDict()
        if train_RF:
            params_to_train.update(self._rf_mapper.collect_params())
        params_to_train.update(self._network.collect_params())
        self._trainer = Trainer(params_to_train, "adam", {"beta1": 0.5, "learning_rate": 0.0002})
        
    @property
    def rf_mapper(self) -> HybridBlock:
        return self._rf_mapper
        
    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> HybridSequential:
        return self._network

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    def train(self, d: HybridSequential, to_learn_x: List[NDArray], y: NDArray) -> float:
        if not self._train_RF:
            learned_RF = self._rf_mapper(to_learn_x)
#             merged_RF = learned_RF + brain_RF 
        with autograd.record():
            if self._train_RF:
                learned_RF = self._rf_mapper(to_learn_x)
#                 merged_RF = learned_RF + brain_RF 
            loss = (lambda y_hat: self.lossfun(1, d(concat(learned_RF, y_hat, dim=1)), y, y_hat))(self._network(learned_RF))

        loss.backward()
        self.trainer.step(1)

        return float(loss.asscalar())

    
class Generator_prediction:
    def __init__(self, input_channels, context, RF_in_units, conv_input_shape=(96,96)) -> None:
        self._rf_mapper = RFLayer(RF_in_units, conv_input_shape)
        self._network = Network(3, input_channels)

    @property
    def network(self) -> HybridSequential:
        return self._network

    