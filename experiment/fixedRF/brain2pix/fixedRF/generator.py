from typing import Any

from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L1Loss, Loss, SigmoidBCELoss
from mxnet.gluon.nn import Activation, BatchNorm, Conv2D, Conv2DTranspose, Dropout, HybridBlock, HybridSequential, \
    LeakyReLU
from mxnet.ndarray import NDArray, concat, full, mean, random_normal, zeros


class Lossfun:
    def __init__(self, alpha: float, beta: float) -> None:
        self._alpha = alpha
        self._bce = SigmoidBCELoss()
        self._beta = beta
        self._l1 = L1Loss()
        

    def __call__(self, p: float, p_hat: NDArray, y: NDArray, y_hat: NDArray) -> NDArray:
        return self._alpha * mean(self._bce(p_hat, full(p_hat.shape, p))) + self._beta * mean(self._l1(y_hat, y))

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


class Generator:
    def __init__(self, input_channels) -> None:
        self._lossfun = Lossfun(1, 100)
        self._network = Network(3, input_channels)
        self._trainer = Trainer(self._network.collect_params(), "adam", {"beta1": 0.5, "learning_rate": 0.0002})

    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> HybridSequential:
        return self._network

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    def train(self, d: HybridSequential, x: NDArray, y: NDArray) -> float:
        with autograd.record():
            loss = (lambda y_hat: self.lossfun(1, d(concat(x, y_hat, dim=1)), y, y_hat))(self._network(x))

        loss.backward()
        self.trainer.step(1)

        return float(loss.asscalar())
