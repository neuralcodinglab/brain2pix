from typing import List

from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import Loss, SigmoidBCELoss
from mxnet.gluon.nn import BatchNorm, Conv2D, HybridSequential, LeakyReLU
from mxnet.ndarray import NDArray, concat, full, mean, random_normal, stack, zeros
from mxnet.ndarray.random import randint, uniform


class History:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data = []

    def __call__(self, z_prime: NDArray) -> NDArray:
        z = []

        for i in range(z_prime.shape[0]):
            if len(self._data) < self._capacity:
                z.append(z_prime[i])
                self._data.append(z_prime[i])
            elif uniform().asscalar() < 0.5:
                z.append(self._data.pop(randint(0, self._capacity).asscalar()))
                self._data.append(z_prime[i])
            else:
                z.append(z_prime[i])

        return stack(*z, axis=0)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def data(self) -> List[NDArray]:
        return self._data


class Lossfun:
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha
        self._bce = SigmoidBCELoss()

    def __call__(self, p: float, p_hat: NDArray) -> NDArray:
        return self._alpha * mean(self._bce(p_hat, full(p_hat.shape, p)))

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def bce(self) -> Loss:
        return self._bce


class Network(HybridSequential):
    def __init__(self, count: int, depth: int) -> None:
        super(HybridSequential, self).__init__()

        self._count = count
        self._depth = depth

        with self.name_scope():
            self.add(Conv2D(64, 4, 2, 1, in_channels=depth))
            self.add(LeakyReLU(0.2))
            self.add(Conv2D(128, 4, 2, 1, use_bias=False, in_channels=64))
            self.add(BatchNorm(momentum=0.1, in_channels=128))
            self.add(LeakyReLU(0.2))
            self.add(Conv2D(256, 4, 2, 1, use_bias=False, in_channels=128))
            self.add(BatchNorm(momentum=0.1, in_channels=256))
            self.add(LeakyReLU(0.2))
            self.add(Conv2D(512, 4, padding=1, use_bias=False, in_channels=256))
            self.add(BatchNorm(momentum=0.1, in_channels=512))
            self.add(LeakyReLU(0.2))
            self.add(Conv2D(count, 3, 2, padding=1, in_channels=512))

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


class Discriminator:
    def __init__(self, input_channels) -> None:
        self._history = History(50)
        self._lossfun = Lossfun(1)
        self._network = Network(1, input_channels+18)
        self._trainer = Trainer(self._network.collect_params(), "adam", {"beta1": 0.5, "learning_rate": 0.0002})

    @property
    def history(self) -> History:
        return self._history

    @property
    def lossfun(self) -> Lossfun:
        return self._lossfun

    @property
    def network(self) -> HybridSequential:
        return self._network

    @property
    def trainer(self) -> Trainer:
        return self._trainer

    def train(self, g: HybridSequential, x: NDArray, y: NDArray) -> float:
        z = self._history(concat(x, g(x), dim=1))

        with autograd.record():
            loss = 0.5 * (self.lossfun(0, self._network(z)) + self.lossfun(1, self._network(concat(x, y, dim=1))))

        loss.backward()
        self.trainer.step(1)

        return float(loss.asscalar())
