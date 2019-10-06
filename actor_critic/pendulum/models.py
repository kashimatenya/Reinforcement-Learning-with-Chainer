import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils, iterators,\
    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L
import chainer.distributions as D

#価値関数
class ValueNet(Chain):
    def __init__(self):
        super(ValueNet, self).__init__(
                _hidden_layers = ChainList(
                                    L.Linear(None, 32),
                                    L.Linear(None, 32),
                                    L.Linear(None, 16),
                                ),
                _output_layer = L.Linear(None, 1)
            )
    #def

    def __call__(self, x):
        h = x
        for layer in self._hidden_layers:
            h = F.relu(layer(h))
        #for
        v = self._output_layer(h)
        return v
    #def
#class


#方策
class PolicyNet(Chain):
    def __init__(self):
        super(PolicyNet, self).__init__(
                _hidden_layers = ChainList(
                                    L.Linear(None, 32),
                                    L.Linear(None, 32),
                                    L.Linear(None, 16),
                                ),
                _output_layer = L.Linear(None, 2)
            )

        self._eps = 1e-5 #sigma=0を避けるための微小数
    #def

    def __call__(self, x):
        h = x
        for layer in self._hidden_layers:
            h = F.relu(layer(h))
        #for
        theta = F.sigmoid(self._output_layer(h))

        mu    = 4*theta[:, 0]-2
        sigma = 0.5*theta[:, 1]+self._eps
        return D.Normal(mu, sigma)
    #def
#class
