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
                hidden_layers = ChainList(
                                    L.Linear(None, 32),
                                    L.Linear(None, 32),
                                    L.Linear(None, 16),
                                ),
                output_layer = L.Linear(None, 1)
            )
    #def

    def __call__(self, x):
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        #for
        v = self.output_layer(h)
        return v
    #def
#class


#方策
class PolicyNet(Chain):
    def __init__(self):
        super(PolicyNet, self).__init__(
                hidden_layers = ChainList(
                                    L.Linear(None, 32),
                                    L.Linear(None, 32),
                                    L.Linear(None, 16),
                                ),
                output_layer = L.Linear(None, 2)
            )
    #def

    def __call__(self, x):
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        #for
        p = F.softmax(self.output_layer(h))
        return D.Categorical(p)
    #def
#class
