import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils, iterators,\
    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

#最適な行動価値関数を近似する
class QNet(Chain):
    def __init__(self):
        super(QNet, self).__init__(
                                        _hidden_layers = ChainList(
                                            L.Linear(None, 64),
                                            L.Linear(None, 64),
                                            L.Linear(None, 32),
                                        ),
                                        _output_layer = L.Linear(None, 2)
                                    )

    #def

    def __call__(self, x):
        h = x
        for layer in self._hidden_layers:
            h = F.relu(layer(h))
        #for
        q = self._output_layer(h)
        return q
    #def
#class
