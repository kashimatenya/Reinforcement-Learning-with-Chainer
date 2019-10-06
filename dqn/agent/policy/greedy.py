import numpy as np
import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils, iterators,\
    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

#greedy法
class Greedy:
    def __init__(self, model):
        self._model = model
    #def

    def action(self, s):
        with chainer.no_backprop_mode():
            q = self._model(s.reshape((1,) + s.shape)).array
        #with
        return np.argmax(q)
    #def

    def __eq__(self, other):
        if other is None or type(self) != type(other):
            return False
        #if
        return self.__dict__ == other.__dict__
    #def

    def __ne__(self, other):
        return not self.__eq__(other)
    #def

#class
