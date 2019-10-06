import numpy as np
import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils, iterators,\
    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

#完全ランダム
class Random:
    def __init__(self, actions_count):
        self._actions_count = actions_count
    #def

    def action(self, s):
        return np.random.randint(0, self._actions_count)
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
