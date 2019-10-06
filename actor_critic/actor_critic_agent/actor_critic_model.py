import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils, iterators,\
    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

#価値関数のモデルと方策関数のモデル
class ActorCriticModel(Chain):
    def __init__(self, V_model, pi_model):
        super(ActorCriticModel, self).__init__(
            _V_model = V_model,
            _pi_model = pi_model
        )
    #def

    #価値関数のモデル
    def V(self, x):
        return self._V_model(x)
    #def

    #方策関数のモデル
    def pi(self, x):
        return self._pi_model(x)
    #def    
#def