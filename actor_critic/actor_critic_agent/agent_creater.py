#エージェントを生成する
import json

import chainer
from chainer import cuda, Function, gradient_check,\
                    Variable, optimizers, serializers, utils, iterators,\
                    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L


from actor_critic_agent.agent import Agent
from actor_critic_agent.actor_critic_model import ActorCriticModel

from actor_critic_agent.memory import Memory


#Agentを生成する
class AgentCreater:
    def __init__(self, filename):
        with open(filename, 'r') as file :
            self._parameters = json.load(file)
        #with
    #def

    def create(self, v_model, pi_model):
        parameters = self._parameters
        #報酬の割引率γ
        gamma = parameters["gamma"]

        #モデル
        model = ActorCriticModel(v_model, pi_model)

        #optimizerを生成する
        optimizer_parameters= (
                                parameters["optimizer"]["alpha"],
                                parameters["optimizer"]["eps"]
                            )

        optimizer = optimizers.Adam(alpha=optimizer_parameters[0], eps=optimizer_parameters[1])
        optimizer.setup(model)

        #lossを計算するときの重み
        V_loss_coef = parameters["V_loss_coef"]
        pi_loss_coef = parameters["pi_loss_coef"]
        entropy_coef = parameters["entropy_coef"]

        #経験
        memory_size = parameters["memory_size"] 
        memory = Memory(memory_size)


        #バッチ学習に使うデータの数
        batch_size = parameters["batch_size"]

        #agentを生成する
        agent = Agent(
                    gamma,
                    model,
                    optimizer,                        
                    V_loss_coef, 
                    pi_loss_coef, 
                    entropy_coef,
                    memory, 
                    batch_size
                )
 
        return agent
    #def
#def
