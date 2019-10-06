#エージェントを作成する
import copy
import json

import chainer
from chainer import cuda, Function, gradient_check,\
                    Variable, optimizers, serializers, utils, iterators,\
                    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L

from agent.dqn_agent import DQNAgent
from agent.memory import Memory
from agent.policy import Greedy, Random, DecayEpsilonPolicy
from agent.model_updater import SoftModelUpdater


#Agentを生成する
class DQNAgentCreater:
    def __init__(self, filename):
        with open(filename, 'r') as file :
            self._parameters = json.load(file)
        #with
    #def

    def create(self, model):
        parameters = self._parameters

        #行動の数
        actions_count = 2

        #方策(ここでいう方策とは、greedyかε-greedyということ)
        policies = self._create_policies(model, parameters, actions_count)
    
        #経験を記憶する
        memory = Memory(parameters["memory_size"])

        #割引率γ
        gamma = parameters["gamma"]

        #replay_start_memory_size個のデータが貯まるまで学習を開始しない
        replay_start_memory_size =  parameters["replay_start_memory_size"]

        #experience_replayするときのデータ数
        replay_count = parameters["replay_count"]

        #学習する間隔を少し開ける
        training_interval_steps = parameters["training_interval_steps"]


        #optimizerを生成する
        optimizer_parameters = (parameters["optimizer"]["alpha"], parameters["optimizer"]["epsilon"])
        optimizer = optimizers.Adam(alpha=optimizer_parameters[0], eps=optimizer_parameters[1])
        optimizer.setup(model)

        #モデルを更新する処理
        model_updater = SoftModelUpdater(parameters["tau"])

        #agent生成
        agent = DQNAgent(
                        gamma,

                        model,
                        optimizer,
                        model_updater,

                        memory,   
                        replay_start_memory_size, 
                        replay_count, 

                        training_interval_steps,

                        policies
                    )
 
        return agent
    #def
    
    def _create_policies(self, model, parameters, actions_count):
 
        policies = {}

        #random
        random = Random(actions_count)
        policies["Random"] = random


        #greedy
        greedy = Greedy(model) 
        policies["Greedy"] = greedy

        #ε-greedy    
        policy_A = Greedy(model)
        policy_B = Random(actions_count)

        e_greedy_parameters = (
                                parameters["epsilon"]["upper"], 
                                parameters["epsilon"]["lower"], 
                                parameters["epsilon"]["decay"],
                                parameters["epsilon"]["delay"]
                            )

        e_greedy = DecayEpsilonPolicy(policy_A, policy_B, e_greedy_parameters)

        policies["e-Greedy"] = e_greedy
        return policies
    #def
 
 #def