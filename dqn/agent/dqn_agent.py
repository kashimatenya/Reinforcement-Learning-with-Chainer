'''エージェント'''
import numpy as np
from copy import deepcopy
import pickle
from collections import deque

import chainer
from chainer import cuda, Function, gradient_check,\
                    Variable, optimizers, serializers, utils, iterators,\
                    Link, Chain, ChainList

import chainer.functions as F
import chainer.links as L


#強化学習するエージェント
class DQNAgent:
    def __init__(
                self, 
                gamma,          #報酬の割引率
                model,          #最適行動価値関数のモデル
                optimizer,      #モデルのoptimizer
                model_updater,  #target_modelを更新するルール
                memory,         #経験を貯める
                replay_start_memory_size, #学習を開始する経験の数
                replay_count,   #experience_replyするときの経験の数
                training_interval_steps, #間隔をあけて学習する
                policies        #方策(random/greedy/ε-greedy)
            ):


        #報酬の割引率
        self._gamma = gamma


        #最適な行動価値関数を近似するためのモデル
        self._model = model

        #ターゲットモデル
        self._target_model = self._model.copy('copy')

        #optimizer
        self._optimizer = optimizer

        #ターゲットモデルを更新するときのルール
        self._model_updater = model_updater


        #経験を蓄える
        self._memory = memory

        #経験がreplay_start_memory_size個たまるまで学習を開始しない
        self._replay_start_memory_size = replay_start_memory_size

        #experience replayするときのデータの数
        self._replay_count = replay_count

        #training_interval_stepsの間隔をあけて学習する
        self._training_interval_steps = training_interval_steps

        #間隔をあけて学習するためのカウンター
        self._experience_count = 0

        #方策たち。  ここでいう方策はrandomかgreedyかε-greedyかという意味
        self._policies = policies

        #現在の方策
        self._policy = None
    #def

    #次の行動を決める
    def action(self, state, greedy=False):        
        action = self._policy.action(state)
        return action
    #def

    #経験を記憶する
    def gain_experience(self, state, action, reward, state_dash, done, info):
        self._memory.append(state, action, reward, state_dash, done)
        self._experience_count += 1
        return
    #def

    #経験を元に学習する
    def train(self, estimate_loss=False):
        memoey_size = len(self._memory)

        if memoey_size < self._replay_start_memory_size:
            self._experience_count = 0
            return None
        #if

        model = self._model
        target_model = self._target_model
        gamma = self._gamma

        #experience replay
        count = min(self._replay_count, len(self._memory))
        (State, Action, Reward, State_dash, Done) = self._memory.sample(k=count) 

        #TD誤差を計算する
        #Q'は定数
        with chainer.no_backprop_mode():
            Q_dash = target_model(State_dash).array
        #with chainer.no_backprop_mode():

        Value_dash = np.max(Q_dash, axis=1)
        Value_dash[Done==True] = 0

        Q = model(State)

        TD_error = Reward +gamma*Value_dash -Q[range(count), Action]
        #※hurber_lossの関数を使うためにreshapeする
        TD_error = TD_error.reshape(-1,1) 
        
        zeros = np.zeros(TD_error.shape, dtype=np.float32)

        #損失
        loss = F.mean(F.huber_loss(TD_error, zeros, 1))

        #パラメータを更新する
        optimizer= self._optimizer
        model.cleargrads()
        loss.backward()
        optimizer.update()

        #target Q-netowrkを更新する
        self._model_updater(self._target_model, self._model)

        #訓練時の誤差を計算したい場合
        if estimate_loss:
            squared_TD_error = self._get_squared_TD_error(State, Action, Reward, State_dash, Done) 
            return squared_TD_error
        #if

        self._experience_count = 0
        return None
    #def

    #td誤差の2乗を計算する
    def estimate_squared_TD_error_from_experience(self, experience):
        (State, Action, Reward, State_dash, Done) = experience
        return self._get_squared_TD_error(State, Action, Reward, State_dash, Done)
    #def

    #td誤差の2乗を計算する
    def _get_squared_TD_error(self, State, Action, Reward, State_dash, Done):
        model = self._model
        target_model = self._target_model
        gamma = self._gamma

        count = State.shape[0]
        
        with chainer.no_backprop_mode():
            Q = model(State).array
            Q_dash = target_model(State_dash).array
        #with chainer.no_backprop_mode():

        Value_dash = np.max(Q_dash, axis=1)
        Value_dash[Done==True] = 0
        
        TD_error = Reward +gamma*Value_dash -Q[range(count), Action] 
        squared_TD_error = float(np.mean(np.square(TD_error)))

        return squared_TD_error
    #def

    #方策を変更する(ここでいう方策はrandomなのかgreedyなのかε-greedyなのかという意味)
    def set_policy(self, policy):
        self._policy = self._policies[policy]
        return
    #def

    #エピソード終了時の処理
    def end_episode(self):
        #εをエピソード単位で減衰させる
        self._policies["e-Greedy"].decay()
        return
    #def

    #学習の間隔は空いたか？
    def is_to_train(self):
        return self._experience_count >= self._training_interval_steps
    #def

    #モデルを保存する
    def save(self, model_filename, memory_filename):
        if model_filename is not None:
            serializers.save_npz(model_filename, self._model)
        #if

        if memory_filename is not None:
            with open(memory_filename, "wb") as file:
                pickle.dump(self._memory, file)
            #with
        #end
        return
    #def

    #モデルを読み込む
    def load(self, model_filename, memory_filename):
        if model_filename is not None:
            serializers.load_npz(model_filename, self._model)
            self._target_model = deepcopy(self._model)
        #if

        if memory_filename is not None:
            with open(memory_filename, "rb") as file:
                self._memory = pickle.load(file)
            #with
        #if
        return
    #def

#class
