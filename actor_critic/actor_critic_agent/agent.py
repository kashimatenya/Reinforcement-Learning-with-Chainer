import numpy as np
import pickle

import chainer
from chainer import cuda, Function, gradient_check,\
                    Variable, optimizers, serializers, utils, iterators,\
                    Link, Chain, ChainList, distributions

import chainer.functions as F
import chainer.links as L

#エージェントのクラス
class Agent:
    def __init__(
                self,
                gamma,          #報酬の割引率
                model,          #価値関数のモデルと方策モデル
                optimizer,      #optimizer
                V_loss_coef,    #損失計算に価値関数の誤差にかける重み
                pi_loss_coef,   #損失計算に方策関数の誤差にかける重み
                entropy_coef,   #損失計算に方策のエントロピーにかける重み
                memory,         #経験を蓄える
                batch_size      #バッチ学習に使う経験の数
            ):

        #報酬の割引率γ
        self._gamma = gamma

        #モデル(価値関数のモデルと方策のモデル)
        self._model = model

        #optimizer
        self._optimizer = optimizer
    
        #lossを計算するときの重み
        self._V_loss_coef = V_loss_coef
        self._pi_loss_coef = pi_loss_coef
        self._entropy_coef = entropy_coef

        #経験を蓄える
        self._memory = memory

        #1バッチ学習に使う経験の数
        self._batch_size = batch_size

    #def

    #動作を決定する
    def action(self, state):
        model = self._model

        pi = model.pi(state.reshape((-1,)+state.shape))
        action = float(pi.sample().array)

        return action
    #def

    #経験を記憶する
    def gain_experience(self, state, action, reward, state_dash, done, info):
        self._memory.append(state, action, reward, state_dash, done)
        return
    #def

    #経験は十分か？
    def is_experienced(self):
        return len(self._memory) >= self._batch_size
    #def

    #経験をリセットする
    def reset_experience(self):
        self._memory.clear()
        return
    #def

    #学習する
    def train(self, fix_policy= False, estimate_loss=False):
        memory = self._memory
        gamma = self._gamma
        V_loss_coef =self._V_loss_coef
        pi_loss_coef = self._pi_loss_coef
        entropy_coef = self._entropy_coef
        model = self._model

        #今までの経験
        (State, Action, Reward, State_dash, Done) = memory.refer()

        #方策評価
        Value  = model.V(State)
        Value_dash = model.V(State_dash).array
        Value_dash[Done==True] = 0
        Advantage = Reward.reshape(-1, 1) +gamma*Value_dash -Value

        V_loss = F.mean(F.square(Advantage))

        #方策改善
        Pi = model.pi(State)
        log_Prob = Pi.log_prob(Action)
        Entropy = Pi.entropy

        pi_loss = F.mean(-log_Prob*(Advantage.reshape((-1,)).array) -entropy_coef*Entropy)     
        

        #損失
        loss = V_loss_coef*V_loss +pi_loss_coef*pi_loss

        #パラメータを更新する
        model.cleargrads()
        loss.backward()
        optimizer = self._optimizer
        optimizer.update()
        
        #訓練誤差を計算したい場合
        if estimate_loss:
            (V_loss, pi_loss, pi_entropy) = self._get_loss(State, Action, Reward, State_dash, Done)
            return V_loss, pi_loss, pi_entropy
        #if
        
        return None, None, None
    #def

    #経験から誤差を計算する
    def estimate_loss_from_experience(self, experiences):
        (State, Action, Reward, State_dash, Done) = experiences
        return self._get_loss(State, Action, Reward, State_dash, Done)
    #def

    #各種誤差を計算する
    def _get_loss(self, State, Action, Reward, State_dash, Done):
        model = self._model
        
        gamma = self._gamma
        entropy_coef = self._entropy_coef
        
        with chainer.no_backprop_mode():
            Value = model.V(State).array

            Value_dash = model.V(State_dash).array
            Value_dash[Done==True] = 0
  
            Pi = model.pi(State)
            log_Prob = Pi.log_prob(Action).array
            Entropy =  Pi.entropy.array
        #with

        Advantage = Reward.reshape(-1, 1) +gamma*Value_dash -Value

        #Vの誤差
        V_loss = float(np.mean(np.square(Advantage)))

        #piの誤差
        pi_loss = float(np.mean(-log_Prob*Advantage.reshape((-1,)) -entropy_coef*Entropy))
        
        #平均のエントロピー
        pi_entropy = float(np.mean(Entropy))

        return V_loss, pi_loss, pi_entropy
    #def

    #モデルを保存する
    def save(self, model_filename):
        serializers.save_npz(model_filename, self._model)
        return
    #def

    #モデルを読み込む
    def load(self, model_filename):
        serializers.load_npz(model_filename, self._model)
        return
    #def
#class
