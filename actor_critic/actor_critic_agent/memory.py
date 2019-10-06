import pickle
import numpy as np

from actor_critic_agent.agent_exception import AgentException

import random

#経験を蓄える
class Memory:
    def __init__(self, memory_size):
        #状態
        self._State = None

        #行動
        self._Action = np.zeros(shape=memory_size, dtype=np.float32)

        #報酬
        self._Reward = np.zeros(shape=memory_size, dtype=np.float32)

        #行動後の状態
        self._State_dash = None

        #エピソード終端か？
        self._Done = np.zeros(shape=memory_size, dtype=np.bool)

        self._count = 0
        self._maxlen = memory_size
    #def

    #経験を追加する
    def append(self, state, action, reward, state_dash, done):
        if self._State is None:
            shape = state.shape
            self._State = np.zeros(shape=(self._maxlen,)+shape, dtype=np.float32)
        #if

        if self._State_dash is None:
            shape = state_dash.shape
            self._State_dash = np.zeros(shape=(self._maxlen,)+shape, dtype=np.float32)
        #if

        if self._count == self._maxlen:
            raise AgentException(self.__class__)
        #if
        
        i = self._count
        
        self._State[i] = state
        self._Action[i] = action
        self._Reward[i] = reward
        self._State_dash[i] = state_dash
        self._Done[i] = done

        self._count +=1
        return
    #def

    def clear(self):
        self._count = 0
        return
    #def

    #経験を参照する
    def refer(self):
        #読み取り専用のviewを取得する
        def immutable_view(X):
            view = X.view()
            view.setflags(write=False)
            return view
        #def

        count = len(self)

        State  = immutable_view(self._State[:count])
        Action = immutable_view(self._Action[:count])
        Reward = immutable_view(self._Reward[:count])
        State_dash = immutable_view(self._State_dash[:count])
        Done   = immutable_view(self._Done[:count])

        return State, Action, Reward, State_dash, Done
    #def

    #状態を記録する
    def create_memento(self):
        return pickle.dumps(self.__dict__)
    #def

    #記録してあった状態に巻き戻す
    def set_memento(self, memento):
        self.__dict__ = pickle.loads(memento)
        return
    #def


    def __len__(self):
        return self._count
    #def


    @property
    def maxlen(self):
        return self._maxlen
    #def

#class
