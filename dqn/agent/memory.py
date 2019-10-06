'''経験(s,a,s',r,done)の組を記憶する'''

import pickle
import numpy as np

from utils.ringed_index import RingedIndex
import random


def _immutable_view(X):
    view = X.view()
    view.setflags(write=False)
    return view
#def


#経験を蓄える
class Memory:
    def __init__(self, memory_size):
        self._State = None
        self._Action = np.zeros(shape=memory_size,dtype=np.int32)
        self._Reward = np.zeros(shape= memory_size, dtype=np.float32)
        self._State_dash = None
        self._Done = np.zeros(shape= memory_size, dtype=np.bool)

        self._count = 0
        self._maxlen = memory_size

        self._index = RingedIndex(0, memory_size)
    #def

    #経験を追加する
    def append(self, state, action, reward, state_dash, done):
        if self._State is None:
            #入力されたstateのshapeに合わせて配列を確保する
            shape = state.shape
            self._State = np.zeros(shape=(self._maxlen,)+shape, dtype=np.float32)
        #if

        if self._State_dash is None:
            #入力されたstate_dashのshapeに合わせて配列を確保する
            shape = state_dash.shape
            self._State_dash = np.zeros(shape=(self._maxlen,)+shape, dtype=np.float32)
        #if    

        #末尾のindex
        i = self._index.next()

        self._State[i] = state
        self._Action[i] = action
        self._Reward[i] = reward
        self._State_dash[i] = state_dash
        self._Done[i] = done

        if self._count < self._maxlen:
            self._count += 1
        #if
        return
    #def

    #経験の一部を抽出する
    def sample(self, k):
        count = self._count

        index = random.sample(range(count), k) 

        State  = _immutable_view(self._State[index])
        Action = _immutable_view(self._Action[index])
        Reward = _immutable_view(self._Reward[index])
        State_dash = _immutable_view(self._State_dash[index])
        Done = _immutable_view(self._Done[index])
        
        return (State, Action, Reward, State_dash, Done) 
    #end

    #経験の全部を参照する
    def refer(self):
        count = self._count

        State  = _immutable_view(self._State[:count])
        Action = _immutable_view(self._Action[:count])
        Reward = _immutable_view(self._Reward[:count])
        State_dash = _immutable_view(self._State_dash[:count])
        Done = _immutable_view(self._Done[:count])

        return (State, Action, Reward, State_dash, Done) 
    #end

    #クリアする
    def clear(self):
        self._index.reset()
        self._count = 0
        return
    #def

    def create_memento(self):
        return pickle.dumps(self.__dict__)
    #def

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

    def __eq__(self, other):
        if other is None or type(self) != type(other):
            return False
        #if

        def arrays_are_equal(a,b) : 
            return (a==b).all()
        #def
        
        if not arrays_are_equal(self._State, other._State):
            return False
        #if

        if not arrays_are_equal(self._Action, other._Action):
            return False
        #if

        if not arrays_are_equal(self._Reward, other._Reward):
            return False
        #if

        if not arrays_are_equal(self._State_dash, other._State_dash):
            return False
        #if

        if not  arrays_are_equal(self._Done, other._Done):
            return False
        #if

        if self._count != other._count:
            return False
        #if

        if self._maxlen != other._maxlen:
            return False
        #if
        if self._index != other._index:
            return False
        #if

        return True
    #def

    def __ne__(self, other):
        return not self.__eq__(other)
    #def
#class
