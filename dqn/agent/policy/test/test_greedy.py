import sys
import os
import unittest
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from dqn_agent.policy import Greedy

import numpy as np


class DummyModel:
    def __init__(self, actions_count):
        self.input_shape = (1,1)
        self._actions_count = actions_count
        self.a = None
    #def

    def __call__(self, s):
        self.a = np.random.randint(0, self._actions_count)
        #１つめのインデックスはサンプル方向
        return np.eye(N=1, M=self._actions_count, k=self.a)
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

class TestGreedy(unittest.TestCase):
    #__init__をテストする
    def test_001_001(self):
        expected_greedy = Greedy()
        del expected_greedy._model
        expected_greedy._model = None
        actual_greedy = Greedy()
        
        self.assertEqual(actual_greedy, expected_greedy)
        return
    #def

    #setupをテストする
    def test_002_001(self):
        action_count = 100
        model = DummyModel(action_count)

        actual_random = Greedy()

        expected_random = deepcopy(actual_random)
        expected_random._model = model
        
        actual_random.setup(model)
        
        self.assertEqual(actual_random, expected_random)
        return
    #def

    def test_003_001(self):
        action_count = 100
        model = DummyModel(action_count)
        Repeat = int(1e3)

        greedy = Greedy()
        greedy.setup(model) 

        expected_greedy = deepcopy(greedy)
        expected_count = Repeat

        result = []
        s = np.ndarray(1, np.float32)
        for _ in range(Repeat):
            a = greedy.action(s)
            #正解したか？
            flag = (a == greedy._model.a)
            result.append(flag)
        #for

        #正解した回数を数える
        actual_count = result.count(True)
        
        actual_greedy = greedy
        expected_greedy._model = actual_greedy._model

        self.assertEqual(actual_count, expected_count)
        self.assertEqual(actual_greedy, expected_greedy)
        return
    #def
#class

#テスト実行
if __name__ == "__main__":

    unit_test_flag = True

    if unit_test_flag:
        try:
            unittest.main()
        except SystemExit as exception:
            pass
        #try-except
    #if
#if