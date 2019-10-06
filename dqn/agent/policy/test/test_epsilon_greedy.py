#一部手動テストあり

import sys
import os
import unittest
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from dqn_agent.policy import EpsilonGreedy, Random, Greedy

import numpy as np


class DummyModel:
    def __init__(self, output_shape):
        self.input_shape = (1,1)
        self.output_shape = output_shape
    #def

    def __call__(self, s):
        return np.eye(N=1, M=self.output_shape[0], k=int(self.output_shape[0]/2))
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

class TestEpsilonGreedy(unittest.TestCase):
    #__init__をテストする
    def test_001_001(self):
        epsilon =0.5

        expected_policy = EpsilonGreedy(epsilon)

        del expected_policy._greedy
        del expected_policy._random
        del expected_policy._epsilon

        expected_policy._greedy = Greedy()
        expected_policy._random = Random()
        expected_policy._epsilon = epsilon

        actual_policy = EpsilonGreedy(epsilon)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def


    #setupをテストする
    def test_002_001(self):
        epsilon = 0.5
        shape = (10000,1)
        model = DummyModel(shape)

        actual_policy = EpsilonGreedy(epsilon)

        expected_policy = deepcopy(actual_policy)

        expected_policy._greedy.setup(model)
        expected_policy._random.setup(model)
        
        actual_policy.setup(model)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def

    def test_004_001(self):
        epsilon = 0.5
    
        policy = EpsilonGreedy(epsilon)

        expected_epsilon = policy._epsilon
        actual_epsilon = policy._epsilon

        self.assertEqual(actual_epsilon, expected_epsilon)
        return
    #def

#class

# actionをテストする. 手動テスト
def test_003_001():
    out_shape = (int(1e3), 1)
    #epsilonを色々変えて結果を確認する
    #εの高さのピークが立つはず
    epsilon = 0
#    epsilon = 1
    model = DummyModel(out_shape)
    Repeat = int(1e4)

    policy = EpsilonGreedy(epsilon)
    policy.setup(model)

    result = []
    s = np.ndarray(1)
    for _ in range(Repeat):
        a = policy.action(s)
        result.append(a)
    #for

    #分布を目視で確認する
    plt.hist(result, bins=1000, normed=True) 
    plt.show() 
        
    count = result.count(int(out_shape[0]/2))
    print(count)

    return
#def

#テスト実行
if __name__ == "__main__":

    unit_test_flag = True

    if unit_test_flag:
        try:
            unittest.main()
        except SystemExit as exception:
            pass
        #try-except

        test_003_001()
    #if
    


else:
    Warning('manual test did not run')
#if-else