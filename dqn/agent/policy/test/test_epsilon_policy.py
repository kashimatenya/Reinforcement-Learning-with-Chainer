#一部手動テストあり

import sys
import os
import unittest
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from dqn_agent.policy import EpsilonPolicy

import numpy as np


class DummyPolicy:
    def __init__(self, value):
        self._value = value
        self._model = None
    #def

    def setup(self, model):
        self._model = model
        return
    #def

    def action(self, s):
        return self._value
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

class TestEpsilonPolicy(unittest.TestCase):
    #__init__をテストする
    def test_001_001(self):
        epsilon =0.5

        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')

        expected_policy = EpsilonPolicy(policy_A, policy_B, epsilon)

        del expected_policy._epsilon
        del expected_policy._policy_A
        del expected_policy._policy_B

        expected_policy._epsilon = epsilon
        expected_policy._policy_A = policy_A
        expected_policy._policy_B = policy_B

        actual_policy = EpsilonPolicy(policy_A, policy_B, epsilon)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def


    #setupをテストする
    def test_002_001(self):
        epsilon = 0.5

        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
        model = 'model'

        actual_policy = EpsilonPolicy(policy_A, policy_B, epsilon)

        expected_policy = deepcopy(actual_policy)

        expected_policy._policy_A.setup(model)
        expected_policy._policy_B.setup(model)
        
        actual_policy.setup(model)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def

    def test_004_001(self):
        epsilon = 0.5
    
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')

        policy = EpsilonPolicy(policy_A, policy_B, epsilon)

        expected_epsilon = policy._epsilon
        actual_epsilon = policy._epsilon

        self.assertEqual(actual_epsilon, expected_epsilon)
        return
    #def

#class

# actionをテストする. 手動テスト
def test_003_001():
    #epsilonを色々変えてAとBの割合を見る
    epsilon = 0.1
    
    policy_A = DummyPolicy('A')
    policy_B = DummyPolicy('B')
    
    policy = EpsilonPolicy(policy_A, policy_B, epsilon)

    Repeat = int(1e4)

    result = []
    s = np.ndarray(1)
    for _ in range(Repeat):
        a = policy.action(s)
        result.append(a)
    #for

    count_A = result.count('A')
    count_B = result.count('B')
    print('')
    print('count of A:'+str(count_A))
    print('count of B:'+str(count_B))

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