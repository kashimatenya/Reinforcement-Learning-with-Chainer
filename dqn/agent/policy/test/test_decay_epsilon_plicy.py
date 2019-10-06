#一部手動テストあり

import sys
import os
import unittest
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from dqn_agent.policy import DecayEpsilonPolicy

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
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
        epsilon_parameters = [1,0, 1, 1]

        expected_policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        del expected_policy._epsilon
        del expected_policy._policy_A
        del expected_policy._policy_B

        del expected_policy._epsilon_upper
        del expected_policy._epsilon_lower
        del expected_policy._epsilon_decay
        del expected_policy._delay

        expected_policy._epsilon = epsilon_parameters[0]
        expected_policy._epsilon_upper = epsilon_parameters[0]
        expected_policy._epsilon_lower = epsilon_parameters[1]
        expected_policy._epsilon_decay = epsilon_parameters[2]
        expected_policy._delay         = epsilon_parameters[3]
        expected_policy._policy_A = policy_A
        expected_policy._policy_B = policy_B

        actual_policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def


    #setupをテストする
    def test_002_001(self):
        epsilon_parameters = [1, 0, 1, 1]

        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
        model = 'model'

        actual_policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        expected_policy = deepcopy(actual_policy)

        expected_policy._policy_A.setup(model)
        expected_policy._policy_B.setup(model)
        
        actual_policy.setup(model)
        
        self.assertEqual(actual_policy, expected_policy)
        return
    #def


    def test_003_001(self):
        epsilon_parameters = [1 ,0, 1, 1]
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
    
        policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        expected_policy = deepcopy(policy)
        expected_policy._count = 1

        s = np.ndarray(1)
        _ = policy.action(s)

        actual_policy = policy
        self.assertEqual(actual_policy, expected_policy)
        return
    #def

    def test_003_002(self):
        epsilon_parameters = [1 ,0, 1, 1]
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
    
        policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        expected_policy = deepcopy(policy)
        expected_policy._count = 2
        expected_policy._epsilon = epsilon_parameters[1]

        s = np.ndarray(1)
        _ = policy.action(s)
        _ = policy.action(s)

        actual_policy = policy
        self.assertEqual(actual_policy, expected_policy)
        return
    #def

    def test_003_003(self):
        epsilon_parameters = [1 ,0, 1, 1]
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')
    
        policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        expected_policy = deepcopy(policy)
        expected_policy._count = 3
        expected_policy._epsilon = epsilon_parameters[1]

        s = np.ndarray(1)
        _ = policy.action(s)
        _ = policy.action(s)
        _ = policy.action(s)

        actual_policy = policy
        self.assertEqual(actual_policy, expected_policy)
        return
    #def

    def test_004_001(self):
        epsilon_parameters = [1 ,0, 0.01, 100]
    
        policy_A = DummyPolicy('A')
        policy_B = DummyPolicy('B')

        policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

        expected_epsilon = policy._epsilon
        actual_epsilon = policy._epsilon

        self.assertEqual(actual_epsilon, expected_epsilon)
        return
    #def


#def

#class

# actionをテストする. 手動テスト
def test_003_004():
    #epsilonを色々変えてAとBの割合を見る
    epsilon_parameters = [0.2 ,0, 0, 1]
    
    policy_A = DummyPolicy('A')
    policy_B = DummyPolicy('B')
    
    policy = DecayEpsilonPolicy(policy_A, policy_B, epsilon_parameters)

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

        test_003_004()
    #if
    


else:
    Warning('manual test did not run')
#if-else