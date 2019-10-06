#一部手動テストあり
import sys
import os
import unittest
from copy import deepcopy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from dqn_agent.policy import Random

import numpy as np

class DummyModel:
    def __init__(self, output_shape):
        self.output_shape = output_shape
    #def
#class

class TestRandom(unittest.TestCase):
    #__init__をテストする
    def test_001_001(self):
        expected_random = Random()
        del expected_random._actions_count
        expected_random._actions_count = 0

        actual_random = Random()
        
        self.assertEqual(actual_random, expected_random)

        return
    #def

    #setupをテストする
    def test_002_001(self):
        shape = (10, 1)
        model = DummyModel(shape)

        actual_random = Random()

        expected_random = deepcopy(actual_random)
        expected_random._actions_count = shape[0]
        
        actual_random.setup(model)
        
        self.assertEqual(actual_random, expected_random)
        return
    #def

#class

#actionを手動テストする. 目視確認
def test_003_001():
    shape = (10, 1)
    model = DummyModel(shape)
    Repeat = int(1e4)

    random = Random()
    random.setup(model) 

    result = []
    for _ in range(Repeat):
        a = random.action(None)
        result.append(a)
    #for

    #分布を目視で確認する
    plt.hist(result, bins=shape[0], normed=True) 
    plt.show() 
    print ("")
    print ("min:" + str(min(result)))
    print ("max:" + str(max(result)))
        
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
    #if

    test_003_001()

else:
    Warning('manual test did not run')
#if-esle