import numpy as np

#(1-ε)の確率で方策A, εの確率で方策Bに従い行動を決定する。εが漸減する
class DecayEpsilonPolicy:
    def __init__(self, policy_A, policy_B, epsilon_parameters):
        self._epsilon_upper = epsilon_parameters[0]
        self._epsilon_lower = epsilon_parameters[1]
        self._epsilon_decay = epsilon_parameters[2]
        self._delay         = epsilon_parameters[3]

        self._policy_A = policy_A
        self._policy_B = policy_B

        self._epsilon = self._epsilon_upper
        self._count = 0
        #if-else
    #def


    def action(self, s):
        # numpyのrandは[0,1)で乱数を生成する
        if np.random.rand() < self._epsilon:
            a = self._policy_B.action(s)
        else:
            a = self._policy_A.action(s)
        #if-else

        return a
    #def

    def decay(self):
        if self._count >= self._delay:
            self._epsilon = max(self._epsilon-self._epsilon_decay, self._epsilon_lower)
        #if
        self._count += 1
        return
    #def

    @property
    def epsilon(self):
        return self._epsilon
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
