import gym
from gym import wrappers
import numpy as np
from datetime import datetime

#CartPoleの環境
class Environment:
    def __init__(self, monitor=False):
        env = gym.make('CartPole-v0')
        #動画を残したい場合
        if monitor:
            directory = "./replay/log/" + datetime.now().strftime('%Y%m%d%H%M%S')
            env = wrappers.Monitor(env, directory, video_callable = lambda x:True)
        #if
        self._env = env

        #envの状態をキャッシュする
        self._env_state = None

        self._steps = 0
        self._max_steps = 200
    #def

    #リセット
    def reset(self,):
        self._env_state = self._env.reset()
        self._steps=0

        state = self.observe()
        return state
    #def

    #行動を実行する
    def step(self, action):
        (env_state_dash, reward, done, info) = self._env.step(int(action))

        self._env_state = env_state_dash

        self._steps +=1

        #MaxStep経過した場合でもdoneにしない
        if done and self._steps == self._max_steps:
            done = False
        #if

        #報酬をスケーリングする
        reward /=20

        state_dash = self.observe()
        return (state_dash, reward, done, info)
    #def

    #状態を観測する
    def observe(self, normalize=True):
        state = self._env_state.copy().astype(np.float32)

        #正規化する
        if normalize == True:
            state[0] = state[0]/2.4
#            state[1] = state[1]/1.
            state[2] = state[2]/12.0
#            state[3] = state[3]/1.
        #if

        return state
    #def

    #描画する
    def render(self):
        self._env.render()
        return
    #def

    #終了処理
    def close(self):
        self._env.close()
    #def
    
    @property
    def max_steps(self):
        return self._max_steps
    #def

#class
