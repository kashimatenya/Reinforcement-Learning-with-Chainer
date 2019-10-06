#学習済みのCartPoleのモデルを動かしてみる
from cartpole import Environment
from cartpole.models import QNet

from agent.dqn_agent_creater import DQNAgentCreater

import time

max_episodes = int(1e4+1)

#動画を残すか？
monitor = False

#環境を作成
environment = Environment(monitor=monitor)

#モデルを作成
model = QNet()

#エージェントを作成
parameters_filename = "parameters.json"
creater = DQNAgentCreater(parameters_filename)
agent = creater.create(model)
agent.set_policy("Greedy")

filename = "replay/replay_cartpole.model"
agent.load(filename, None)

max_steps = environment.max_steps

#リプレイ
for t in range(1, max_episodes+1):
    state = environment.reset()
    environment.render()

    total_reward = 0

    for step in range(1, max_steps+1):        
        action = agent.action(state)
        (state_dash, reward, done, _) = environment.step(action)

        environment.render()

        total_reward += reward

        state = state_dash 

        #エピソード終端か？
        if done:
            break
        #if

        time.sleep(0.05)
    #for

    print("episode:" + str(t)  +", steps:"+ str(step), ", total reward:" +str(total_reward))
    time.sleep(0.5)
#for

environment.close()
