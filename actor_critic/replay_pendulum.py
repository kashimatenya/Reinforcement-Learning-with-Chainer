#Pendulumの学習済みモデルを動かしてみる
from actor_critic_agent import AgentCreater

from pendulum import Environment
from pendulum.models import ValueNet, PolicyNet

import time

max_episodes = int(1e6)


#動画を残すか？
monitor = False
#環境を作成する
environment = Environment(monitor=monitor)

#モデルを作成する
v_model = ValueNet()
pi_model = PolicyNet()


#エージェントを作成する
parameters_filename = "pendulum_parameters.json"
model_filename = ".\\replay\\pendulum.model"

creater = AgentCreater(parameters_filename)
agent = creater.create(v_model, pi_model)
agent.load(model_filename)

#最大ステップ数
max_steps = environment.max_steps

#リプレイ
for t in range(1, max_episodes):
    state = environment.reset()
    environment.render()

    total_reward = 0

    for _ in range(1, max_steps+1):
        action = agent.action(state)
        (state_dash, reward, _, _) = environment.step(action)

        environment.render()

        total_reward += reward
        state = state_dash

        time.sleep(0.02)
    #for _ in range(1, max_steps+1):

    time.sleep(0.5)
    print("episode:" +str(t) +" reward:" + str(total_reward))
#for t in range(1, max_episodes):

environment.close()
