#CartPoleの学習済みモデルを動かしてみる
from actor_critic_agent import AgentCreater
from cartpole import Environment
from cartpole.models import ValueNet, PolicyNet

import time

max_episodes = int(1e6)


#環境を作成する
monitor=False    
environment = Environment(monitor=monitor)

#モデルを作成する
v_model = ValueNet()
p_model = PolicyNet()

#エージェントを作成する
parameters_filename = "cartpole_parameters.json"
model_filename = ".\\replay\\cartpole.model"

creater = AgentCreater(parameters_filename)
agent = creater.create(v_model, p_model)
agent.load(model_filename)

#最大ステップ数
max_steps = environment.max_steps

#リプレイ
for t in range(1, max_episodes):
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
    #for step in range(1, max_steps+1)

    time.sleep(0.5)
    print("episode:" +str(t)+ " step:" +str(step)+ " reward:" + str(total_reward))
#for t in range(1, max_episodes):

environment.close()
