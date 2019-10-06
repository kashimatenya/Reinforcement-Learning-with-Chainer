#Pendulumの問題を強化学習で解く
from datetime import datetime
import os
import shutil 
from statistics import median, mean

from actor_critic_agent import AgentCreater
from actor_critic_agent.memory import Memory

from pendulum import Environment
from pendulum.models import ValueNet, PolicyNet 

from test_log import TestReport, DisplayTestLog, FileTestLog, TestLogs


max_episodes = int(1e5)
test_interval = 500
test_repeat = 10


#ログをファイルに残すか？
leaving_logs = True

#環境を作成する
environment = Environment()

#モデルを作成する
v_model = ValueNet()
pi_model = PolicyNet()

#エージェントを作成する
parameters_filename = "pendulum_parameters.json"
creater = AgentCreater(parameters_filename)
agent = creater.create(v_model, pi_model)

if leaving_logs:
    #ログフォルダを作成する
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    #if
    
    if not os.path.exists("./log/pendulum/"):
        os.mkdir("./log/pendulum/")
    #if
    
    log_folder = "./log/pendulum/" + datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(log_folder)

    #テストログファイル
    test_log_filename =  log_folder + "/test_log.txt"

    #テストログを画面に表示&ファイルに保存する
    test_logs = [DisplayTestLog(), FileTestLog(test_log_filename)]

    #パラメーターファイルをコピーする
    shutil.copy(parameters_filename, log_folder+"/"+parameters_filename)

    #モデル(のソースコード)をコピーする
    shutil.copy("./pendulum/models.py", log_folder+"/models.py")

else:
    #テスト結果をディスプレイに表示する
    test_logs = [DisplayTestLog(),]
#if-else

#テストログ表示
test_log = TestLogs(test_logs)

max_steps = environment.max_steps

#試行を繰り返して訓練する
for t in range(1, max_episodes+1):
    state = environment.reset()

    for step in range(1, max_steps+1):
        #試行する
        action = agent.action(state)
        (state_dash, reward, done, info) = environment.step(action)
        agent.gain_experience(state, action, reward, state_dash, done, info)

        #経験がたまったら学習する
        if agent.is_experienced():
            
            #テストの周期にあたる場合は訓練誤差を計算する    
            estimate_loss = (t%test_interval == 0)
            
            (training_V_loss, training_pi_loss, training_pi_entropy) = agent.train(estimate_loss=estimate_loss)
            agent.reset_experience()
        #if

        state = state_dash
    #for step in range(1, max_steps+1)

    #テストする
    if t % test_interval == 0:
        total_rewards = []
        steps = []

        #テスト用のメモリー
        test_memory = Memory(test_repeat*max_steps)

        #テストを複数エピソードを繰り返す
        for _ in range(test_repeat):
            total_reward = 0

            state = environment.reset()
            for step in range(1, max_steps+1):
                #行動する
                action = agent.action(state)
                (state_dash, reward, done, info) = environment.step(action)

                test_memory.append(state, action, reward, state_dash, done)
                total_reward += reward

                state = state_dash
            #for step in range(1, max_steps+1)
        
            total_rewards.append(total_reward)
            steps.append(step)
        #for _ in range(test_repeat):


        #収益のmedianと最小値
        median_total_reward = median(total_rewards)
        worst_total_reward = min(total_rewards)

        #ステップ数
        step = median(steps)

        #テスト時の誤差を計算する
        experiences = test_memory.refer()
        (test_V_loss, test_pi_loss, test_pi_entropy) = agent.estimate_loss_from_experience(experiences)

        #テスト結果を表示する
        report = TestReport(
                            t,
                            step, 
                            median_total_reward, worst_total_reward, 
                            training_V_loss, training_pi_loss, training_pi_entropy, 
                            test_V_loss, test_pi_loss, test_pi_entropy
                        )
        test_log.print(report)

        #モデルを保存する
        if leaving_logs:
            episode_string = "Episode-" + str(t).zfill(10)
            model_filename = log_folder +"/" +episode_string + ".model"
            agent.save(model_filename)
        #if

    #if t % test_interval == 0:
#for t in range(1, max_episodes):

environment.close()
