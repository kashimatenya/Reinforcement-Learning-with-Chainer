#CartPole問題を強化学習で解く
import os
from datetime import datetime
import shutil 
from statistics import median, mean

from agent import DQNAgentCreater
from agent.memory import Memory

from cartpole import Environment
from cartpole.models import QNet

from test_log import TestReport, TestLogs, FileTestLog, DisplayTestLog


#ログを残すか？
leaving_logs = True

max_episodes = int(1e6+1)
test_interval = 500
test_repeat = 10

#環境を作成する
environment = Environment()

#モデルを作成する
q_model = QNet()

#エージェントを作成
parameters_filename = "parameters.json"
creater = DQNAgentCreater(parameters_filename)
agent = creater.create(q_model)
agent.set_policy("e-Greedy")


if leaving_logs:
    #ログフォルダを作成する
    if not os.path.exists("./log/"):
            os.mkdir("./log/")        
    #if
    if not os.path.exists("./log/cartpole/"):
            os.mkdir("./log/cartpole/")        
    #if

    log_folder = "./log/cartpole/" + datetime.now().strftime('%Y%m%d%H%M%S')
    os.mkdir(log_folder)

    #テストログ
    test_log_filename =  log_folder + "/test_log.txt"

    #テストログを画面に表示&ファイルに保存する
    test_logs = [DisplayTestLog(), FileTestLog(test_log_filename)]

    #パラメーターファイルをコピーする
    shutil.copy(parameters_filename, log_folder+"/"+parameters_filename)

    #モデル(のソースコード)をコピーする
    shutil.copy("./cartpole/models.py", log_folder+"/models.py")
else:
    #テスト結果をディスプレイに表示する
    test_logs = [DisplayTestLog(),]
#if-else

#テストログ表示
test_log = TestLogs(test_logs)

max_steps = environment.max_steps

#学習する
for t in range(1, max_episodes+1):
    state = environment.reset()
    
    for _ in range(1, max_steps+1):
        action = agent.action(state)
        (state_dash, reward, done, info) = environment.step(action)
        agent.gain_experience(state, action, reward, state_dash, done, info)

        state = state_dash

        #訓練する間隔を少し開ける
        if agent.is_to_train():
            #テストの周期にあたるときは訓練誤差を計算する        
            estimate_loss = (t%test_interval == 0)    
            training_TD_error = agent.train(estimate_loss=estimate_loss)
        #if
        
        #エピソード終端か？
        if done:
            break
        #if
    #while

    if t % test_interval == 0:
        #テストする
        steps = []
        total_rewards = []

        test_memory = Memory(test_repeat*max_steps)
        agent.set_policy("Greedy")

        #テストは複数回行う
        for _ in range(test_repeat):

            state = environment.reset()
            total_reward = 0

            for step in range(1, max_steps+1):                
                action = agent.action(state)
                (state_dash, reward, done, info) = environment.step(action)

                total_reward += reward
                test_memory.append(state, action, reward, state_dash, done)

                state = state_dash

                #エピソード終端か？
                if done:
                    break
                #if
            # for step in range(1, MaxSteps+1):

            steps.append(step)
            total_rewards.append(total_reward)
        #for _ in range(test_times):

        #テスト結果
        step = median(steps)        
        median_total_reward = median(total_rewards)
        worst_total_reward = min(total_rewards)

        #テスト時のTD誤差を見積もる
        experience = test_memory.refer()
        test_TD_error = agent.estimate_squared_TD_error_from_experience(experience)
        
        #テスト結果を表示する
        report = TestReport(
                            t,
                            step, 
                            median_total_reward, 
                            worst_total_reward, 
                            training_TD_error, 
                            test_TD_error, 
                        )
        test_log.print(report)

        #モデルを保存する
        if leaving_logs:
            episode_string = "Episode-" + str(t).zfill(10)
            model_filename  = log_folder +"/" +episode_string + ".model"
            memory_filename = None
            agent.save(model_filename, memory_filename)
        #if leaving_logs
        
        agent.set_policy("e-Greedy")
    #if t % test_interval == 0

    #エピソード終了処理
    agent.end_episode()
#for t in range(1, max_episodes+1)

environment.close()
