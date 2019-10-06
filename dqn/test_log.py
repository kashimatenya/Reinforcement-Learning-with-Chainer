import datetime
from collections import namedtuple
from time import sleep

#テスト結果
TestReport = namedtuple("TestReport", 
                        ( "episode", 
                          "steps",
                          "median_total_reward",
                          "worst_total_reward",
                          "training_TD_error", 
                          "test_TD_error" )
                        )


#テスト結果のログを画面に出力する
class DisplayTestLog:
    def __int__(self):
        pass
    #def

    def print(self, data):
        print( "episode:" +str(data.episode)
              +", steps:" +str(data.steps)
              +", median_toal_reward:{:.2f}".format(data.median_total_reward)
              +", worst_toal_reward:{:.2f}".format(data.worst_total_reward)
            )
    #def
#class


#テスト結果のログをファイルに残す
class FileTestLog:
    def __init__(self, filename):
        self._retry = 0.1
        self._filename = filename

        with open(self._filename, "a") as file:
            file.write("#episode, steps, median_total_rewad, worst_total_rewad, training squared_TD_error, test squared_TD_error \n")
        #with
    #def

    def print(self, data):
        while True:
            #ファイルのアクセス権が取れるまでリトライ
            try:
                with open(self._filename, "a") as file:
                    data_list = list(map(lambda value: str(value) , data))
                    time_stamp = datetime.datetime.now()

                    file.write(','.join(data_list)+"," +str(time_stamp) +"\n")
                #with
                #書き込みに成功したら抜ける
                break

            except Exception:
                #アクセス権が取れなかったらリトライ
                sleep(self._retry)
                continue
            #try-except
        #while

        return
    #def
#class


#各種のテストログを残す
class TestLogs:
    def __init__(self, logs):
        self._logs = logs
    #def

    def print(self, data):
        for log in self._logs:
            log.print(data)
        #for
        return
    #def
#class
