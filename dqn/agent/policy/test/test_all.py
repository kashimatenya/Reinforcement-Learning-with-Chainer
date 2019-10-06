import os
import unittest

if __name__ == "__main__":
    #このファイル自体もテスト対象に含まれてしまうけど気にしない
    directory = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.defaultTestLoader.discover(directory)
    unittest.TextTestRunner().run(suite)
#if
