#簡易なリング構造を実現する
class RingedIndex:
    def __init__(self, first, last):
        self._base = first
        self._length  = last-first
        self._position = 0
    #def

    def next(self):
        try:
            return self._position + self._base
        finally:
            self._position = (self._position + 1) % self._length
        #try-finally
    #def

    def reset(self):
        self._position = 0
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
