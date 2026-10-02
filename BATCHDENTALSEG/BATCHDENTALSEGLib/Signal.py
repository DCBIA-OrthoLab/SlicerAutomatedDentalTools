from copy import copy
from itertools import count


class Signal(object):
    """ Qt like signal slot connections. Enables using the same semantics with Slicer as qt.Signal lead to application
    crash.
    (see : https://discourse.slicer.org/t/custom-signal-slots-with-pythonqt/3278/5)
    """

    def __init__(self, *typeInfo):
        self._id = count(0, 1)
        self._connectDict = {}
        self._typeInfo = str(typeInfo)
        self._isSignalBlocked = False

    def emit(self, *args, **kwargs):
        if self._isSignalBlocked:
            return

        for slot in copy(self._connectDict).values():
            slot(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.emit(*args, **kwargs)

    def connect(self, slot):
        assert slot, "Chosen slot should be a callable"
        next_id = next(self._id)
        self._connectDict[next_id] = slot
        return next_id

    def disconnect(self, connectId):
        if connectId in self._connectDict:
            del self._connectDict[connectId]
            return True
        return False

    def disconnectAll(self):
        for connect_id in list(self._connectDict.keys()):
            self.disconnect(connect_id)

    def blockSignals(self, isBlocked):
        self._isSignalBlocked = isBlocked
