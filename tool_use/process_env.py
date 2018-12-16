import sys
import atexit
import traceback
import multiprocessing


class ProcessEnv(object):
    """
    Adapted from Batch PPO:
    https://github.com/google-research/batch-ppo/blob/master/agents/tools/wrappers.py#L303
    """

    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, constructor):
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=self._worker, args=(constructor, conn))

        atexit.register(self.close)

        self._process.start()
        self._observ_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._observ_space:
            self._observ_space = self.__getattr__('observation_space')
        return self._observ_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # connection already closed
            pass

        self._process.join()

    def seed(self, seed, blocking=True):
        promise = self.call('seed', seed)
        if blocking:
            return promise()
        else:
            return promise

    def step(self, action, blocking=True):
        promise = self.call('step', action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        promise = self.call('reset')
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        message, payload = self._conn.recv()

        # re-raise exceptions in the main process
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)

        if message == self._RESULT:
            return payload

        raise KeyError(
            'Received message of unexpected type {}'.format(message))

    def _worker(self, constructor, conn):
        try:
            env = constructor()

            while True:
                try:
                    # only block for short times to support keyboard exceptions
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break

                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue

                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue

                if message == self._CLOSE:
                    assert payload is None
                    break

                raise KeyError(
                    'Received message of unknown type {}'.format(message))

        except Exception:
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            conn.close()
