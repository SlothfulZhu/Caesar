class ControlHandler:
    def __init__(self, module_name, result, mutex, queue):
        self.module_name = module_name
        self.result = result
        self.mutex = mutex
        self.queue = queue

    def msg_handler(self, channel, data):
        print("channel: {}".format(channel))
        msg = self.result.decode(data)

        self.mutex.acquire()

        self.queue.put(msg.ans)
        print("[{}] receive message: {}".format(self.module_name, msg.ans))

        self.mutex.release()
