class ClientHandler:
    def __init__(self, module_name, transaction, mutex, queue):
        self.module_name = module_name
        self.transaction = transaction
        self.mutex = mutex
        self.queue = queue

    def msg_handler(self, channel, data):
        print("channel: {}".format(channel))
        msg = self.transaction.decode(data)

        self.mutex.acquire()

        self.queue.put(msg.id)
        print("[{}] receive message: {}".format(self.module_name, msg.id))

        self.mutex.release()
