import argparse
import os
import threading
import time
import numpy as np
import pandas as pd
import lcm

from common import constant
from common.client import ModuleClient
from online.exlcm import Transaction, ResultR
import queue
import _thread

from online.handler.client_handler import ClientHandler

parser = argparse.ArgumentParser()

parser.add_argument("--channel_result_Rr", default="ResultR", type=str, help="channel of Rr results")
parser.add_argument("--channel_id_Rr", default="ID_R", type=str, help="channel of ids")

parser.add_argument("--data_path", default="./data", type=str, help="path where data is located")
parser.add_argument("--data_file", default="data.csv", type=str, help="file name of data")

parser.add_argument("--model_path", default="../model", type=str, help="path where the models are located")
parser.add_argument("--model_file", default="Rr.model", type=str)


class RiskReview:
    def __init__(self, client, pub_channel):
        self.client = client

        self.data_source = None

        self.pub_channel = pub_channel

        self.handler = None

    def register_handler(self, handler):
        self.handler = handler

    def load_source(self, path):
        self.data_source = pd.read_csv(path, low_memory=False)

    def receive(self, channel):
        m = lcm.LCM()
        m.subscribe(channel, self.handler.msg_handler)

        try:
            while True:
                m.handle()
        except KeyboardInterrupt:
            print("receive exits")

    def run(self):
        m = lcm.LCM()
        while True:
            if not self.handler.queue.empty():
                id = int(self.handler.queue.get())

                record = self.data_source[self.data_source.id == id]
                del record[constant.FIELD_RECORD_ID]
                del record[constant.FIELD_LABEL]

                ans = self.client.invoke(record)
                msg = ResultR()
                msg.ans = ans

                m.publish(self.pub_channel, msg.encode())


def main(overwrite_args):
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Debugging
            setattr(args, k, v)

    handler = ClientHandler(constant.MODULE_RR, Transaction, threading.Lock(), queue.Queue())

    client = ModuleClient(os.path.join(args.model_path, args.model_file))

    # risk review
    risk_review = RiskReview(client, args.channel_result_Rr)
    risk_review.load_source(os.path.join(args.data_path, args.data_file))
    risk_review.register_handler(handler)

    _thread.start_new_thread(risk_review.run, ())

    risk_review.receive(args.channel_id_Rr)


if __name__ == '__main__':
    main()
