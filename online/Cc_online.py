import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import sys
import _thread
import threading
import queue
import time
import lcm

from common import constant
from common.strategyx import decision_strategy
from online.exlcm import Transaction, ResultR, ResultS, ResultA
from online.handler.client_handler import ClientHandler
from online.handler.control_handler import ControlHandler

parser = argparse.ArgumentParser()

parser.add_argument("--channel_result_Rr", default="ResultR", type=str, help="channel of Rr results")
parser.add_argument("--channel_result_Sa", default="ResultS", type=str, help="channel of Sa results")
parser.add_argument("--channel_result_Ae", default="ResultA", type=str, help="channel of Ae results")
parser.add_argument("--channel_id", default="ID", type=str, help="channel of transaction ids")
parser.add_argument("--channel_Rr", default="ID_R", type=str, help="channel to Rr")
parser.add_argument("--channel_Sa", default="ID_S", type=str, help="channel to Sa")
parser.add_argument("--channel_Ae", default="ID_A", type=str, help="channel to Ae")

parser.add_argument("--model_path", default="../model", type=str, help="path where the models are located")
parser.add_argument("--Cc_model_file", default="Cc.model", type=str)

parser.add_argument("--data_path", default="./data", type=str, help="path where data is located")
parser.add_argument("--data_file", default="data.csv", type=str, help="file name of data")


class ResultTracker:
    def __init__(self):
        self.visual_ans = list()
        self.visual_id = list()
        self.visual_Rr = list()
        self.visual_Sa = list()
        self.visual_Ae = list()
        self.visual_sid = list()

        self.start_time = time.time()

        self.num_Rr = 0
        self.num_Sa = 0
        self.num_Ae = 0
        self.num_records = 0


class CenterControl:
    def __init__(self, handler_Rr, handler_Sa, handler_Ae, handler_id, model, args):
        self.handler_Rr = handler_Rr
        self.handler_Sa = handler_Sa
        self.handler_Ae = handler_Ae
        self.handler_id = handler_id

        self.model = model

        self.queue_sid = queue.Queue()

        self.channel_handler_dict = dict()

        self.data_source = None

        self.result_tracker = ResultTracker()

        self.args = args

        self.thresholds = None

        self.pub_lcm = lcm.LCM()

    def register_handler(self, channel_handler_dict):
        self.channel_handler_dict = channel_handler_dict

    def load_source(self, path):
        self.data_source = pd.read_csv(path, low_memory=False)

    def receive(self):
        m = lcm.LCM()
        for channel, handler in self.channel_handler_dict.items():
            m.subscribe(channel, handler)
        try:
            while True:
                m.handle()
        except KeyboardInterrupt:
            print("receive exits")

    def _publish_id_msg(self, id, sid):
        id = int(id)
        sid = int(sid)

        msg = Transaction()
        msg.id = id

        if sid in decision_strategy.r_list:
            self.pub_lcm.publish(self.args.channel_Rr, msg.encode())
            self.result_tracker.num_Rr += 1
        if sid in decision_strategy.s_list:
            self.pub_lcm.publish(self.args.channel_Sa, msg.encode())
            self.result_tracker.num_Sa += 1
        if sid in decision_strategy.a_list:
            self.pub_lcm.publish(self.args.channel_Ae, msg.encode())
            self.result_tracker.num_Ae += 1

        self.result_tracker.num_records += 1

    def distribute(self):
        while True:
            if not self.handler_id.queue.empty():
                self.handler_id.mutex.acquire()
                id = int(self.handler_id.queue.get())
                self.handler_id.mutex.release()

                record = self.data_source[self.data_source.id == id]
                if not record.empty:
                    del record[constant.FIELD_RECORD_ID]
                    del record[constant.FIELD_LABEL]

                    sid = self.model.predict(record)

                    self.result_tracker.visual_id.append(id)
                    self.queue_sid.put(sid)
                    self._publish_id_msg(id, sid)
                else:
                    print("record (id: {}) dose not exist".format(id))

    def blend(self):
        if self.thresholds is None:
            self.thresholds = [0.5, 0.5, 0.5]
        while True:
            if not self.queue_sid.empty():
                sid = int(self.queue_sid.get())
                results = [0, 0, 0]
                if sid in decision_strategy.r_list:
                    while True:
                        if not self.handler_Rr.queue.empty():
                            self.handler_Rr.mutex.acquire()
                            res = self.handler_Rr.queue.get()
                            results[0] = 0 if res < self.thresholds[0] else 1
                            self.handler_Rr.mutex.release()
                            break
                else:
                    results[0] = 0

                if sid in decision_strategy.s_list:
                    while True:
                        if not self.handler_Sa.queue.empty():
                            self.handler_Sa.mutex.acquire()
                            res = self.handler_Sa.queue.get()
                            results[1] = 0 if res < self.thresholds[1] else 1
                            self.handler_Sa.mutex.release()
                            break
                else:
                    results[1] = 0

                if sid in decision_strategy.a_list:
                    while True:
                        if not self.handler_Ae.queue.empty():
                            self.handler_Ae.mutex.acquire()
                            res = self.handler_Ae.queue.get()
                            results[2] = 0 if res < self.thresholds[2] else 1
                            self.handler_Ae.mutex.release()
                            break
                else:
                    results[2] = 0

                ans = decision_strategy.blend(sid, *results)

                self.result_tracker.visual_ans.append(ans)
                self.result_tracker.visual_sid.append(sid)
                self.result_tracker.visual_Rr.append(results[0])
                self.result_tracker.visual_Sa.append(results[1])
                self.result_tracker.visual_Ae.append(results[2])


def get_model(model_path):
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model(model_path)
    clf._Booster = booster
    return clf


def main(overwrite_args=None):
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Debugging
            setattr(args, k, v)

    # initialize queue
    queue_Rr = queue.Queue()
    queue_Sa = queue.Queue()
    queue_Ae = queue.Queue()
    queue_id = queue.Queue()

    # initialize mutex
    mutex_Rr = threading.Lock()
    mutex_Sa = threading.Lock()
    mutex_Ae = threading.Lock()
    mutex_id = threading.Lock()

    # initialize handler
    handler_Rr = ControlHandler(constant.MODULE_RR, ResultR, mutex_Rr, queue_Rr)
    handler_Sa = ControlHandler(constant.MODULE_SA, ResultS, mutex_Sa, queue_Sa)
    handler_Ae = ControlHandler(constant.MODULE_AE, ResultA, mutex_Ae, queue_Ae)
    handler_id = ClientHandler(constant.MODULE_CC, Transaction, mutex_id, queue_id)

    channel_handler_dict = dict()
    channel_handler_dict[args.channel_result_Rr] = handler_Rr.msg_handler
    channel_handler_dict[args.channel_result_Sa] = handler_Sa.msg_handler
    channel_handler_dict[args.channel_result_Ae] = handler_Ae.msg_handler
    channel_handler_dict[args.channel_id] = handler_id.msg_handler

    # load model
    model = get_model(os.path.join(args.model_path, args.Cc_model_file))

    # center control
    center_control = CenterControl(handler_Rr, handler_Sa, handler_Ae, handler_id, model, args)
    center_control.load_source(os.path.join(args.data_path, args.data_file))
    center_control.register_handler(channel_handler_dict)

    _thread.start_new_thread(center_control.receive, ())

    print("ready...")
    _thread.start_new_thread(center_control.distribute, ())
    _thread.start_new_thread(center_control.blend, ())
    while True:
        time.sleep(10)


if __name__ == '__main__':
    main()
