import argparse

import numpy as np
import pandas as pd
import lcm

from common import constant
from online.exlcm import Transaction
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="./data", type=str, help="path where data is located")
parser.add_argument("--id_file", default="id.csv", type=str, help="file name of data")

parser.add_argument("--channel_id", default="ID", type=str, help="channel of transaction ids")


def main(overwrite_args=None):
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Debugging
            setattr(args, k, v)

    id_path = os.path.join(args.data_path, args.id_file)
    id_data = pd.read_csv(id_path, low_memory=False)

    total_num = 0

    start_time = time.time()

    m = lcm.LCM()
    for index, row in id_data.iterrows():
        msg = Transaction()
        msg.id = row[constant.FIELD_RECORD_ID]
        m.publish(args.channel_id, msg.encode())
        total_num += 1

    end_time = time.time()
    time_cost = end_time - start_time
    print("total records: {}, time cost: {}", total_num, time_cost)


if __name__ == '__main__':
    main()
