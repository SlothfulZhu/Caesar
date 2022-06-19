# coding=utf-8
import argparse
import os

import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm

from common import constant
from common.client import ModuleClient
from common.strategyx import decision_strategy

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", default="./data", type=str, help="path where data is located")
parser.add_argument("--data_file", default="data.csv", type=str, help="file name of data")

parser.add_argument("--model_path", default="../model", type=str, help="path where the models are located")
parser.add_argument("--Rr_model_file", default="Rr.model", type=str)
parser.add_argument("--Sa_model_file", default="Sa.model", type=str)
parser.add_argument("--Ae_model_file", default="Ae.model", type=str)

parser.add_argument("--train_file", default="train.csv", type=str)
parser.add_argument("--Cc_model_file", default="Cc.model", type=str)

parser.add_argument("--n_estimators", default=60, type=int)
parser.add_argument("--max_depth", default=3, type=int)
parser.add_argument("--min_child_weight", default=1.0, type=float)
parser.add_argument("--gamma", default=0.0, type=float)
parser.add_argument("--subsample", default=0.9, type=float)
parser.add_argument("--colsample_bytree", default=1.0, type=float)
parser.add_argument("--reg_alpha", default=0, type=float)
parser.add_argument("--reg_lambda", default=1, type=float)
parser.add_argument("--learning_rate", default=0.1, type=float)
parser.add_argument("--random_state", default=0, type=int)


def init_clients(models):
    clients = [ModuleClient(model) for model in models]
    return clients


def get_strategy_id(sid, r, s, a, label):
    if label == decision_strategy.blend(sid, r, s, a):
        return sid
    else:
        return np.nan


def generate_train_data(data, clients):
    features = list(data.columns)
    features.remove(constant.FIELD_RECORD_ID)
    features.remove(constant.FIELD_LABEL)

    columns = features[:]
    columns.append(constant.FIELD_STRATEGY_ID)

    for i in tqdm(range(decision_strategy.n_strategies)):
        data[str(i)] = data.apply(lambda x: get_strategy_id(i,
                                                            *[int(client.invoke(x[features])) for client in clients],
                                                            int(x[constant.FIELD_LABEL])
                                                            ), axis=1)

    train = data[features + ['0']].values
    for i in range(1, decision_strategy.n_strategies):
        train = np.vstack((train, data[features + [str(i)]].values))

    train = pd.DataFrame(data=train, columns=columns)
    train.dropna(axis=0, subset=[constant.FIELD_STRATEGY_ID], inplace=True)

    return train


def train_model(args):
    train_path = os.path.join(args.data_path, args.train_file)
    print(train_path)

    train = pd.read_csv(train_path)
    features = list(train.columns)
    features.remove(constant.FIELD_STRATEGY_ID)
    X_train, y_train = train[features], train[constant.FIELD_STRATEGY_ID]

    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,

        random_state=args.random_state
    )

    clf.fit(X_train, y_train)
    model_path = os.path.join(args.model_path, args.Cc_model_file)
    clf.get_booster().save_model(model_path)
    print(model_path)


def main(overwrite_args=None):
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Debugging
            setattr(args, k, v)

    clients = init_clients([os.path.join(args.model_path, args.Rr_model_file),
                            os.path.join(args.model_path, args.Sa_model_file),
                            os.path.join(args.model_path, args.Ae_model_file)])

    data = pd.read_csv(os.path.join(args.data_path, args.data_file))

    train_data = generate_train_data(data, clients)
    train_data.to_csv(os.path.join(args.data_path, args.train_file), index=False)


if __name__ == '__main__':
    main()
