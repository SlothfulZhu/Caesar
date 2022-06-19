# coding=utf-8
class DecisionStrategy:
    def __init__(self):
        self.decision_strategy_list = [
            lambda r, s, a: r,
            lambda r, s, a: s,
            lambda r, s, a: a,
            lambda r, s, a: r | s,
            lambda r, s, a: s | a,
            lambda r, s, a: r | a,
            lambda r, s, a: r & s,
            lambda r, s, a: s & a,
            lambda r, s, a: r & a,
            lambda r, s, a: r | s | a,
            lambda r, s, a: r & s & a,
            lambda r, s, a: r | (s & a),
            lambda r, s, a: s | (r & a),
            lambda r, s, a: a | (r & s),
            lambda r, s, a: r & (s | a),
            lambda r, s, a: s & (r | a),
            lambda r, s, a: a & (r | s),
            lambda r, s, a: (r | s) & (r | a) & (s | a)
        ]

        self.r_list = {0, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
        self.s_list = {1, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17}
        self.a_list = {2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

        self.n_strategies = len(self.decision_strategy_list)

    def set_decision_strategy(self, strategy_id, func):
        self.decision_strategy_list[strategy_id] = func

    def get_decision_strategy(self, strategy_id):
        return self.decision_strategy_list[strategy_id]

    def blend(self, strategy_id, r_pred, s_pred, a_pred):
        return self.decision_strategy_list[strategy_id](r_pred, s_pred, a_pred)


decision_strategy = DecisionStrategy()
