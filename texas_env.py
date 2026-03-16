import numpy as np
import torch
import eval7
import math


class TexasStateEncoder:
    """
    多档位限注德州扑克状态编码器。
    """

    def __init__(self):
        # 维度构成不变: 52 + 52 + 4 + 3 = 111 维
        self.input_dim = 111

        self.ranks = "23456789TJQKA"
        self.suits = "cdhs"
        self.card_to_idx = {
            f"{r}{s}": i
            for i, (r, s) in enumerate((r, s) for r in self.ranks for s in self.suits)
        }

    def encode(self, private_cards, public_cards, pot, p0_commit, p1_commit, round_idx):
        vec = np.zeros(self.input_dim, dtype=np.float32)

        # 1. 编码底牌
        for card in private_cards:
            idx = self.card_to_idx[str(card)]
            vec[idx] = 1.0

        # 2. 编码公共牌
        for card in public_cards:
            idx = self.card_to_idx[str(card)]
            vec[52 + idx] = 1.0

        # 3. 编码当前轮次
        vec[104 + round_idx] = 1.0

        # 4. 编码下注信息 (核心修改：使用对数缩放对抗指数爆炸)
        # np.log1p(x) 等价于 ln(1 + x)，能将数万的筹码平滑压缩到个位数
        vec[108] = np.log1p(pot) / 10.0
        vec[109] = np.log1p(p0_commit) / 10.0
        vec[110] = np.log1p(p1_commit) / 10.0

        return torch.tensor(vec).unsqueeze(0)


class TexasEnv:
    """
    多档位限注 (Discrete Pot-Limit) 单挑德州扑克引擎。
    支持 0.2pot, 0.5pot, 1.0pot 三种不同力度的加注。
    """

    def __init__(self):
        self.encoder = TexasStateEncoder()
        self.deck = None
        self.cards = []

    def reset(self):
        """蒙特卡洛发牌"""
        self.deck = eval7.Deck()
        self.deck.shuffle()
        self.cards = self.deck.sample(9)
        return ""

    def get_turn(self, history):
        rounds = history.split("/")
        return len(rounds[-1]) % 2

    def evaluate_history(self, history):
        """
        动态解析复杂的多档位加注历史
        动作: f(Fold), c(Call/Check), s(0.2pot), h(0.5pot), p(1.0pot)
        """
        # 为了方便计算倍数，底注(Ante)放大为 10.0
        p0_commit, p1_commit = 10.0, 10.0

        rounds = history.split("/")

        for round_history in rounds:
            p0_turn = True
            for action in round_history:
                # 在计算比例加注前，标准的扑克规则是：先假设你已经跟注 (Call)
                matched_commit = max(p0_commit, p1_commit)
                # 假设跟注后的底池总大小
                pot_after_call = matched_commit * 2.0

                if action == "c":
                    if p0_turn:
                        p0_commit = matched_commit
                    else:
                        p1_commit = matched_commit
                elif action == "s":
                    raise_amt = 0.2 * pot_after_call
                    if p0_turn:
                        p0_commit = matched_commit + raise_amt
                    else:
                        p1_commit = matched_commit + raise_amt
                elif action == "h":
                    raise_amt = 0.5 * pot_after_call
                    if p0_turn:
                        p0_commit = matched_commit + raise_amt
                    else:
                        p1_commit = matched_commit + raise_amt
                elif action == "p":
                    raise_amt = 1.0 * pot_after_call
                    if p0_turn:
                        p0_commit = matched_commit + raise_amt
                    else:
                        p1_commit = matched_commit + raise_amt

                p0_turn = not p0_turn

        is_terminal = False
        if history.endswith("f"):
            is_terminal = True
        # 终局判定更通用：第4轮，且动作序列长度>=2，且以 'c' 结束（必然是跟注或过牌）
        elif len(rounds) == 4 and len(rounds[-1]) >= 2 and rounds[-1].endswith("c"):
            is_terminal = True

        return is_terminal, p0_commit, p1_commit

    def get_legal_actions(self, history):
        """获取当前合法的动作 [0:Fold, 1:Call, 2:0.2Pot, 3:0.5Pot, 4:1.0Pot]"""
        is_terminal, _, _ = self.evaluate_history(history)
        if is_terminal:
            return []

        rounds = history.split("/")
        current_round = rounds[-1]

        # 本轮已经结束
        if len(current_round) >= 2 and current_round.endswith("c"):
            return []

        legal_actions = [0, 1]

        # 统计本轮总加注次数。即使档位不同，总计也最多允许 4 次加注
        num_raises = (
            current_round.count("s")
            + current_round.count("h")
            + current_round.count("p")
        )
        if num_raises < 2:
            legal_actions.extend([2, 3, 4])

        return legal_actions

    def is_next_round(self, history):
        """通用换轮判定：当前轮次有大于等于2个动作，且以 'c' 结束"""
        is_terminal, _, _ = self.evaluate_history(history)
        if is_terminal:
            return False

        rounds = history.split("/")
        current_round = rounds[-1]
        if len(current_round) >= 2 and current_round.endswith("c"):
            return True

        return False

    def get_payoff(self, history):
        """收益结算"""
        _, p0_commit, p1_commit = self.evaluate_history(history)

        if history.endswith("f"):
            turn = self.get_turn(history)
            if turn == 1:
                return -p0_commit
            else:
                return p1_commit

        p0_hand = self.cards[0:2]
        p1_hand = self.cards[2:4]
        board = self.cards[4:9]

        p0_score = eval7.evaluate(p0_hand + board)
        p1_score = eval7.evaluate(p1_hand + board)

        if p0_score > p1_score:
            return p1_commit
        elif p1_score > p0_score:
            return -p0_commit
        else:
            return 0.0

    def get_state_tensor(self, history, player):
        _, p0_commit, p1_commit = self.evaluate_history(history)
        pot = p0_commit + p1_commit

        private_cards = self.cards[player * 2 : player * 2 + 2]

        rounds = history.split("/")
        round_idx = len(rounds) - 1

        if round_idx == 0:
            public_cards = []
        elif round_idx == 1:
            public_cards = self.cards[4:7]
        elif round_idx == 2:
            public_cards = self.cards[4:8]
        elif round_idx == 3:
            public_cards = self.cards[4:9]

        return self.encoder.encode(
            private_cards, public_cards, pot, p0_commit, p1_commit, round_idx
        )
