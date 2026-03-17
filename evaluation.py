import torch
import numpy as np

# 导入新的环境和网络
from texas_env import TexasEnv
from models import SD_CFR_ValueNetwork, get_strategy_from_value_net


def sample_network_from_BM(B_M_player, input_dim=111, output_dim=5):
    """
    轨迹采样：按照迭代次数 t 为权重，随机抽取一个历史网络快照
    """
    T = len(B_M_player)
    weights = np.arange(1, T + 1, dtype=np.float32)
    probabilities = weights / np.sum(weights)
    sampled_index = np.random.choice(T, p=probabilities)

    # 实例化一个新网络，并加载抽中的权重字典
    net = SD_CFR_ValueNetwork(input_dim, output_dim)
    net.load_state_dict(B_M_player[sampled_index])
    net.eval()
    return net


def play_one_hand(env, B_M_p0, B_M_p1):
    """
    让两个 AI 使用抽样出的网络打一局完整的德州扑克
    """
    # 1. 游戏开始时抽取大脑
    net_p0 = sample_network_from_BM(B_M_p0)
    net_p1 = sample_network_from_BM(B_M_p1)

    history = env.reset()
    print(f"🃏 发牌完毕！")
    # eval7 的卡牌对象自带漂亮的字符串表示
    print(f"P0 底牌: {[str(c) for c in env.cards[0:2]]}")
    print(f"P1 底牌: {[str(c) for c in env.cards[2:4]]}")

    # 5 档动作映射
    action_map = {
        0: "Fold (弃牌)",
        1: "Call/Check (跟注/过牌)",
        2: "Raise 0.2P (小注)",
        3: "Raise 0.5P (半池)",
        4: "Raise 1.0P (满池)",
    }
    action_char = {0: "f", 1: "c", 2: "s", 3: "h", 4: "p"}

    while True:
        is_terminal, p0_commit, p1_commit = env.evaluate_history(history)

        # 终局结算
        if is_terminal:
            payoff = env.get_payoff(history)
            print(f"💰 终局！历史动作: {history} | 最终底池: {p0_commit + p1_commit}")
            if not history.endswith("f"):
                print(f"公共牌: {[str(c) for c in env.cards[4:9]]}")
            print(f"👉 P0 收益: {payoff}")
            return payoff

        # 换轮与发公共牌提示
        if env.is_next_round(history) and not history.endswith("/"):
            history += "/"
            rounds = history.split("/")
            round_idx = len(rounds) - 1
            if round_idx == 1:
                print(f"\n--- 翻牌圈 (Flop): {[str(c) for c in env.cards[4:7]]} ---")
            elif round_idx == 2:
                print(f"\n--- 转牌圈 (Turn): {[str(c) for c in env.cards[4:8]]} ---")
            elif round_idx == 3:
                print(f"\n--- 河牌圈 (River): {[str(c) for c in env.cards[4:9]]} ---")
            continue

        # 轮到某位玩家行动
        turn = env.get_turn(history)
        state_tensor = env.get_state_tensor(history, turn)
        legal_actions = env.get_legal_actions(history)

        active_net = net_p0 if turn == 0 else net_p1
        strategy = get_strategy_from_value_net(active_net, state_tensor, legal_actions)

        action_idx = np.random.choice(5, p=strategy)
        chosen_char = action_char[action_idx]

        # 打印保留 3 位小数的概率分布，方便观察
        print(
            f"玩家 P{turn} 动作分布: {np.round(strategy, 3)} -> 选择了: {action_map[action_idx]}"
        )
        history += chosen_char


if __name__ == "__main__":
    print("加载德州扑克历史网络池 B_M...")
    try:
        B_M = torch.load("texas_sdcfr_models_BM.pth")
    except FileNotFoundError:
        print(
            "错误：找不到 texas_sdcfr_models_BM.pth 文件，请先运行 train.py 进行训练！"
        )
        exit()

    env = TexasEnv()

    # 观战 n 局
    for i in range(15):
        print(f"\n================ 第 {i+1} 局 ================")
        play_one_hand(env, B_M[0], B_M[1])
