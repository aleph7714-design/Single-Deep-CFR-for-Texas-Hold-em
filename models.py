import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SD_CFR_ValueNetwork(nn.Module):
    """
    SD-CFR 的单一价值网络 (Value Network) —— 德州扑克升级版。
    用于预测在特定信息集下，采取各个合法动作的反事实遗憾值 (Advantage)。
    """

    def __init__(self, input_dim=111, output_dim=5):
        super(SD_CFR_ValueNetwork, self).__init__()

        # 针对德州扑克的复杂状态空间，网络显著加宽并加深
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

        # 输出层：对应 5 个合法动作的预测遗憾值
        # 0: Fold, 1: Call/Check, 2: Raise(0.2 Pot), 3: Raise(0.5 Pot), 4: Raise(1.0 Pot)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # 最后一层不加激活函数，因为反事实遗憾值可以是负数
        return self.output(x)


def get_strategy_from_value_net(value_net, state_tensor, legal_actions):
    """
    通过遗憾匹配 (Regret Matching) 将预测的遗憾值转化为概率分布。

    参数:
    - value_net: 当前训练的价值网络
    - state_tensor: 当前状态的张量表示 (1, 111)
    - legal_actions: 当前合法的动作索引列表，例如 [0, 1, 2, 3, 4]

    返回:
    - strategy: 长度为 5 的 numpy 数组，表示采取每个动作的概率
    """
    # 确保网络处于评估模式
    value_net.eval()
    with torch.no_grad():
        # 获取网络输出，转移到 CPU 并转为一维 numpy 数组 (兼容将来 RTX 3070 上的 GPU 张量)
        advantages = value_net(state_tensor).squeeze(0).cpu().numpy()

    # 初始化长度为 5 的动作概率数组和正向遗憾值数组
    strategy = np.zeros(5, dtype=np.float32)
    positive_advantages = np.zeros(5, dtype=np.float32)

    # 仅计算合法动作的正向遗憾值
    for a in legal_actions:
        # x_+ = max(x, 0)
        positive_advantages[a] = max(advantages[a], 0.0)

    sum_advantages = np.sum(positive_advantages)

    # 如果正向遗憾值之和大于 0，则按比例分配概率
    if sum_advantages > 0:
        strategy = positive_advantages / sum_advantages
    else:
        # SD-CFR 深度学习启发式回退策略：
        # 如果没有正向遗憾值，贪婪地选择 advantage 预测值最高的合法动作
        max_adv = -float("inf")
        best_a = -1
        for a in legal_actions:
            if advantages[a] > max_adv:
                max_adv = advantages[a]
                best_a = a

        # 将最优动作概率设为 1.0
        if best_a != -1:
            strategy[best_a] = 1.0

    return strategy
