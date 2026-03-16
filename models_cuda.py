import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SD_CFR_ValueNetwork(nn.Module):
    """
    SD-CFR 价值网络 (Input: 111, Output: 5)
    """

    def __init__(self, input_dim=111, output_dim=5):
        super(SD_CFR_ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.output(x)


def get_strategy_from_value_net(value_net, state_tensor, legal_actions):
    value_net.eval()
    with torch.no_grad():
        # 👇 核心适配：把 CPU 上的张量送进显卡进行推理，然后再拉回 CPU
        state_tensor = state_tensor.to(device)
        advantages = value_net(state_tensor).squeeze(0).cpu().numpy()

    strategy = np.zeros(5, dtype=np.float32)
    positive_advantages = np.zeros(5, dtype=np.float32)

    for a in legal_actions:
        positive_advantages[a] = max(advantages[a], 0.0)

    sum_advantages = np.sum(positive_advantages)

    if sum_advantages > 0:
        strategy = positive_advantages / sum_advantages
    else:
        max_adv = -float("inf")
        best_a = -1
        for a in legal_actions:
            if advantages[a] > max_adv:
                max_adv = advantages[a]
                best_a = a
        if best_a != -1:
            strategy[best_a] = 1.0

    return strategy
