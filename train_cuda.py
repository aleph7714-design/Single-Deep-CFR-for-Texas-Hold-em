import torch
import torch.optim as optim
import numpy as np
import time
import copy
import os

from texas_env import TexasEnv
from models_cuda import SD_CFR_ValueNetwork, get_strategy_from_value_net
from buffer import ReservoirBuffer

# 自动检测 NVIDIA 显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

ITERATIONS = 200
TRAVERSALS_PER_ITER = 100000
BATCH_SIZE = 10240
UPDATE_STEPS = 4000
BUFFER_CAPACITY = 20000000

ACTION_MAP = {0: "f", 1: "c", 2: "s", 3: "h", 4: "p"}


def traverse(env, history, traverser, iteration, nets, buffers):
    """外部采样遍历器"""
    is_terminal, _, _ = env.evaluate_history(history)
    if is_terminal:
        payoff = env.get_payoff(history)
        return payoff if traverser == 0 else -payoff

    if env.is_next_round(history) and not history.endswith("/"):
        return traverse(env, history + "/", traverser, iteration, nets, buffers)

    turn = env.get_turn(history)
    state_tensor = env.get_state_tensor(history, turn)
    legal_actions = env.get_legal_actions(history)

    if not legal_actions:
        return 0.0

    strategy = get_strategy_from_value_net(nets[turn], state_tensor, legal_actions)

    if turn != traverser:
        a = np.random.choice(5, p=strategy)
        action_str = ACTION_MAP[a]
        return traverse(env, history + action_str, traverser, iteration, nets, buffers)
    else:
        expected_utility = 0.0
        action_utilities = np.zeros(5, dtype=np.float32)

        for a in legal_actions:
            action_str = ACTION_MAP[a]
            util = traverse(
                env, history + action_str, traverser, iteration, nets, buffers
            )
            action_utilities[a] = util
            expected_utility += strategy[a] * util

        regrets = np.zeros(5, dtype=np.float32)
        for a in legal_actions:
            regrets[a] = action_utilities[a] - expected_utility

        buffers[traverser].add(state_tensor, regrets, iteration)
        return expected_utility


def train_value_network(net, buffer, optimizer, iteration):
    """训练价值网络，全程在 CUDA 上进行矩阵加速"""
    if len(buffer) < BATCH_SIZE:
        return 0.0

    net.train()
    total_loss = 0.0

    for _ in range(UPDATE_STEPS):
        states, target_regrets, weights = buffer.sample(BATCH_SIZE)

        # 把这一个小 Batch 搬运到显存里
        states = states.to(device)
        target_regrets = target_regrets.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        predicted_regrets = net(states)

        normalized_weights = weights / (weights.mean() + 1e-8)
        loss = (normalized_weights * (predicted_regrets - target_regrets) ** 2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / UPDATE_STEPS


if __name__ == "__main__":
    print("Start training on device:", device)

    env = TexasEnv()

    # 将两个大脑放入显存
    nets = {0: SD_CFR_ValueNetwork().to(device), 1: SD_CFR_ValueNetwork().to(device)}

    optimizers = {
        0: optim.Adam(nets[0].parameters(), lr=0.001),
        1: optim.Adam(nets[1].parameters(), lr=0.001),
    }

    # Buffer 留在系统内存中，防止爆显存
    buffers = {0: ReservoirBuffer(BUFFER_CAPACITY), 1: ReservoirBuffer(BUFFER_CAPACITY)}

    B_M = {0: [], 1: []}
    start_time = time.time()

    for t in range(1, ITERATIONS + 1):
        # 1. 数据生成阶段
        for traverser in [0, 1]:
            # 加上无梯度上下文，彻底切断反向传播图，防止OOM！
            with torch.no_grad():
                for _ in range(TRAVERSALS_PER_ITER):
                    history = env.reset()
                    traverse(env, history, traverser, t, nets, buffers)

        # 2. 网络训练阶段
        loss_p0 = train_value_network(nets[0], buffers[0], optimizers[0], t)
        loss_p1 = train_value_network(nets[1], buffers[1], optimizers[1], t)

        # 3. 将本轮训练好的大脑快照存入 B^M
        # 把网络权重从显存拉回系统内存再保存，杜绝显存泄漏
        B_M[0].append({k: v.cpu() for k, v in nets[0].state_dict().items()})
        B_M[1].append({k: v.cpu() for k, v in nets[1].state_dict().items()})

        if t % 5 == 0 or t == 1:
            elapsed = time.time() - start_time
            print(
                f"Iter {t:3d}/{ITERATIONS} | "
                f"P0 Loss: {loss_p0:.4f} | P1 Loss: {loss_p1:.4f} | "
                f"Buffer: {len(buffers[0])} | Time: {elapsed:.1f}s"
            )

    print("\n 训练完成！")
    save_path = "texas_sdcfr_models_BM.pth"
    torch.save(B_M, save_path)
    print(f"德扑历史网络池已成功保存至 {save_path}。")
