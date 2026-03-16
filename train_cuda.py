import torch
import torch.optim as optim
import numpy as np
import time
import copy
import os

from texas_env import TexasEnv
from models_cuda import SD_CFR_ValueNetwork, get_strategy_from_value_net
from buffer import ReservoirBuffer

# 检测 NVIDIA 显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前炼丹炉: {device}")

# --- 缩短了时间的验证版参数 ---
ITERATIONS = 200
TRAVERSALS_PER_ITER = 20000  # 降到 2 万，让 CPU 跑得飞快
BATCH_SIZE = 10240
UPDATE_STEPS = 4000
BUFFER_CAPACITY = 5000000  # 降到 500 万，防内存溢出

ACTION_MAP = {0: "f", 1: "c", 2: "s", 3: "h", 4: "p"}


def traverse(env, history, traverser, iteration, nets, buffers):
    """外部采样遍历器 (此时传入的 nets 是 cpu_nets)"""
    is_terminal, _, _ = env.evaluate_history(history)
    if is_terminal:
        payoff = env.get_payoff(history)
        return payoff if traverser == 0 else -payoff

    if env.is_next_round(history) and not history.endswith("/"):
        return traverse(env, history + "/", traverser, iteration, nets, buffers)

    turn = env.get_turn(history)
    state_tensor = env.get_state_tensor(history, turn)  # 产生 CPU 张量
    legal_actions = env.get_legal_actions(history)

    if not legal_actions:
        return 0.0

    # 这里的 nets[turn] 是 CPU 模型
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

        # 将 CPU 数据存入 Buffer
        buffers[traverser].add(state_tensor, regrets, iteration)
        return expected_utility


def train_value_network(net, buffer, optimizer, iteration):
    """GPU 核心训练区"""
    if len(buffer) < BATCH_SIZE:
        return 0.0

    net.train()
    total_loss = 0.0

    for _ in range(UPDATE_STEPS):
        states, target_regrets, weights = buffer.sample(BATCH_SIZE)

        # 把小 Batch 搬运到 3070 显卡里
        states = states.to(device)
        target_regrets = target_regrets.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        predicted_regrets = net(states)  # 在 GPU 上前向传播

        normalized_weights = weights / (weights.mean() + 1e-8)
        loss = (normalized_weights * (predicted_regrets - target_regrets) ** 2).mean()

        loss.backward()  # 在 GPU 上反向传播
        optimizer.step()

        total_loss += loss.item()

    return total_loss / UPDATE_STEPS


if __name__ == "__main__":
    print("🤖 启动 SD-CFR 训练 (双引擎分离架构)...")

    env = TexasEnv()

    # 【GPU 引擎】：负责深度学习训练
    nets = {0: SD_CFR_ValueNetwork().to(device), 1: SD_CFR_ValueNetwork().to(device)}
    # 【CPU 引擎】：负责光速博弈树采样
    cpu_nets = {0: SD_CFR_ValueNetwork().cpu(), 1: SD_CFR_ValueNetwork().cpu()}

    optimizers = {
        0: optim.Adam(nets[0].parameters(), lr=0.001),
        1: optim.Adam(nets[1].parameters(), lr=0.001),
    }

    buffers = {0: ReservoirBuffer(BUFFER_CAPACITY), 1: ReservoirBuffer(BUFFER_CAPACITY)}
    B_M = {0: [], 1: []}
    start_time = time.time()

    for t in range(1, ITERATIONS + 1):

        # 【极其关键的修复】：把 GPU 权重转移给 CPU 时，强制加上 .cpu() 洗掉 CUDA 标签！
        # 这一步阻止了 PyTorch 偷偷把你的 cpu_nets 变成 GPU 模型
        cpu_nets[0].load_state_dict(
            {k: v.cpu() for k, v in nets[0].state_dict().items()}
        )
        cpu_nets[1].load_state_dict(
            {k: v.cpu() for k, v in nets[1].state_dict().items()}
        )

        # 1. 外部采样数据生成阶段 (使用 cpu_nets)
        for traverser in [0, 1]:
            with torch.no_grad():
                for _ in range(TRAVERSALS_PER_ITER):
                    history = env.reset()
                    # 这里传的是 cpu_nets！
                    traverse(env, history, traverser, t, cpu_nets, buffers)

        # 2. 网络训练阶段 (使用 GPU 上的 nets)
        loss_p0 = train_value_network(nets[0], buffers[0], optimizers[0], t)
        loss_p1 = train_value_network(nets[1], buffers[1], optimizers[1], t)

        # 3. 保存快照 (强制洗掉 CUDA 标签再保存)
        B_M[0].append({k: v.cpu() for k, v in nets[0].state_dict().items()})
        B_M[1].append({k: v.cpu() for k, v in nets[1].state_dict().items()})

        if t % 5 == 0 or t == 1:
            elapsed = time.time() - start_time
            print(
                f"Iter {t:3d}/{ITERATIONS} | "
                f"P0 Loss: {loss_p0:.4f} | P1 Loss: {loss_p1:.4f} | "
                f"Buffer: {len(buffers[0])} | Time: {elapsed:.1f}s"
            )

    print("\n✅ 训练完成！")
    save_path = "texas_sdcfr_models_BM.pth"
    torch.save(B_M, save_path)
    print(f"德扑历史网络池已成功保存至 {save_path}。")
