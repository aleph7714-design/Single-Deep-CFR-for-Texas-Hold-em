import torch
import torch.optim as optim
import numpy as np
import time
import copy
import os

# --- 注意：这里改为导入新的德州扑克环境 ---
from texas_env import TexasEnv
from models import SD_CFR_ValueNetwork, get_strategy_from_value_net
from buffer import ReservoirBuffer

# 唤醒 MacBook M4 的 MPS 加速引擎 (Apple Silicon GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🔥 当前计算设备: {device}")

# --- 超参数设置 (MacBook M4 加速版) ---
ITERATIONS = 100
TRAVERSALS_PER_ITER = 200
BATCH_SIZE = 1024
UPDATE_STEPS = 2000
BUFFER_CAPACITY = 2000000

# --- 核心修改：动作映射扩容至 5 个 ---
# 0: Fold, 1: Call/Check, 2: Raise(0.2 Pot), 3: Raise(0.5 Pot), 4: Raise(1.0 Pot)
ACTION_MAP = {0: "f", 1: "c", 2: "s", 3: "h", 4: "p"}


def traverse(env, history, traverser, iteration, nets, buffers):
    """
    外部采样遍历器 (External Sampling Traverser)
    注意：这里的 nets 接收的是 cpu_nets
    """
    # 1. 终局判断
    is_terminal, _, _ = env.evaluate_history(history)
    if is_terminal:
        payoff = env.get_payoff(history)
        return payoff if traverser == 0 else -payoff

    # 2. 轮次转换判断
    if env.is_next_round(history) and not history.endswith("/"):
        return traverse(env, history + "/", traverser, iteration, nets, buffers)

    # 3. 获取当前节点信息
    turn = env.get_turn(history)
    state_tensor = env.get_state_tensor(history, turn)
    legal_actions = env.get_legal_actions(history)

    if not legal_actions:
        return 0.0

    # 4. 从当前价值网络获取策略 (在 CPU 上光速推理)
    strategy = get_strategy_from_value_net(nets[turn], state_tensor, legal_actions)

    # ==========================================
    # 分支 A: 当前节点属于对手 (Opponent) -> 采样单一动作
    # ==========================================
    if turn != traverser:
        a = np.random.choice(5, p=strategy)
        action_str = ACTION_MAP[a]
        return traverse(env, history + action_str, traverser, iteration, nets, buffers)

    # ==========================================
    # 分支 B: 当前节点属于遍历者 (Traverser) -> 探索所有合法动作
    # ==========================================
    else:
        expected_utility = 0.0
        action_utilities = np.zeros(5, dtype=np.float32)

        # 遍历所有合法的动作
        for a in legal_actions:
            action_str = ACTION_MAP[a]
            util = traverse(
                env, history + action_str, traverser, iteration, nets, buffers
            )
            action_utilities[a] = util
            expected_utility += strategy[a] * util

        # 计算遗憾值 (Regret)
        regrets = np.zeros(5, dtype=np.float32)
        for a in legal_actions:
            regrets[a] = action_utilities[a] - expected_utility

        # 存入蓄水池 Memory Buffer (只存 CPU 数据)
        buffers[traverser].add(state_tensor, regrets, iteration)

        return expected_utility


def train_value_network(net, buffer, optimizer, iteration):
    """
    使用经验回放池中的数据训练价值网络 (MPS 矩阵加速区)
    """
    if len(buffer) < BATCH_SIZE:
        return 0.0

    net.train()
    total_loss = 0.0

    for _ in range(UPDATE_STEPS):
        # buffer.sample 返回的是存在 CPU 内存上的数据
        states, target_regrets, weights = buffer.sample(BATCH_SIZE)

        # 把这一个小 Batch 瞬间搬运到 M4 的 GPU 显存里
        states = states.to(device)
        target_regrets = target_regrets.to(device)
        weights = weights.to(device)

        optimizer.zero_grad()
        predicted_regrets = net(states)

        # Linear CFR 加权 + 权重归一化 (防止 Adam 梯度爆炸)
        normalized_weights = weights / (weights.mean() + 1e-8)
        loss = (normalized_weights * (predicted_regrets - target_regrets) ** 2).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / UPDATE_STEPS


if __name__ == "__main__":
    print("🤖 启动 SD-CFR 训练 (MacBook M4 MPS 异步加速版)...")

    # 初始化最新的德州扑克环境
    env = TexasEnv()

    # 【MPS 大脑】：专门用来进行大 Batch 的反向传播训练
    nets = {0: SD_CFR_ValueNetwork().to(device), 1: SD_CFR_ValueNetwork().to(device)}

    # 【CPU 分身】：专门用来在博弈树中极速打牌采样
    cpu_nets = {0: SD_CFR_ValueNetwork().cpu(), 1: SD_CFR_ValueNetwork().cpu()}

    # 由于网络变深变宽，可以适当调节学习率
    optimizers = {
        0: optim.Adam(nets[0].parameters(), lr=0.001),
        1: optim.Adam(nets[1].parameters(), lr=0.001),
    }

    buffers = {0: ReservoirBuffer(BUFFER_CAPACITY), 1: ReservoirBuffer(BUFFER_CAPACITY)}

    # 初始化历史网络池 B^M，用于保存模型快照实现轨迹采样
    B_M = {0: [], 1: []}

    start_time = time.time()

    for t in range(1, ITERATIONS + 1):

        # 0. 将 MPS 训练好的最新权重，剥离设备标签后同步给 CPU 分身
        cpu_nets[0].load_state_dict(
            {k: v.cpu() for k, v in nets[0].state_dict().items()}
        )
        cpu_nets[1].load_state_dict(
            {k: v.cpu() for k, v in nets[1].state_dict().items()}
        )

        # 1. 外部采样数据生成阶段
        for traverser in [0, 1]:
            # 必须加入无梯度上下文，防止内存大爆炸
            with torch.no_grad():
                for _ in range(TRAVERSALS_PER_ITER):
                    history = env.reset()
                    # 这里传进去的是 cpu_nets
                    traverse(env, history, traverser, t, cpu_nets, buffers)

        # 2. 网络训练阶段 (传入 MPS 网络和优化器)
        loss_p0 = train_value_network(nets[0], buffers[0], optimizers[0], t)
        loss_p1 = train_value_network(nets[1], buffers[1], optimizers[1], t)

        # 3. 将本轮训练好的大脑快照存入 B^M (存之前必须强制转回 CPU)
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

    # 保存历史网络池，以便稍后用 evaluate.py 载入打牌
    save_path = "texas_sdcfr_models_BM.pth"
    torch.save(B_M, save_path)
    print(f"德扑历史网络池已成功保存至 {save_path}。")
