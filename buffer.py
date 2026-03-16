import random
import numpy as np
import torch


class ReservoirBuffer:
    """
    基于蓄水池采样 (Reservoir Sampling) 的经验回放池。
    用于在有限的内存容量下，保证样本在整个训练历史中的均匀分布。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        # 记录自始至终一共见过的样本总数，这是蓄水池采样算法的核心变量
        self.total_seen = 0

    def add(self, state_tensor, regrets, iteration):
        """
        添加一条经验记录到缓冲区。如果超出容量，使用蓄水池采样算法决定是否替换。

        参数:
        - state_tensor: 当前信息集的状态编码 (PyTorch Tensor)
        - regrets: 该状态下各个合法动作的瞬时反事实遗憾值 (numpy array)
        - iteration: 产生该数据时的迭代轮次 t，用于网络训练时的线性加权 (Linear CFR)
        """
        # 为了节省 MacBook/RTX3070 的宝贵内存，在 Buffer 中一律存为 numpy 数组 (CPU内存)
        if isinstance(state_tensor, torch.Tensor):
            state_array = state_tensor.squeeze(0).cpu().numpy()
        else:
            state_array = state_tensor

        experience = (state_array, regrets, iteration)
        self.total_seen += 1

        # 阶段 1：缓冲区未满，直接追加
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # 阶段 2：缓冲区已满，执行蓄水池采样
            # 生成一个 0 到 total_seen - 1 之间的随机整数
            j = random.randint(0, self.total_seen - 1)
            # 如果随机数落在容量范围内，则替换对应位置的旧样本
            # 随着 total_seen 越来越大，新样本被保留的概率 (capacity / total_seen) 会越来越小
            if j < self.capacity:
                self.buffer[j] = experience

    def sample(self, batch_size):
        """
        从缓冲区中随机抽取一个 Batch 的数据用于训练神经网络。

        返回: 包含 states, regrets, weights(iterations) 的 PyTorch Tensors
        """
        # 如果当前数据量还不够一个 Batch，就取当前所有数据
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, actual_batch_size)

        # 解包 batch 数据
        states = [item[0] for item in batch]
        regrets = [item[1] for item in batch]
        iterations = [item[2] for item in batch]

        # 批量转换为 PyTorch Tensor 以利用张量加速计算
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        regrets_tensor = torch.tensor(np.array(regrets), dtype=torch.float32)
        # 将 iterations 转换为 shape (batch_size, 1) 的张量，方便后续做广播乘法 (Broadcast)
        iterations_tensor = torch.tensor(iterations, dtype=torch.float32).unsqueeze(1)

        return states_tensor, regrets_tensor, iterations_tensor

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """
        清空缓冲区。
        (注意：在 Deep CFR/SD-CFR 的价值网络训练中，通常不需要清空，而是持续累积)
        """
        self.buffer = []
        self.total_seen = 0
