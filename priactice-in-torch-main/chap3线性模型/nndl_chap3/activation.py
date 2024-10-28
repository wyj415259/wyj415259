import torch

def softmax(X):
    """
    计算给定张量X的softmax值。

    Softmax函数将每个元素转换为概率值，使得所有元素的总和为1。这对于将输出层的原始分数转换为概率分布特别有用。

    参数:
    X: 一个任意形状的PyTorch张量，表示网络的原始输出分数。

    返回值:
    一个具有与输入X相同形状的PyTorch张量，其中每个元素都是softmax概率值。
    """
    # 找到每行的最大值并保持维度，用于数值稳定性
    x_max = torch.max(X, dim=1, keepdim=True)
    # 计算指数，这是softmax函数的第一部分
    x_exp = torch.exp(X - x_max)
    # 计算指数的和，这是softmax函数的分母
    partition = torch.sum(x_exp, dim=1, keepdim=True)
    # 返回softmax概率值
    return x_exp / partition
