import torch


def optimizer_lsm(model, X, y, reg_lambda=0):
    """
  输入：
     - model: 模型
     - X: tensor, 特征数据，shape=[N,D]
     - y: tensor, 标签数据，shape=[N]
     - reg_lambda: float, 正则化系数，默认为0
  输出：
     - model: 优化好的模型
  """

    N, D = X.shape

    # 对输入特征数据所有特征向量求平均
    x_bar = torch.mean(X, dim=0).mT

    # 求标签的均值, shape=[1]
    y_bar = torch.mean(y)

    # 通过广播的方式实现矩阵减向量
    x_sub = X - x_bar.view(1, -1)

    # 检查 x_sub 是否全为0
    if torch.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = torch.zeros(D)
        return model

    # 求方阵的逆
    tmp = torch.inverse(torch.matmul(x_sub.mT, x_sub) +
                        reg_lambda * torch.eye(D))

    # 计算 w
    w = torch.matmul(torch.matmul(tmp, x_sub.mT), (y - y_bar))

    # 计算 b
    b = y_bar - torch.matmul(x_bar, w)

    model.params['b'] = b
    model.params['w'] = w.squeeze(axis=-1)

    return model
