import torch
from torch.nn import Module

torch.manual_seed(10)  # 设置随机种子

# 线性算子
class Linear(Module):
    def __init__(self, input_size):
        self.params = {'w': None, 'b': None}
        """
        输入：
           - input_size: 模型要处理的数据特征向量长度
        """
        super(Linear, self).__init__()
        self.input_size = input_size

        # 模型参数
        self.params['w'] = torch.randn(input_size, 1, dtype=torch.float32, requires_grad=True)
        self.params['b'] = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, X):
        """
        输入：
           - X: tensor, shape=[N, D]
           注意这里的X矩阵是由N个x向量的转置拼接成的，与原教材行向量表示方式不一致
        输出：
           - y_pred： tensor, shape=[N]
        """
        N, D = X.shape

        if self.input_size == 0:
            return torch.full((N, 1), fill_value=self.b.item())

        assert D == self.input_size, "输入数据维度不匹配"

        # 使用torch.matmul计算两个tensor的乘积
        y_pred = torch.matmul(X, self.params['w']) + self.params['b']

        return y_pred.squeeze()