import torch
from nndl_chap3.activation import softmax


torch.manual_seed(10)

class Op(object):
    def __init__(self,inputs):
        self.inputs = inputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError

# 线性算子
class Linear(Op):
    def __init__(self, dimension):
        """
        输入：
         - dimension:模型要处理的数据特征向量长度
        """
        self.dim = dimension

        # 模型参数
        self.params = {}
        self.params['w'] = torch.randn(size=[self.dim,1],dtype=torch.float32)
        self.params['b'] = torch.randn(size=[1],dtype = torch.float32)

    def __call__(self,X):
        return self.forward(X)
    # 前向传播
    def forward(self, X):
        '''
        输入：
          -X：tensor，shape=[N,D]
          注意这里的X矩阵是由N个x向量的转置拼接的，与原教材表达的特征向量的方式不一样。
        输出：
          -y_pred: tensor,shape=[N]
        :param X:
        :return:
        '''
        N,D = X.shape

        if self.dim ==0:
            return torch.full((N,1),fill_value=self.params['b'].item())
        assert D == self.dim

        y_pred = torch.matmul(X,self.params['w']) + self.params['b']

        return y_pred

#新增Softmax算子
class model_SR(Op):
        def __init__(self, input_dim, output_dim):
            super(model_SR, self).__init__()
            self.params = {}
            #将线性层的权重参数全部初始化为0
            self.params['w'] = torch.zeros(input_dim, output_dim)
            self.params['b'] = torch.zeros(output_dim)
            self.grad = {}
            self.X = None
            self.outputs = None
            self.output_dim = output_dim

        def __call__(self,inputs):
            return self.forward(inputs)
        def forward(self,inputs):
            self.X = inputs
            #线性计算
            score = torch.matmul(self.X,self.params['w'])+self.params['b']
            #Softmax 函数
            self.outputs = softmax(score)
            return self.outputs
        def backward(self,labels):
            '''

            :param labels:
            输入：
              - labels：真实标签，shape=[N,1],其中N为样本数量
            :return:
            '''
            N = labels.shape[0]
            labels = torch.nn.functional.one_hot(labels, self.output_dim)
            self.grads['W'] = -1 / N * torch.matmul(self.X.t(), (labels - self.outputs))
            self.grads['b'] = -1 / N * torch.matmul(torch.ones(shape=[N]), (labels - self.outputs))