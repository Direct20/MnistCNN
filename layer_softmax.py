from layerbase import LayerBase
import numpy as np
import math

class SoftmaxLayer(LayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
            :param X: FC层输出
            :return 向量,归一化的分类结果
        """
        Z = np.exp(X - np.max(X))  # 防止溢出  
        return Z / np.sum(Z)

    def backward(self, Y_A):
        """
            :param Y_A=(Y,A) Y,Softmax层输出;A,标签,预期输出
            :return dL/dX
        """
        return Y_A[0] - Y_A[1]

if(__name__=='__main__'):
    layer=SoftmaxLayer()
    Y=layer.forward(np.array([0.1,0.8]))
    X=layer.backward((Y,np.array([0,1])))
    print(Y,'\n',X)
    pass