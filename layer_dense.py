import numpy as np
from layerbase import LayerBase
import numpy as np

class DenseLayer(LayerBase):
    def __init__(self, input_dim, output_dim, learning_rate=0.01): 
        """
            :param input_dim: 输入维度
            :param output_dim: 输出维度
            :param learning_rate: 学习率
        """ 
        self.W = np.random.randn(input_dim, output_dim)
        self.B = np.zeros((1, output_dim))
        
        self.learning_rate = learning_rate
        
        self.dLdW = np.zeros_like(self.W)
        self.dLdB = np.zeros_like(self.B)

    def forward(self, X):
        """
            :param X: (input_size,)
            return Y: (output_size,)
        """
        self.X = X.reshape(1, -1)  # 将输入重塑为 (1, input_size)
        Y = np.dot(self.X, self.W) + self.B
        return Y.flatten()  # 将输出展平为 (output_size,)

    def backward(self, dLdY):
        """
        :param dLdY: (output_size,)
        :return dLdX: (input_size,)
        """
        dLdY = dLdY.reshape(1, -1)  # 将dLdY重塑为 (1, output_size)
        
        self.dLdW = np.dot(self.X.T, dLdY)
        self.dLdB = dLdY
        dLdX = np.dot(dLdY, self.W.T)

        self.W -= self.learning_rate * self.dLdW
        self.B -= self.learning_rate * self.dLdB
        
        return dLdX.flatten()  # 将dLdX展平为 (input_size,)


if __name__=="__main__":
    input_size = 5
    output_size = 3
    learning_rate = 0.01
    layer = DenseLayer(input_size, output_size, learning_rate)
    X = np.random.randn(input_size)
    Y = layer.forward(X)
  
    dL_dY = np.random.randn(output_size)
    dL_dX = layer.backward(dL_dY)
  
    print("dL/dX:", dL_dX)
    print("Updated weights:", layer.W)
    print("Updated bias:", layer.B)