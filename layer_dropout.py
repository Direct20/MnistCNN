import numpy as np  
from layerbase import *
  
class Dropout(LayerBase):  
    def __init__(self, keep_prob=0.5):  
        super().__init__()
        
        """  
            :param keep_prob: Probability to keep 
        """  
        self.keep_prob = keep_prob  
        self.mask = None  
        self.scale = 1.0 / keep_prob  
        self.train_mode=True
        
    def enable_train_mode(self,enable):
        self.train_mode=enable
    
    def forward(self, X):  
        """  
            :param X
            :return: 
        """  
        if self.train_mode:  
            self.mask = np.random.rand(*X.shape) < self.keep_prob  
            return X * self.mask * self.scale  
        else:  
            return X  
  
    def backward(self, dLdY):  
        """  
            :param dLdY
            :return: 
        """  
        if self.train_mode:  
            return dLdY * self.mask * self.scale  
        else:  
            return dLdY  
  
if __name__ == "__main__":  
    X = np.array([[1, 2, 3]])  
    dropout = Dropout(0.5)  
  
    output_train = dropout.forward(X)  
  
    dout_train = np.ones_like(output_train)  
    gradient_train = dropout.backward(dout_train)  
  
    dropout.enable_train_mode(False)
    output_test = dropout.forward(X)  
  
    dout_test = np.ones_like(output_test)  
    gradient_test = dropout.backward(dout_test)  