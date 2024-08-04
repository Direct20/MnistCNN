from layerbase import LayerBase
import numpy as np
import function as f

class ActivateLayerM(LayerBase):
    def __init__(self,activation="relu"):
        super().__init__()
        self.activation=activation
        pass
        
    def activate(self,X):
        Y=None
        if(self.activation=="relu"):
            Y=f.ReLU(X)
        elif(self.activation=="sigmoid"):
            Y=f.sigmoid(X)
        else:
            Y=X
        return Y
    def d_activate(self,X):
        Y=None
        if(self.activation=="relu"):
            Y=f.dReLU(X)
        elif(self.activation=="sigmoid"):
            Y=f.dsigmoid(X)
        else:
            Y=1
        return Y
        
    def forward(self,X):
        """
            :param X: 输入,(channel,height,width)
            :return f(X)
        """
        self.Y=self.activate(X)
        self.last_X=np.copy(X)
        return self.Y
        
    def backward(self,dLdY):
        dLdX=dLdY*self.d_activate(self.Y)
        return dLdX
        