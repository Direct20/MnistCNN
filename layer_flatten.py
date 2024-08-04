from layerbase import LayerBase
import numpy as np

class FlattenLayer(LayerBase):
    def __init__(self):
        super().__init__()
        self.last_X_shape=None

    def forward(self, X):
        self.last_X_shape=X.shape
        return X.flatten()

    def backward(self, dLdY):
        return dLdY.reshape(self.last_X_shape)

if(__name__=='__main__'):
    layer=FlattenLayer()
    Y=layer.forward(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    X=layer.backward(Y)
    print(Y,'\n',X)
    pass