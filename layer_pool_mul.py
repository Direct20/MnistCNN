from layerbase import LayerBase
import numpy as np
import math

class PoolLayerM(LayerBase):
    def __init__(self, kernel_size,stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride=stride
        self.last_X=None

    def forward(self, X):
        self.last_X=np.copy(X)

        X_channel,X_row,X_col=X.shape
        K_row,K_col,stride=self.kernel_size[0],self.kernel_size[1],self.stride
        
        Y_row=1+(X_row-K_row)//stride
        Y_col=1+(X_row-K_col)//stride
        Y=np.zeros((X_channel,Y_row,Y_col))
        
        for c in range(X_channel):
            for i in range(Y_row):
                h_stride=i*stride
                for j in range(Y_col):
                    w_stride=j*stride
                    patch=X[c,h_stride:h_stride+K_row,w_stride:w_stride+K_col]
                    Y[c,i,j]=np.max(patch)
        
        return Y

    def backward(self, dLdY):
        """
            :param dLdY: L对Y的导数矩阵,可以分为dLdY_11 dLdY_12 ...
            :return dLdX: L对X的导数矩阵,可以分为dLdX_11 dLdx_12
        """
        X=self.last_X
        X_Channel,X_row,X_col=X.shape
        K_row,K_col,stride=self.kernel_size[0],self.kernel_size[1],self.stride

        dLdX_row=1+(X_row-K_row)//stride
        dLdX_col=1+(X_col-K_col)//stride
        dLdX=np.zeros(X.shape)
        
        for l in range(X_Channel):
            for i in range(dLdX_row):
                h_stride=stride*i
                for j in range(dLdX_col):
                    w_stride=stride*j
                    patch=X[l,h_stride:h_stride+K_row,w_stride:w_stride+K_col]
                    patch_max=np.max(patch)
                    dLdX[l,h_stride:h_stride+K_row,w_stride:w_stride+K_col]+=(patch_max==patch)*dLdY[l,i,j]
        return dLdX


if(__name__=='__main__'):
    # layer=PoolLayer(4,4,(2,2))
    # Y=layer.forward(np.array([[1,2,3,4],
    #                           [5,6,7,8],
    #                           [9,0,1,2],
    #                           [3,4,5,6]]))
    # print(Y)
    # print(layer.Z)
    # X=layer.backward(Y)
    # print(X)
    X=np.random.randn(2,8,8)
    df=np.random.randn(2,4,4)
    
    pool=PoolLayerM((2,2),2)
    f=pool.forward(X)
    dx=pool.backward(df)
    print(dx)
    pass