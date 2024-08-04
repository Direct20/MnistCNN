from layerbase import LayerBase
import numpy as np
import math

class ConvolutionLayerHardM(LayerBase):
    def __init__(self,kernels,padding=0,stride=1, learning_rate=0.01):
        """
            :param kernels: Convolution kernels, in shape (num,channel,height,width)
            :param padding: padding length
            :param stride: 
            :param learning_rate: 
        """
        super().__init__()

        self.in_channels=kernels.shape[1]
        self.out_channels=kernels.shape[0]
        self.kernel_size = (kernels.shape[2],kernels.shape[3])
        self.padding=padding
        self.stride=stride
        self.learning_rate = learning_rate

        self.kernel = kernels#np.random.randn((self.out_channels,self.in_channels,self.kernel_size[0], self.kernel_size[1]))
        self.B = np.zeros_like(self.kernel)
        self.last_X = None

    def forward(self, X):
        """
            :param X: Input, in shape (channel,row,col)
            :return Output, in shape (channel,row,col), the number of channels eqs kernels
        """
        X_channel,X_row,X_col=X.shape
        _,_,K_row,K_col=self.kernel.shape

        X_padded=np.pad(X,((0,0),(self.padding,self.padding),(self.padding,self.padding)),mode="constant")

        Y_row = math.floor((X_row +2*self.padding- K_row )/self.stride)+ 1
        Y_col =  math.floor((X_col+2*self.padding - K_col )/self.stride)+ 1
        Y = np.zeros((self.out_channels,Y_row, Y_col))

        for k in range(0,self.out_channels):
            for i in range(0, Y_row, 1):
                h_stride=i*self.stride
                for j in range(0, Y_col, 1):
                    w_stride=j*self.stride
                    patch = X_padded[:,h_stride:h_stride+K_row,w_stride:w_stride+K_col]# extract a patch of input
                    Y[k,i, j] += np.sum(patch * self.kernel[k] + self.B[k])

        self.last_X = np.copy(X)
        return Y

    def backward(self, dLdY):
        """
            :param dLdY: (channel,row,col), the number of channels eqs kernels of current layer.
            :return dLdX,(channel,row,col), the number of channels eqs channels of X.
        """
        X = self.last_X
        
        K,dLdY_row,dLdY_col=dLdY.shape
        C,X_row,X_col=X.shape
        K,_,K_row,K_col=self.kernel.shape
        
        padding=self.padding
        
        H_=1+(X_row+2*padding-K_row)//self.stride
        W_=1+(X_col+2*padding-K_col)//self.stride
        
        dLdX=np.zeros_like(X)
        dLdK = np.zeros_like(self.kernel)  # dL/dK, a multi-channel matrix, has the same dim with kernel
        X_padded=np.pad(X,[(0,0),(padding,padding),(padding,padding)],'constant')
        dLdX_padded=np.pad(dLdX,[(0,0),(padding,padding),(padding,padding)],'constant')

        for f in range(K):
            for i in range(H_):
                h_stride=i*self.stride
                for j in range(W_):
                    w_stride=j*self.stride
                    dLdK[f] += X_padded[:,h_stride:h_stride + K_row, w_stride:w_stride + K_col]*dLdY[f,i,j]
                    dLdX_padded[:,h_stride:h_stride+K_row,w_stride:w_stride+K_col]+=self.kernel[f]*dLdY[f,i,j]

        dLdX=dLdX_padded[:,padding:padding+X_row,padding:padding+X_col]
        
        # self.kernel -= self.learning_rate * dLdK
        return dLdX
    
    def save_parameters(self):
        pass
    
    def load_parameters(self):
        pass


if(__name__=='__main__'):
    # layer=ConvolutionLayer(4,4,(2,2))
    # layer.kernel=[[1,1],[1,1]]
    # print(layer.kernel)
    # Y=layer.forward(np.array([[1,2,3,4],
    #                           [5,6,7,8],
    #                           [9,0,1,2],
    #                           [3,4,5,6]]))
    # print(Y)
    # dLdK=layer.backward(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    # print(dLdK)
    # print(layer.kernel)
    pass