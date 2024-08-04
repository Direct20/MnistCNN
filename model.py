import numpy as np
from layer_conv_hard_mul import *
from layer_activate import *
from layer_pool_mul import *
from layer_flatten import *
from layer_softmax import *
from layer_dense import *
from layer_dropout import *

class Model:
    def __init__(self,learning_rate=0.01):
        self.layers=[]
        
        self.add(ConvolutionLayerHardM(np.array(
        [
            [
                [[-1,-2,-1],
                [0,0,0],
                [1,2,1]],  
            ],
            [

                [[-1,0,1],
                [-2,0,2],
                [-1,0,1]]
            ]
        ],float),learning_rate=learning_rate),False)
        self.add(ActivateLayerM(activation="relu"),False)
        # self.add(PoolLayerM((2, 2),stride=2),False)
        self.add(FlattenLayer(),False)
        # self.add(Dropout(0.2),False)
        self.add(DenseLayer(input_dim=1352,output_dim=128,learning_rate=learning_rate))
        self.add(ActivateLayerM(activation="relu"))
        self.add(DenseLayer(input_dim=128,output_dim=10,learning_rate=learning_rate))
        self.add(SoftmaxLayer())
        
    def add(self,layer,need_backward=True):
        self.layers.append((layer,need_backward))
        
    def forward(self, X):
        Y=X
        for layer in self.layers:
            Y=layer[0].forward(Y)
        return Y

    def backward(self, Y, A):
        """
            反向传播一次
            :param Y: 上一次前向传播的输出
            :param A: 上一次前向传播的预期输出
        """
        dLdX=(Y,A)
        for i in range(len(self.layers)-1,-1,-1):
            layer=self.layers[i]
            if(layer[1]):
                dLdX=layer[0].backward(dLdX)
        return dLdX

    def loss(self, Y, A):
        '''
            计算损失
            :param Y: 预测得到的值,10维向量
            :param A: 标签,10维向量
        '''
        Y = np.clip(Y, 1e-15, 1 - 1e-15)  # 防止log(0)  
        return -np.sum(A*np.log(Y))

    def train(self, X, A, epochs,loss_sample_step=10,evaluate_step=500,evaluate_batch_size=20):
        """
            训练模型
            :param X: (n*1*28*28)
            :param A: 标签,(n*10)
            :param epochs: 训练轮数
            :return [[...],[...],...],第0个元素是第1个epoch的训练loss数据
        """
        
        loss_data=[[]]*epochs
        evaluate_data=[[]]*epochs
        
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch} begin")
            
            
            for i in range(X.shape[0]):
                Y = self.forward(X[i])
                self.backward(Y, A[i])

                if(i%loss_sample_step==0):
                    loss=self.loss(Y,A[i])
                    loss_data[epoch-1].append(loss)
                    print(f"loss {i} = {loss}")
                    
                if(i%evaluate_step==0):
                    evaluate_data[epoch-1].append(self.evaluate(X[i:i+evaluate_batch_size],A[i:i+evaluate_batch_size]))
                    
                    
            print(f"epoch {epoch} end")
            
            
        return loss_data,evaluate_data
    
    
    def predict(self,X):
        '''
            使用训练好的参数进行预测
            :param X: 一张1*28*28的图片张量
            :return 维度为10的向量
        '''
        return self.forward(X)
    
    def predict2(self,X):
        '''
            使用训练好的参数进行预测
            :param X: 一张1*28*28的图片张量
            :return 所属类别0..9
        '''
        Y= self.predict(X)  
        return np.argmax(Y) 
    
    def predict3(self,X):
        '''
            使用训练好的参数进行预测
            :param X: n*1*28*28
            :return 所属类别n*10
        '''
        Y=[]
        for x in X:
            Y.append(self.predict(x))  
        return np.array(Y) 
    
    def evaluate(self,X,A):
        """
            使用输入的n个样本,评估模型的性能
            :param X: (n,1,28,28)
            :param A: (n,10)
            :return (accuracy,macro_precision,macro_recall,confusion_matrix) macro意为宏平均
        """
        num_class,num_sample=A.shape[1],X.shape[0]
        
        confusion_matrix=np.zeros((num_class,num_class))
        
        accuracy=0.
        precision=np.zeros(num_class)
        recall=np.zeros(num_class)
        
        for i in range(num_sample):
            image_tensor=X[i]
            label_tensor=A[i]
            
            label=np.argmax(label_tensor)
            p_label=self.predict2(image_tensor)
            
            confusion_matrix[label,p_label]+=1

        # accuracy
        accuracy=np.sum(np.diagonal(confusion_matrix))/num_sample
            
        # precision
        for i in range(num_class):
            sum=np.sum(confusion_matrix[:,i])
            precision[i]=confusion_matrix[i,i]/sum if sum!=0 else 1e-9
        macro_precision=np.sum(precision)/num_class
        
        # recall
        for i in range(num_class):
            sum=np.sum(confusion_matrix[i,:])
            recall[i]=confusion_matrix[i,i]/sum if sum!=0 else 1e-9
        macro_recall=np.sum(recall)/num_class
        
        # F1_score
        F1_score=2*macro_precision*macro_recall/(macro_recall+macro_precision)
        
        return (accuracy,macro_precision,macro_recall,F1_score,confusion_matrix)
    
    
    def evaulate_batches(self,X,A,batch_size=128):
        """
            :param X (N,1,28,28)
            :param A (N,10)
            :return (accuracy,macro_precision,macro_recall)
        """
        num_batch=X.shape[0]//batch_size
        accuracy=np.zeros(num_batch)
        macro_precision=np.zeros(num_batch)
        macro_recall=np.zeros(num_batch)
        macro_F1_score=np.zeros(num_batch)
        print("Evaluation begin")
        for i in range(num_batch):
            print(f"Evaluating batch {i+1}/{num_batch}")
            (accuracy[i],macro_precision[i],macro_recall[i],macro_F1_score[i],_)=self.evaluate(
                X[i:i+batch_size],A[i:i+batch_size])# 这个batch的评估数据
        print("Evaluation end")
        return (np.average(accuracy),np.average(macro_precision),np.average(macro_recall),np.average(macro_F1_score))
    
    
    def pr(self,Y,A):
        """
            :param Y: (n*10)
            :param A: (n*10)
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import matplotlib.pyplot as plt
        
        num_class=A.shape[1]
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(num_class):
            precision[i],recall[i],_ = precision_recall_curve(A[:,i],Y[:,i])
            average_precision[i] = average_precision_score(A[:,i],Y[:,i])
            
        precision["micro"],recall["micro"],_ = precision_recall_curve(A.ravel(),Y.ravel())
        average_precision["micro"] = average_precision_score(A,Y,average="micro") 
        
        plt.clf()
        plt.plot(recall["micro"],precision["micro"],label = "micro_average P_R(area={0:0.2f})".format(average_precision["micro"]))
        for i in range(num_class):
            plt.plot(recall[i],precision[i],label = "P_R curve of class{0}(area={1:0.2f})".format(i,average_precision[i]))
        
        plt.xlim([0.0,0.1])
        plt.ylim([0.0,1.05])
        plt.legend(loc = "lower right")
        plt.show()