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
        
        self.add(ConvolutionLayerHardM(np.array( # Sobel filter
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
            Backward propagation for once
            :param Y: Last output of forward propagation (predict result)
            :param A: The expected output of last forward propagation (label)
        """
        dLdX=(Y,A)
        for i in range(len(self.layers)-1,-1,-1):
            layer=self.layers[i]
            if(layer[1]):
                dLdX=layer[0].backward(dLdX)
        return dLdX

    def loss(self, Y, A):
        '''
            Compute loss value
            :param Y: Last output of forward propagation (predict result), 10 dimension vector
            :param A: The expected output of last forward propagation (label), 10 dimension vector
        '''
        Y = np.clip(Y, 1e-15, 1 - 1e-15)  # get rif of log(0)  
        return -np.sum(A*np.log(Y))

    def train(self, X, A, epochs,loss_sample_step=10,evaluate_step=500,evaluate_batch_size=20):
        """
            train the model
            :param X: (n*1*28*28)
            :param A: Label,(n*10)
            :param epochs: Times to train
            :return [[...],[...],...], the 0th element is the loss data of the 1st epoch
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
            Conduct prediction using trained parameters.
            :param X: An image tensor in shape 1*28*28.
            :return A 10d vector, softmax layer raw output.
        '''
        return self.forward(X)
    
    def predict2(self,X):
        '''
            Conduct prediction using trained parameters.
            :param X: An image tensor in shape 1*28*28.
            :return An integer between 0 and 9 representing the class of the image.
        '''
        Y= self.predict(X)  
        return np.argmax(Y) 
    
    def predict3(self,X):
        '''
            Conduct prediction using trained parameters.
            :param X: n*1*28*28.
            :return A tensor in shape n*10, i.e. n raw outputs of softmax layer.
        '''
        Y=[]
        for x in X:
            Y.append(self.predict(x))  
        return np.array(Y) 
    
    def evaluate(self,X,A):
        """
            evaluate the performance of the model using n input examples.
            :param X: (n,1,28,28)
            :param A: (n,10)
            :return (accuracy,macro_precision,macro_recall,confusion_matrix), macro means macro-mean
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
                X[i:i+batch_size],A[i:i+batch_size])# the evaluation data of this batch
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