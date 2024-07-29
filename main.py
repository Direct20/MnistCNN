from keras.datasets import mnist
from keras.utils import to_categorical
from model import *
import matplotlib.pyplot as plt
import time

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
# 将像素值缩放到0-1之间
train_images = train_images / 255.0
test_images = test_images / 255.0   

# 转换为one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 维度扩展  
train_images = train_images[:,np.newaxis,...]  
test_images = test_images[:,np.newaxis,...]  

num_train_sample=3000
num_test_sample=1024

model = Model(0.003)
loss_data,evaluate_data=model.train(train_images[0:num_train_sample], train_labels[0:num_train_sample], 
                      epochs=1,loss_sample_step=25,evaluate_step=25,evaluate_batch_size=20)


evaluate_result=model.evaulate_batches(test_images[0:num_test_sample],test_labels[0:num_test_sample],128)
print(f'Evaluate result: \nAccuracy={evaluate_result[0]}\nPrecision={evaluate_result[1]}\nRecall={evaluate_result[2]}\nF1-score={evaluate_result[3]}')

for i in range(len(loss_data)):
    plt.figure()
    plt.subplot(211)
    plt.plot(np.linspace(0,num_train_sample,len(loss_data[i]),dtype=int),loss_data[i])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve, Epoch={i}')

    plt.subplot(212)
    acc_data=[x[0] for x in evaluate_data[i]]
    plt.plot(np.linspace(0,num_train_sample,len(acc_data),dtype=int),acc_data)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title(f'Train Accuracy Curve, Epoch={i}')
    plt.show()
    
while(1):
    print("Enter an index of test image:")
    x = int(input())
    
    image = test_images[x]
    predicted=model.predict(image)
    print(f'Label: {test_labels[x]}, Predicted: {predicted}')
    
    plt.imshow(test_images[x].squeeze(), cmap='gray')
    plt.title(f'Label: {np.argmax(test_labels[x])}, Predicted: {np.argmax(predicted)}')
    plt.axis('off')
    plt.show()
