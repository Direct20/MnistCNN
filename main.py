from keras.datasets import mnist
from keras.utils import to_categorical
from model import *
import matplotlib.pyplot as plt
import time

# Load MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data pre-process
# Scale the RGB values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert to one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Extends the dimentions
train_images = train_images[:, np.newaxis, ...]
test_images = test_images[:, np.newaxis, ...]

# The number of samples to use for training. The model will use MNIST slice [0:num_train_sample] to train.
num_train_sample = 3000
# The number of samples to use for testing. The model will use MNIST slice [0:num_train_sample] to test.
num_test_sample = 1024

model = Model(learning_rate=0.003)
loss_data, evaluate_data = model.train(train_images[0:num_train_sample],
                                       train_labels[0:num_train_sample],
                                       epochs=1,
                                       loss_sample_step=25, # sample loss data per 25 steps
                                       evaluate_step=25, # evaluate model performance per 25 steps
                                       evaluate_batch_size=20) # evaluate using 20 following images from current image index

evaluate_result = model.evaulate_batches(test_images[0:num_test_sample],
                                         test_labels[0:num_test_sample],
                                         batch_size=128)
print(
    f'Evaluate result: \nAccuracy={evaluate_result[0]}\nPrecision={evaluate_result[1]}\nRecall={evaluate_result[2]}\nF1-score={evaluate_result[3]}'
)

for i in range(len(loss_data)):
    plt.figure()
    plt.subplot(211)
    plt.plot(np.linspace(0, num_train_sample, len(loss_data[i]), dtype=int),
             loss_data[i])
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve, Epoch={i}')

    plt.subplot(212)
    acc_data = [x[0] for x in evaluate_data[i]]
    plt.plot(np.linspace(0, num_train_sample, len(acc_data), dtype=int),
             acc_data)
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title(f'Train Accuracy Curve, Epoch={i}')
    plt.show()

while (1):
    print("Enter an index of test image:")
    x = int(input())

    image = test_images[x]
    predicted = model.predict(image)
    print(f'Label: {test_labels[x]}, Predicted: {predicted}')

    plt.imshow(test_images[x].squeeze(), cmap='gray')
    plt.title(
        f'Label: {np.argmax(test_labels[x])}, Predicted: {np.argmax(predicted)}'
    )
    plt.axis('off')
    plt.show()
