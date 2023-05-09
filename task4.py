# Author: Tejaswini Dilp Deore

# import statements
import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from task1 import train_network, test_network
import csv
import os

batch_size_test = 64 # batch size for testing

# class definition
'''
This code defines a deep neural network that can take different numbers of convolution layers, filter sizes, and dropout rates.
The network architecture consists of the following layers:
    A convolution layer with 10 filters of size 5x5.
    A max pooling layer with a 2x2 window and ReLU activation function.
    Another convolution layer with 20 filters of size 5x5.
    A dropout layer with a rate determined by a parameter passed when the model is initialized.
    Another max pooling layer with a 2x2 window and ReLU activation function.
    Some convolution layers with different filter sizes (determined by a parameter passed when the model is initialized).
    A flattening operation that transforms the output of the previous layers into a 1D tensor.
    A fully connected linear layer with 50 nodes and ReLU activation function.
    A final fully connected linear layer with 10 nodes and a log_softmax activation function applied to the output.
'''
class ExperimentNetwork(nn.Module):
    def __init__(self, num_of_conv, conv_filter_size, dropout_rate):
        super(ExperimentNetwork, self).__init__()
        self.input_size = 28 # input image size is 28x28
        self.num_of_conv = num_of_conv
        self.conv1 = nn.Conv2d(1, 10, kernel_size=conv_filter_size, padding='same')
        self.conv2 = nn.Conv2d(10, 20, kernel_size=conv_filter_size, padding='same')
        self.conv = nn.Conv2d(20, 20, kernel_size=conv_filter_size, padding='same')
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(self.get_fc1_input_size(), 50)
        self.fc2 = nn.Linear(50, 10)

    # The function gets the input size for the first fully connected layer
    def get_fc1_input_size(self):
        fc1_input_size = self.input_size / 2
        fc1_input_size = fc1_input_size / 2
        fc1_input_size = fc1_input_size * fc1_input_size * 20
        return int(fc1_input_size)

    # The function computes a forward path for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        for i in range(self.num_of_conv):
            x = F.relu(self.conv(x))
        # x = x.view(-1, )
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, 1)


'''
The function loads training and test data, initializes a network, trains the network.
The function prints the model accuracy and plots the training and testing losses.
'''
def experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, learning_rate, momentum):
    log_interval = 10
    path = "/home/tejaswini/PRCV/Project5/Experiment/Images"
    filename = "Image " + str(num_epochs) + "_" + str(batch_size_train) + "_" + str(num_of_conv) + "_" + str(conv_filter_size) + "_" + str(dropout_rate) + ".png"

    # load training and testing data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('./Experiment', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train)

    test_loader = DataLoader(
        torchvision.datasets.MNIST('./Experiment', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_test)

    # initialize the network and the optimizer
    network = ExperimentNetwork(num_of_conv, conv_filter_size, dropout_rate)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    accuracy_list = []
    test_counter = [i * len(train_loader.dataset) for i in range(num_epochs + 1)]

    # run the training and testing loops
    test_network(network, test_loader, test_losses) 
    for epoch in range(1, num_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter, accuracy_list)
        test_network(network, test_loader, test_losses)

    # plot training curve
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(os.path.join( path, filename))
    plt.close()

    # Find the maximum accuracy and loss 
    acc= max(accuracy_list)
    loss_train = max(train_losses)
    loss_test = max(test_losses)
    
    data = [num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, learning_rate, acc, loss_train, loss_test]

    # append the data to a csv file
    with open('Experiment.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the data
        writer.writerow(data)

'''
Run 64 experiments using experiment() by modifying the parameters, and display the results
epoch sizes: 3, 5
training batch sizes: 64, 128
the number of convolution layers: add an additional 1 - 4 convolution layers
convolution layer filter size: 3, 5
dropout rate: 0.3, 0.5
'''
def main():
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # set the parameters
    learning_rate = 0.01
    momentum = 0.5

    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # write the header to the csv file
    header = ['Epoch', 'Batch Size', 'Number of Convolutions', 'Filter Size', 'Dropout_Rate', 'Learning_Rate', 'Training Accuracy', 'Training Loss', 'Testing Loss']
    with open('Experiment.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

    # run the experiments with different parameters
    for num_epochs in [3, 5]:  
        for batch_size_train in [64, 128]:
            for num_of_conv in range(1, 5):
                for conv_filter_size in [3, 5]:
                    for dropout_rate in [0.3, 0.5]:
                        print('Number of Epochs: ' + str(num_epochs))
                        print('Train Batch Size: ' + str(batch_size_train))
                        print('Number of Convolution Layer: ' + str(num_of_conv))
                        print('Convolution Filter Size: ' + str(conv_filter_size))
                        print('Dropout Rate: ' + str(dropout_rate))
                        experiment(num_epochs, batch_size_train, num_of_conv, conv_filter_size, dropout_rate, learning_rate, momentum)


if __name__ == "__main__":
    main()
