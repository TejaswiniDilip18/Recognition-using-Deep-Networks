# Author: Tejaswini Dilp Deore

# import statements
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from task1 import train_network
from task1 import test_network
import numpy as np
import cv2

'''
This function defines the network with the Gabor filter
The first layer of the network is replaced with a Gabor filter
'''
class GaborNetwork(nn.Module):
    def __init__(self,gabor):
        super(GaborNetwork, self).__init__()
        
        # replace the first layer with a Gabor filter
        self.gabor = gabor        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.5)                 
        self.fc1 = nn.Linear(320 , 50)
        self.fc2 = nn.Linear(50, 10)        
    
    # define the forward pass
    def forward(self, x):
        # apply the Gabor filter
        imgs =[]
        for i in range(len(x)):
            kernels = []
            for j in self.gabor:
                image = cv2.filter2D(x[i][0].detach().numpy(), -1, j)
                H = np.floor(np.array(j.shape)/2).astype(np.int64)                
                image = image[H[0]:-H[0],H[1]:-H[1]]
                kernels.append(image)               
            imgs.append(kernels)
        
        x = torch.from_numpy(np.array(imgs))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320 )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,1) 

# function to generate 10 Gabor filters
def build_gabor_filters():
    gabor_filters=[]
    # loop through 10 different orientations
    for orientatiion in np.arange(0, np.pi, np.pi / 10):        
        # create a gabor filter
        kernel = cv2.getGaborKernel((5, 5),1.0, orientatiion,np.pi/2.0, 0.5, 0, ktype=cv2.CV_32F)
        # normalize the filter
        kernel = kernel / 1.5*kernel.sum()
        # add the filter to the list
        gabor_filters.append(kernel)
    return gabor_filters

# This code loads the MNIST dataset, trains the network with first layer of the MNIST network replaced with a Gabor filter
def main(argv):
    
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # define the hyperparameters for the network
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 15 
 
    # get the MNIST dataset 
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])), batch_size= batch_size_train, shuffle=True)
    test_loader = DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),batch_size=batch_size_test, shuffle=True)
    
    # look at the first six example digits
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    # create a gabor filter
    gabor_filters = build_gabor_filters()
    
    # create a network with the gabor filter
    network = GaborNetwork(gabor_filters)  

    # define the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # initialize the lists for plotting the loss  
    train_losses = []
    train_counter = []
    test_losses = []
    accuracy_list = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    # train and test the network
    test_network(network, test_loader, test_losses)

    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter, accuracy_list)
        test_network(network, test_loader, test_losses)
    
    # plot the loss
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
