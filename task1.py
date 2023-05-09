# Author: Tejaswini Dilp Deore

# import statements
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

'''
Task 1:
Get the MNIST digit data set
Look at the first six example digits
Build a network model
Train the model
Save the model
Read the network and run it on the test set
'''

# class definition
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,1)

'''
This function trains the model for a specified number of epochs and calculates the training accuracy and loss
'''
def train_network(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter, accuracy_list):
  network.train()
  correct = 0
  for batch_idx, (data, target) in enumerate(train_loader):
    output = network(data)
    optimizer.zero_grad()    
    loss = F.nll_loss(output, target)
    correct += (output.argmax(1) == target).type(torch.float).sum().item()    
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*len(data)) + ((epoch-1)*len(train_loader.dataset)))
  #calculate the accuracy
  accuracy = 100. * correct / len(train_loader.dataset)
  accuracy_list.append(accuracy)
  print('Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset), accuracy))

'''
This function evaluates the model on a test dataset and calculate the test accuracy and loss.
'''
def test_network( model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

'''
Load the MNIST dataset, create a neural network model, train and evaluate the model, and save the model and optimizer state.
'''
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # set the parameters
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    # get the MNIST dataset
    train_loader = DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])), batch_size=batch_size_train, shuffle=True)
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

    # display the first six example digits
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i])) 
        plt.xticks([])
        plt.yticks([])
    plt.show()

    model= MyNetwork() # build a network model
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # define the optimizer
    
    train_losses = []
    train_counter = []
    test_losses = []
    accuracy_list = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    test_network( model, test_loader, test_losses )

    for epoch in range(1, n_epochs + 1):
        #train_network( model, train_loader, optimizer, epoch, log_interval, train_losses, train_counter,batch_size_train )
        train_network(epoch, model, train_loader, optimizer, log_interval, train_losses, train_counter, accuracy_list)
        test_network( model, test_loader, test_losses )

    # plot the training and test losses
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # save the model
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')

    return

if __name__ == "__main__":
    main(sys.argv)
