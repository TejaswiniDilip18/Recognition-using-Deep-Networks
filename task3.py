# Author: Tejaswini Dilp Deore

# import statements
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from task1 import MyNetwork, train_network

greek_letters = ['alpha', 'beta', 'gamma'] # list of greek letters

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )


def main(argv):
    # make the code repeatable
    torch.manual_seed(35)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # define the parameters
    n_epochs = 70 # number of epochs to train the model
    learning_rate = 0.01 # learning rate for the optimizer
    momentum = 0.5 # momentum for the optimizer
    log_interval = 2 # define how many batches to wait before logging training status

    # load the model pretrained on MNIST
    model= MyNetwork()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    
    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False
    model.fc2= torch.nn.Linear(50, 3)
    print(model)

    # DataLoader for the Greek data set
    training_set_path = '/home/tejaswini/PRCV/Project5/greek_train'
    greek_train = DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
                                          transform = torchvision.transforms.Compose( [
                                            torchvision.transforms.ToTensor(),
                                            GreekTransform(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,) 
                                                ) ] ) ), batch_size = 5, shuffle = True )

    # DataLoader for the custom Greek data set
    test_set_path = '/home/tejaswini/PRCV/Project5/custom_greek'

    greek_test = DataLoader(torchvision.datasets.ImageFolder(test_set_path,
                                         transform=transforms.Compose([transforms.Resize((28, 28)),
                                                                       transforms.Grayscale(),
                                                                       transforms.functional.invert,
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize((0.1307,), (0.3081,))])))

    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    train_losses = []
    train_counter = []
    accuracy_list = []
    
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, model, greek_train, optimizer, log_interval, train_losses, train_counter, accuracy_list)

    # plot the training and test losses
    plt.plot(train_counter, train_losses, color='blue')
   # plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # save the model
    torch.save(model.state_dict(), 'Greek_model.pth')
    torch.save(optimizer.state_dict(), 'Greek_optimizer.pth')    

    first_n = []
    labels = []
    count = 0
    # get the images
    for data, target in greek_test:
        if count < 13:
            squeeze_data = np.transpose(torch.squeeze(data, 1).numpy(), (1, 2, 0)) # squeeze the data and convert to numpy array
            first_n.append(squeeze_data)
            with torch.no_grad():
                output = model(data)
                print(f'{count + 1} - output: {output}')
                print(f'{count + 1} - index of the max output value: {output.argmax().item()}')
                label = output.data.max(1, keepdim=True)[1].item()
                print(f'{count + 1} - prediction label: {label}')
                labels.append(label)
                count += 1
    # plot the images
    for i in range(13):
        plt.subplot(6,3,i+1)
        plt.tight_layout()
        plt.imshow(first_n[i][:,:,0], cmap='gray', interpolation='none')
        plt.title('Prediction: {}'.format(greek_letters[labels[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
