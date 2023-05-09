# Author: Tejaswini Dilp Deore

# import statements
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from task1 import MyNetwork

'''
This function plots the images and labels
'''
def plot_fig(data, labels, n, rows, cols):
    for i in range(n):
        plt.subplot(rows,cols,i+1)
        plt.tight_layout()
        plt.imshow(data[i][:,:,0], cmap='gray', interpolation='none')
        plt.title('Prediction: {}'.format(labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

'''
This function returns "n" images and labels from the test data
'''
def get_images(test_loader, model, n):
    first_n = []
    labels = []
    count = 0
    for data, target in test_loader:
        if count < n:
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
    return first_n, labels
   
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # load the model
    model= MyNetwork()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # load test data
    test_loader = DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))
    
    # get and plot the first ten images and labels
    first_ten, first_ten_labels = get_images(test_loader, model,10)
    plot_fig(first_ten, first_ten_labels, 9,3,3)

    # load custom digit data, apply the model, and plot the ten results
    image_path = '/home/tejaswini/PRCV/Project5/custom_digits'
    custom_digits = DataLoader(torchvision.datasets.ImageFolder(image_path,
                                         transform=torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                                                       torchvision.transforms.Grayscale(),
                                                                       torchvision.transforms.functional.invert,
                                                                       torchvision.transforms.ToTensor(),
                                                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))])))
    
    first_ten, first_ten_labels= get_images(custom_digits, model, 10)
    plot_fig(first_ten, first_ten_labels, 10,4,3)

    return

if __name__ == "__main__":
    main(sys.argv)
