# Author: Tejaswini Dilp Deore

# import statements
import sys
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from task1 import MyNetwork
import cv2

def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # load the model
    model= MyNetwork()
    model.load_state_dict(torch.load('model.pth'))
    print(model) # print the model

    # get the weights of the first layer
    weights = model.conv1.weight
    print(weights.shape)

    # load training data
    train_loader = DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])))

    # Visualize the filters
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        filter = weights[i, 0].detach().numpy()
        subplt = fig.add_subplot(3, 4, i+1)
        subplt.imshow(filter, cmap=None)
        plt.title(f'Filter {i + 1}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # load the first training example
    image_1, label  = next(iter(train_loader)) 
    image = np.transpose(torch.squeeze(image_1, 1).numpy(), (1, 2, 0))
    with torch.no_grad():
        # apply the filters using OpenCV's filter2D function
        filtered_imgs = []
        for i in range(10):
            im = weights[i, 0]
            filtered_img = cv2.filter2D(np.array(image), -1, np.array(weights[i, 0]))
            filtered_imgs.append(im)
            filtered_imgs.append(filtered_img)

        # Plot the filtered images
        fig = plt.figure(figsize=(10, 10))
        for i in range(20):
            plt.subplot(5, 4, i + 1)
            plt.tight_layout()
            plt.imshow(filtered_imgs[i], cmap='gray', interpolation='none') 
            plt.xticks([])
            plt.yticks([])
        plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
