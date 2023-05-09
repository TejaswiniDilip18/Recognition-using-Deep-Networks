# Author: Tejaswini Dilp Deore

import ssl
import sys
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import cv2

# allow using unverified SSL due to some configuration issue
ssl._create_default_https_context = ssl._create_unverified_context

'''
A deep network modified from the pre-trained ResNet from PyTorch
Only contains the first two convolution layers of the original ResNet network
'''
class ResNet(nn.Module):
    # initialize the model
    def __init__(self):
        super(ResNet, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.conv1 = model.layer1[0].conv1 #  keep the first conv layer
        self.conv2 = model.layer1[0].conv2 #  keep the second conv layer
        #self.conv2 = model.conv2 #  keep the second conv layer
        #self.conv2 = list(model.features)[0][1] #  keep the second conv layer

    # compute a forward pass of the first two convolution layerss
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


'''
Initialize the modified ResNet network
Load an image from local directory
Apply the 64 (3x3) filters of the first and second convolution layer to the dog image
Plot the 64 filters 
Plot the first 32 filtered dog images
'''
def main(argv):
    # make the code repeatable
    torch.manual_seed(42)
    # turn off CUDA
    torch.backends.cudnn.enabled = False

    # initialize a model, set to eval mode
    resnet = ResNet()
    resnet.eval()

    # get the weights of the first layer
    weights = resnet.conv1.weight
    print(weights.shape)

    # load data
    custom_image_dir = '/home/tejaswini/PRCV/Project5/resnet'
    custom= datasets.ImageFolder(custom_image_dir,
                                transform =transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

    #  get the first image
    first_image, first_label = next(iter(custom))
    squeezed_image = np.transpose(torch.squeeze(first_image, 1).numpy(), (1, 2, 0))

    # Visualize the filters
    fig = plt.figure(figsize=(10, 10))
    for i in range(64):
        filter = weights[i, 0].detach().numpy()
        subplt = fig.add_subplot(8, 8, i+1)
        subplt.imshow(filter, cmap=None)
        plt.title(f'Filter {i + 1}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    with torch.no_grad():
        # apply the filters using OpenCV's filter2D function
        filtered_imgs = []
        for i in range(64):
            im = weights[i, 0]
            filtered_img = cv2.filter2D(np.array(squeezed_image), -1, np.array(weights[i, 0]))
            filtered_imgs.append(im)
            filtered_imgs.append(filtered_img)

        # Plot the filtered images
        fig = plt.figure(figsize=(10, 10))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.tight_layout()
            plt.imshow(filtered_imgs[i], cmap='gray', interpolation='none') 
            plt.xticks([])
            plt.yticks([])
        plt.show()


if __name__ == "__main__":
    main(sys.argv)
