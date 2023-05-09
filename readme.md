# Project Description:
This project involves building, training, analyzing, and modifying a deep network for a recognition task. The project uses MNIST digit recognition data set and PyTorch to build and train a network for digit recognition. This network is then re-used to perform transfer learning on Greek letters (alpha, beta and gamma). The project also includes and experiment which involves changing different aspects of the network (number of epochs, batch size, number of convolution layers, filter size and dropout rate).The project evaluates the first two convolution layers of ResNet18 and analyzes 64 filters and analyzes the network performance after replacing the first layer of the MNIST network with Gabor filters.

# Requirements
The project is tested in the following environment
- ubuntu 20.04

- VScode 1.74.3

- python 3.6


# Instructions for running executables:
Keep all the .py files in one folder

1. To build and train a network to recognize digits (A-F) execute following command 
	python3 task1.py

2. To test handwritten digits dataset, execute following command 
	python3 task1_Test.py

3. To examine your netwrok and analyze the first layer, execute following command
	python3 task2.py

4. To run transfer learning on Greek Letters, execute following command
	python3 task3.py

5. To run a design experiment, execute following command
	python3 task4.py

This experiment involves 64 experiments with 5 dimensions as follows:
- epoch sizes: 3, 5
- training batch sizes: 64, 128
- the number of convolution layers: additional 1 - 4 convolution layers
- convolution layer filter size: 3, 5
- dropout rate: 0.3, 0.5

6. To load and evaluate first two conv layers of ResNet18, execute following command
	python3 resnet.py

7. To replace the first layer of the MNIST network with Gabor filters, execute following command
	python3 gabor_filter.py
