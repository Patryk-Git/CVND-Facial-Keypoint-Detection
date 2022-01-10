## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
         # (W-F)/S=(224-5)/1+1=220, pooling 110x110, 32 channels
            
        self.conv2 = nn.Conv2d(32, 64, 5)
        # (W-F)/S=(110-5)/1+1=106 pooling 53x53 64 channels
      
        self.conv3 = nn.Conv2d(64, 64, 3)
        # (W-F)/S=(53-3)/1+1=51, pooling 25x25, 64 channels
        
        self.conv4 = nn.Conv2d(64, 128, 3)
        # (W-F)/S=(25-3)/1+1=24, pooling 11x11, 128 channels
        
        self.conv5 = nn.Conv2d(128, 256, 3)
        #(W-F)/S=(11-3)/1+1=9, pooling 4x4, 256 channels
        
        self.fc_drop = nn.Dropout(p=0.25)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 68*2)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.fc_drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.fc_drop(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.fc_drop(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.fc_drop(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.fc_drop(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
