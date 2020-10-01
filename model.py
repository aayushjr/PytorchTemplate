import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, debug=False):
        super(Net, self).__init__()
        
        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(in_channels=1, 
                                out_channels=32,
                                kernel_size=3,
                                stride=1)
                                #padding=1)
                                
        self.conv2 = nn.Conv2d(32, 64, (3,3), (1,1))
        
        # Add dropout to randomly make some pixels 0 with probability p during training
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.debug = debug

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.debug:
            print("conv1:\t\t", x.shape)
            
        x = self.conv2(x)
        x = F.relu(x)
        if self.debug:
            print("conv2:\t\t", x.shape)
        
        x = F.max_pool2d(x, 2)
        if self.debug:
            print("max_pool:\t", x.shape)
        
        x = self.dropout1(x)
        if self.debug:
            print("dropout1:\t", x.shape)
        
        x = torch.flatten(x, start_dim=1)
        #x = x.view(-1, 9216)
        if self.debug:
            print("flatten:\t", x.shape)
        
        x = self.fc1(x)
        x = F.relu(x)
        if self.debug:
            print("fc1:\t\t", x.shape)
        
        x = self.dropout2(x)
        if self.debug:
            print("dropout2:\t", x.shape)
        
        x = self.fc2(x)
        if self.debug:
            print("fc2:\t\t", x.shape)
        
        # Depends on criterion. CE Loss from nn already has log_softmax applied.
        #output = F.log_softmax(x, dim=1)
        #if self.debug:
        #    print("Softmax:\t", x.shape)
        
        output = x
        
        return output
        
if __name__=='__main__':
    
    net = Net(debug=True)
    data = torch.rand((1, 1, 28, 28))
    print(net)
    print("Data shape: ", data.shape)
    net(data)