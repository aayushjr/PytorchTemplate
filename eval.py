from __future__ import print_function
import argparse
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from model import Net
import numpy as np
import config
 

def test(model, device, test_loader):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    test_loss = 0
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training 
            loss = F.cross_entropy(output, target, reduction='sum')
            
            # Append loss to overall test loss
            test_loss += loss.item()
            
            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)
            
            # Count correct predictions overall 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:2.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, accuracy
    

def run_main():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = Net().to(device)
    
    
    # Load checkpoint.
    print('==> Loading from checkpoint..')
    model_file_path = os.path.join(config.save_dir, config.checkpoint)
    assert os.path.exists(model_file_path), 'Error: no checkpoint found!'
    checkpoint = torch.load(model_file_path)
    
    model_weights = checkpoint['state_dict']
    best_acc = checkpoint['best_accuracy']
    epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    
    # Load weights to model 
    model.load_state_dict(model_weights)
    
    '''
    For new pytorch versions, you can save/load entire model 
    model = torch.load(PATH_TO_MODEL)
    '''
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    test_loader = DataLoader(dataset2, batch_size = config.batch_size, 
                                shuffle=False, num_workers=4)
    
    # Evaluate the model 
    test_loss, test_accuracy = test(model, device, test_loader)
        
    print("Eval finished")
    
    
if __name__ == '__main__':
    run_main()