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
import pdb 

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval=10):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    log_interval: (default=10) interval to log training stats in
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        print(data.shape)
        print(target)
        exit()
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Get predicted index by selecting maximum log-probability
        pred = output.argmax(dim=1, keepdim=True)
        
        # Count correct predictions overall 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Print log every N intervals
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        # Override exit to speed up 
        #if batch_idx > 100:
        #    break 
    
    train_loss = float(np.mean(losses))
    train_acc = correct / ((batch_idx+1) * config.batch_size)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, (batch_idx+1) * config.batch_size,
        100. * correct / ((batch_idx+1) * config.batch_size)))
    return train_loss, train_acc
    


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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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
    
    # Initialize the criterion for loss computation 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Initialize optimizer type 
    if config.optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        print("Use optimizer type: {}, LR: {}".format(config.optimizer_type, config.learning_rate))
    elif config.optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        print("Use optimizer type: {}, LR: {}".format(config.optimizer_type, config.learning_rate))
    else:
        print("Select optimizer type from {SGD | Adam}")
        exit(0)
    
    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = datasets.MNIST('./data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,
                       transform=transform)
    train_loader = DataLoader(dataset1, batch_size = config.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = config.batch_size, 
                                shuffle=False, num_workers=4)
    
    
    # Optionally, use a scheduler to change learning rate at certain interval manually
    # Used for step LR change, cyclic LR change or manual LR change after some epochs
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=0.1)
    
    # Init variable to store best loss, can use for saving best model 
    best_accuracy = 0.0
    
    # Create summary writer object in specified folder. 
    # Use same head folder and different sub_folder to easily compare between runs
    # Eg. SummaryWriter("my_logs/run1_Adam"), SummaryWriter("my_logs/run2_SGD")
    #     This allows tensorboard to easily compare between run1 and run2
    writer = SummaryWriter("my_logs/run1_Adam", comment="Test_01_LR_1e-3")
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, config.n_epochs + 1):
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, 
                                            log_interval = 50)
        test_loss, test_accuracy = test(model, device, test_loader)
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)        
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if test_accuracy > best_accuracy and config.save:
            best_accuracy = test_accuracy
            save_file_path = os.path.join(config.save_dir, 'model_{}_{:2.2f}.pth'.format(epoch, best_accuracy))
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_accuracy
            }

            try:
                os.mkdir(config.save_dir)
            except:
                pass

            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))
            
            # Alternatively same entire model, but takes larger size
            # torch.save(model, save_file_path)
        
        #if epoch % 5 == 0:
        #    break 
    
    # Flush all log to writer and close 
    writer.flush()
    writer.close()
    
    print("Training finished")
    
    
if __name__ == '__main__':
    run_main()
    
    