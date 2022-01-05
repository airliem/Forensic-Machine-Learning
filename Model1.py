from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Transfer learning model used for image classification with CovNet as fixed feature extractor 

# Interactive mode on
plt.ion()

# Image augmentation and normalisation for loading data to model
# Augmentation and normalization of training data and normalization of testing data
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Sets folderpath to folders containing the images 
data_dir = '/home/melissa/Documents/Hazel_Lara_Pilot/Hazel_Lara_Dataset/'

# Loads dataset from folderpath and applies transforms
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Provides data to model (4 images per batch, random, 4 parallel data imput)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Uses GPU-cuda else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# General function to visualise images after transforms (check augmentations of training data)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Gets a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Makes grid from batch
out = torchvision.utils.make_grid(inputs)

# Shows images for each label
imshow(out, title=[class_names[x] for x in classes])

# General function to train a model
def train_model(model, criterion, optimizer, scheduler, num_epochs=15):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # For each epoch is a training and testing (val) phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterates over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zeros the parameter gradients
                optimizer.zero_grad()

                # Forward 
                # Tracks history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward 
                    # Optimizes only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Runs statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Prints epoch loss and accuracy 
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copies best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    # Prints training time and best model
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Loads best model weights
    model.load_state_dict(best_model_wts)
    return model

# General function to display model predictions for some images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# ConvNet as fixed feature extractor
# Loads pre-trained resnet18 model, freeze network so gradients are not computed backwards
model_conv = torchvision.models.resnet18(pretrained=True)

# Turns off training for pre-trained model (keep pre-trained model weights)
for param in model_conv.parameters(): 
    param.requires_grad = False
num_ftrs = model_conv.fc.in_features

# Output set to number of labels
model_conv.fc = nn.Linear(num_ftrs, 3)

# Uses GPU-cude else CPU
model_conv = model_conv.to(device)

# For each epoch, loss calculated using the criterion function to fine tune the model
criterion = nn.CrossEntropyLoss()

# Parameters of final layer optimized, implements stochastic gradient decent with learining rate 0.001 and momentum 0.9
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decays learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# Fine tunes the model and evaluation (call function)
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)

# Interactive mode off
plt.ioff()
plt.show() 

# Defines path
PATH = 'state_dict_model.pt'

# Saves model
torch.save(model_conv.state_dict(), PATH)