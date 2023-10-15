# thanks to Udacity AI Programming with Python Nanodegree Program

import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os

## https://pymotw.com/3/argparse/
## https://docs.python.org/3/library/argparse.html#the-add-argument-method

## use argparse for command line inputs

# Initialize
parser = argparse.ArgumentParser(description='This is a model training program for a dateset of flowers using pytorch',
								 usage='''
        python train.py (data set shall be initially extracted to the 'flowers' directory)
        python train.py data_dir (data set shall be initially extracted to the 'data_dir' directory)
        python train.py data_dir --save_dir save_directory (set directory to save checkpoints)
        python train.py data_dir --arch "vgg13" (choose architecture from vgg13, vgg16 and vgg19)
        python train.py data_dir --learning_rate 0.01 --hidden_units [1024, 512, 256] --epochs 20 (set hyperparameters)''',
								 prog='train')


## Get dataset location, use flowers as default
# https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# Thanks to: Vinay Sajip 
parser.add_argument('data_directory', action="store", nargs='?', default="flowers", help="dataset directory")

## Set directory to save checkpoints
parser.add_argument('--save_dir', action="store", default="", help="saving directory for checkpoint", dest="save_directory")

## Choose architecture:
parser.add_argument('--arch', action="store", default="vgg16", choices=['vgg13', 'vgg16', 'vgg19'],
					 help="you can only choose vgg13, vgg16 or vgg19", dest="architecture")

## Set hyperparameters
parser.add_argument('--learning_rate', action="store", default="0.001", type=float, help="Set Learning rate",
					 dest="learning_rate")
parser.add_argument('--hidden_units', action="store", nargs=3, default=[1024, 512, 256], type=int, help="enter 3 integers between 25088 and 102 in decreasing order",
					 dest="hidden_units")
parser.add_argument('--epochs', action="store", default=3, type=int, help="set epochs", dest="epochs")

## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

arg_data_dir =  args.data_directory
arg_save_dir =  args.save_directory
arg_architecture =  args.architecture
arg_lr = args.learning_rate
arg_hidden_units = args.hidden_units
arg_epochs = args.epochs
# Use GPU if it's selected by user and it is available
if args.gpu and torch.cuda.is_available(): 
	arg_gpu = args.gpu
# if GPU is selected but not available use CPU and warn user
elif args.gpu:
	arg_gpu = False
	print('GPU is not available, will use CPU...')
	print()
# Otherwise use CPU
else:
	arg_gpu = args.gpu

print()
print("Data directory: root/{}/ \nSave directory: root/{} \nArchitecture: {} ".format(arg_data_dir, arg_save_dir, arg_architecture))
print('Learning_rate: ', arg_lr)
print('Hidden units: ', arg_hidden_units)
print('Epochs: ', arg_epochs)
print('GPU: ', arg_gpu)
print()

## Check hidden units 
if 102 <= arg_hidden_units[2] <= arg_hidden_units[1] <= arg_hidden_units[0] <= 25088:
	print("Hidden units are OK.") 
	print()
else:
	arg_hidden_units.extend([1024, 512, 256])
	for i in range(3):
		arg_hidden_units.pop(0)
	
	print("Hidden units are incompatible with the model. Default hidden units {} will be used".format(arg_hidden_units))
	print()

## set data directory locations

data_dir = arg_data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## Define transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

## Build and train the network

# download architecture
if arg_architecture == 'vgg13':
	print('Downloading VGG-13...')
	model = models.vgg13(pretrained=True)
	print()
	print('Model VGG-13: ')
	print()
	print(model)
elif arg_architecture == 'vgg16':
	print('Downloading VGG-16...')
	model = models.vgg16(pretrained=True)
	print()
	print('Model VGG-16: ')
	print()
	print(model)
else:
	print('Downloading VGG-19...')
	model = models.vgg19(pretrained=True)
	print()
	print('Model VGG-19: ')
	print()
	print(model)

# Use GPU if it's selected by user and it is available
device = torch.device("cuda" if arg_gpu else "cpu")
print()
print('Will use {} for training...'.format(device))
print()

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, arg_hidden_units[0]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(arg_hidden_units[0], arg_hidden_units[1]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(arg_hidden_units[1], arg_hidden_units[2]),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),                                 
                                 nn.Linear(arg_hidden_units[2], 102),
                                 nn.LogSoftmax(dim=1))
print('Model Classifier: ')
print(model.classifier)
print()

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=arg_lr)
print('Optimizer: ')
print(optimizer)
print()

model.to(device);

#https://stackoverflow.com/questions/55278566/runtimeerror-expected-object-of-backend-cuda-but-got-backend-cpu-for-argument
print('Training the model................... ')
print('Do not turn off your computer........ ')
print('..................................... ')
epochs = arg_epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
print()
print('Validating the model................. ')
print('Do not turn off your computer........ ')
print('..................................... ')

## Do validation on the test set

# number of total true classifications
total = 0
# number of total images tested
total_length = 0
# total accuracy for test dataset (calculated so far)
total_accuracy = 0
# batch number
batch = 0
for inputs, labels in testloader:
    batch += 1    
    # Move input and label tensors to the default device
    inputs, labels = inputs.to(device), labels.to(device)
   
    
    accuracy = 0
    model.eval()
    with torch.no_grad():

        logps = model.forward(inputs)
                 
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        # this is batch accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        # number of true classifications so far
        total += torch.sum(equals)
        # number of classification attempts so far
        total_length += len(equals)
        # avarage accuracy for the total classification attempts
        total_accuracy = total.item()/total_length
    print(f"Batch {batch}.. "
          f"Accuracy: {accuracy*100:.3f}%.. "
          f"Total Accuracy: {total_accuracy*100:.3f}%")

    model.train()


## Save the checkpoint 

# I have used the fallowing web site as an additional reference source:
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

print()
print('Saving the model..................... ')
print('Do not turn off your computer........ ')
print('..................................... ')


# https://thispointer.com/how-to-create-a-directory-in-python/
# https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty
# Thanks to @Andrew Clark
if arg_save_dir:
	if not os.path.exists(arg_save_dir):
		os.mkdir(arg_save_dir)
		print("Directory " , arg_save_dir ,  " has been created for saving checkpoints")
	else:
		print("Directory " , arg_save_dir ,  " allready exists for saving checkpoints")
	save_dir = arg_save_dir + '/checkpoint.pth'
else:
	save_dir = 'checkpoint.pth'

print()

model.class_to_idx = train_data.class_to_idx
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epoch': epochs,
              'classifier': model.classifier,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'learning_rate': arg_lr,
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, save_dir)

print('Model Saved...')
print()

## this part is for validating saved checkpoint

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epoch = checkpoint['epoch']
    
    return model, optimizer, input_size, output_size, epoch 

print('Validating checkpoint................ ')
print('Loading model........................ ')
print()

my_model, my_optimizer, input_size, output_size, epoch  = load_checkpoint(save_dir)


print('Saved model:')
print(my_model)