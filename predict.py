# thanks to Udacity AI Programming with Python Nanodegree Program

import argparse
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import random
from PIL import Image
import numpy as np

## A function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
	model = models.vgg16(pretrained=True)

	
	checkpoint = torch.load(filepath)
	lr = checkpoint['learning_rate']
	model.classifier = checkpoint['classifier']
	model.load_state_dict(checkpoint['model_state_dict'])
	model.class_to_idx = checkpoint['class_to_idx']
	optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	input_size = checkpoint['input_size']
	output_size = checkpoint['output_size']
	epoch = checkpoint['epoch']


	return model, optimizer, input_size, output_size, epoch

# https://pillow.readthedocs.io/en/latest/reference/Image.html
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
# https://www.najeebhassan.com/ImageClassifierProject.html 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Get the image
    my_image = Image.open(image)
    # resize the image
    my_image = my_image.resize((256, 256))
    # crop the image
    (left, upper, right, lower) = (16, 16, 240, 240)
    
    my_image = my_image.crop((left, upper, right, lower))
    
    # convert image
    np_image = np.array(my_image)/255
    # normalize the arrays
    np_image = (np_image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    # return transpose
    return np_image.transpose(2, 0, 1)

# https://discuss.pytorch.org/t/runtimeerror-expected-4-dimensional-input-for-4-dimensional-weight-6-3-5-5-but-got-3-dimensional-input-of-size-3-256-256-instead/37189
# thanks to @ ptrblck
# https://github.com/huggingface/transformers/issues/227
# thanks to @ nhatchan 
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device);
    model.eval()
    with torch.no_grad():
        my_image = process_image(image_path)
        
        my_image = torch.from_numpy(my_image).unsqueeze(0)
      
        my_image = my_image.to(device);
        my_image = my_image.float()
        model = model.to(device);
        logps = model.forward(my_image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        return top_p, top_class 

## https://pymotw.com/3/argparse/
## https://docs.python.org/3/library/argparse.html#the-add-argument-method

## use argparse for command line inputs


# Initialize
parser = argparse.ArgumentParser(description="This program predicts flowers' names from their images",
								 usage='''
        needs a saved checkpoint
        python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)
        python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)
        python predict.py --top_k 3 (return top K most likely classes)
        python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)
        python predict.py --gpu (use GPU for inference)''',
								 prog='predict')

## Get path of image
parser.add_argument('path_to_image', action="store", nargs='?', default='flowers/test/1/image_06743.jpg', help="path/to/image")
## Get path of checkpoint
# https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
# Thanks to: Vinay Sajip 
parser.add_argument('path_to_checkpoint', action="store", nargs='?', default='checkpoint.pth', help="path/to/checkpoint")
## set top_k
parser.add_argument('--top_k', action="store", default=1, type=int, help="enter number of guesses", dest="top_k")
## Choose json file:
parser.add_argument('--category_names', action="store", default="cat_to_name.json", help="get json file", dest="category_names")
## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

arg_path_to_image =  args.path_to_image
arg_path_to_checkpoint = args.path_to_checkpoint
arg_top_k =  args.top_k
arg_category_names =  args.category_names
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

# Use GPU if it's selected by user and it is available
device = torch.device("cuda" if arg_gpu else "cpu")
print()
print('Will use {} for prediction...'.format(device))
print()

print()
print("Path of image: {} \nPath of checkpoint: {} \nTopk: {} \nCategory names: {} ".format(arg_path_to_image, arg_path_to_checkpoint, arg_top_k, arg_category_names))
print('GPU: ', arg_gpu)
print()

## Label mapping
print('Mapping from category label to category name...')
print()
with open(arg_category_names, 'r') as f:
    cat_to_name = json.load(f)

## Loading model
print('Loading model........................ ')
print()

my_model, my_optimizer, input_size, output_size, epoch  = load_checkpoint(arg_path_to_checkpoint)

my_model.eval()

# https://knowledge.udacity.com/questions/47967
idx_to_class = {v:k for k, v in my_model.class_to_idx.items()}

# https://github.com/SeanNaren/deepspeech.pytorch/issues/290
# Thanks to @ XuesongYang
# used helper.py
# https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.bar.html

print(arg_path_to_image)
probs, classes = predict('{}'.format(arg_path_to_image), my_model, arg_top_k)

#print('This flower is a/an {}'.format(cat_to_name['{}'.format(test_directory)]))
print()
print('The model predicts this flower as: ')
print()
for count in range(arg_top_k):
     print('{} ...........{:.3f} %'.format(cat_to_name[idx_to_class[classes[0, count].item()]], probs[0, count].item()))
