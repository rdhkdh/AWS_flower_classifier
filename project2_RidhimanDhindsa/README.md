# Udacity AI Programming with Python Project 2
**Submitted by**: Ridhiman Kaur Dhindsa, **Date**: 7 October 2023

## Part 1
**Files included**: Image Classifier Project.html, Image Classifier Project.ipynb, 
fc_model.py, helper.py  
**Steps for execution**:  
1) Open the html file to view code and outputs.  
2) The corresponding Jupyter Notebook and supporting files have also been included.  

## Part 2
**Files included**: train.py, predict.py    
**Files required**: cat_to_name.json, checkpoint.pth, 'flowers' directory       
**Steps for execution**:*    
1. Download flower data set in the directory "flowers". If user is saving the dataset in a different directory, it must be specified in the command line arguments.    
2. Enable GPU and run the file train.py in terminal using ```python train.py --gpu```. Alternatively, other options can also be passed as command line arguments, such as the data directory (for flower data set), save directory (for checkpoint files), learning rate, epochs, model architecture (DenseNet-121 or DenseNet-169), gpu (enble/disable) etc. This will create the file 'checkpoint.pth' in the desired directory.  
3. Include the file 'cat_to_name.json' in the root directory.  
4. Enable GPU and run the file predict.py in terminal using ```python predict.py --gpu```. Alternatively, other options can also be passed as command line arguments, such as the path to image, path to checkpoint file, top k value, gpu (enable/disable) etc.       