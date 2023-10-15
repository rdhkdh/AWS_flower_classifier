# this file is for first time setup only
# to download flowers data from udacity
# thanks to Udacity AI Programming with Python Nanodegree Program

## use argparse for command line inputs

import argparse

# Initialize
parser = argparse.ArgumentParser(description='This is a first time setup program that needs to be run only once.',
								 usage='''
        python first_time_setup.py (data set will be downloaded and extracted to the 'flowers' directory)
        python first_time_setup.py --save_dir save (data set will be downloaded and extracted to the 'save' directory)''',
								 prog='train')


## Set directory to save dataset
parser.add_argument('--save_dir', action="store", default="flowers", help="saving directory for the dataset", dest="save_directory")

## Get the arguments
args = parser.parse_args()

arg_save_dir =  args.save_directory

# Create saving directory
import os
if not os.path.exists(arg_save_dir):
	os.mkdir(arg_save_dir)
	print("Directory " , arg_save_dir ,  " has been created for saving the ")
else:
	print("Directory " , arg_save_dir ,  " allready exists for saving checkpoints")
save_dir = arg_save_dir + '/'



# https://stackoverflow.com/questions/19602931/basic-http-file-downloading-and-saving-to-disk-in-python
# thanks to @ https://stackoverflow.com/users/2702249/om-prakash-sao
print('')
print('downloading flowers data please wait...')
import urllib.request 
urllib.request.urlretrieve("https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz", "flower_data.tar.gz")
print('download finished!')

# https://stackoverflow.com/questions/48466421/python-how-to-decompress-a-gzip-file-to-an-uncompressed-file-on-disk
# thanks to: @ https://stackoverflow.com/users/532312/rakesh

print('extracting data...')
import tarfile
tar = tarfile.open("flower_data.tar.gz")
tar.extractall(save_dir)
tar.close()
print('finished!')

# https://www.dummies.com/programming/python/how-to-delete-a-file-in-python/
# thanks to: @ https://www.dummies.com/?s=&a=john-paul-mueller

print('removing tar file...')
import os
os.remove("flower_data.tar.gz")
print("file removed!")