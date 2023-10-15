# first time setup to download flowers data
import argparse

# Initialize
parser = argparse.ArgumentParser(description='This is a first time setup program that needs to be run only once.', usage='''
        python first_time_setup.py (data set will be downloaded and extracted to the 'flowers' directory)
        python first_time_setup.py --save_dir save (data set will be downloaded and extracted to the 'save' directory)''', prog='train')

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


print('')
print('downloading flowers data please wait...')
import urllib.request 
urllib.request.urlretrieve("https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz", "flower_data.tar.gz")
print('download finished!')


print('extracting data...')
import tarfile
tar = tarfile.open("flower_data.tar.gz")
tar.extractall(save_dir)
tar.close()
print('finished!')


print('removing tar file...')
import os
os.remove("flower_data.tar.gz")
print("file removed!")