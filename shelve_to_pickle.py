# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 19:26:38 2022

@author: Eduin Hernandez
Summary: Converts shelve data into pickle data
"""
import os
import shelve
import pickle
import argparse
from utils.parser_utils import str2bool

"Parser Arguments"
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Cifar10 Training')

    'Model Details'
    parser.add_argument('--epoch-num', type=int, default=100, help='End Iterations for Training')
    
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use. Can be either SGD or Adam')
    
    parser.add_argument('--model-name', type=str, default='NASNetMobile', help='Model to Load for Training')
    
    'Save Details'
    parser.add_argument('--filepath', type=str, default='D:/Dewen/Cifar10/', help='Path used for saving the gradients and statistics')

    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------
"Preparing and Checking Parser"
args = parse_args()

if args.optimizer not in ['SGD', 'Adam']:
    assert False, 'Optimizer not found'
    
# Check if main path exists
if not os.path.exists(args.filepath):
    assert False, '\"' + args.filepath + '\" Path does not exist'

filepath = args.filepath + args.model_name + '/'
# Check if subfolders exists, otherwise it creates them
if not os.path.exists(filepath):
    assert False, '\"' + filepath + '\" Path does not exist'
filepath +=  args.optimizer + '/'
if not os.path.exists(filepath):
    assert False, '\"' + filepath + '\" Path does not exist'
    

my_shelf = shelve.open(filepath + 'Accuracy_and_loss' + '.out', flag='r')
data = my_shelf['data']
my_shelf.close()

pickle.dump(data, open(filepath + 'Accuracy_and_loss' + '.p', "wb" ))

for i0 in range(args.epoch_num):
    my_shelf = shelve.open(filepath + 'Gradient_epoch' + str(i0) + '.out', flag='r')
    data = my_shelf['data']
    my_shelf.close()
            
    pickle.dump(data, open(filepath + 'Gradient_epoch' + str(i0) + '.p', "wb" ))
    
print('Conversion Completed')