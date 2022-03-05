'''
MIT License

Copyright (c) 2022 Drexel Distributed, Intelligent, and Scalable COmputing (DISCO) Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Author: Anup Das
Email : anup.das@drexel.edu
'''

import os
import numpy

import onnx
from onnx2keras import onnx_to_keras

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

def load_trained_model(json_fname,h5_fname):
    from keras.models import model_from_json
    json_file           = open(json_fname, 'r')
    model_json          = json_file.read()
    json_file.close()
    model               = model_from_json(model_json)
    # load weights into new model
    model.load_weights(h5_fname)

    return model

def convert_to_keras(model,x):
    torch.onnx.export(model, x, "torchToOnnx.onnx", verbose=True, input_names = ['input'], output_names = ['output'])   #export to onyx
    onnx_model  = onnx.load('torchToOnnx.onnx')         #load back the model
    k_model     = onnx_to_keras(onnx_model, ['input'])  #convert to keras
    #serialize the model and save in save_dir
    model_dir   = 'lenet'
    #create a directory save_dir if it doesnot exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name  = model_dir+'/bean_lenet'
    json_fname  = model_name + '.json'
    h5_fname    = model_name + '.h5'
    model_json  = k_model.to_json()
    with open(json_fname, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    k_model.save_weights(h5_fname)
