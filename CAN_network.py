#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 22:15:41 2017

@author: soumya
"""

#trying my best to make it work, THE CAN network !!

from __future__ import print_function, division
import torch
import torch.nn as nn
#import torch.nn.init as init
import torch.optim as optim
#from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
#import time
import os, scipy.io
import torch.utils.model_zoo as model_zoo
from scipy import misc
#import helper
#from matlab_imresize_master.imresize import imresize, convertDouble2Byte
from skimage import io
from random import shuffle
import pickle
from datasets import CONFIG

class CAN(nn.Module):

    def __init__(self):
        super(CAN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
                        
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 4096, kernel_size=3, stride=1, padding=0, bias=True, dilation=4), #fc6 layer
            nn.ReLU(inplace=True),
            
#            nn.Dropout(),
            
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0, bias=True), #fc7 layer
            nn.ReLU(inplace=True),
           
#            nn.Dropout(),
            
            nn.Conv2d(4096, 19, kernel_size=1, stride=1, padding=0, bias=True), #final layer
            nn.ReLU(inplace=True),
           
            nn.ZeroPad2d(1),
            
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True), #ctx_conv
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(2),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=2),
            nn.ReLU(inplace=True),
           
            nn.ZeroPad2d(4),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=4),
            nn.ReLU(inplace=True),
            
            nn.ZeroPad2d(8),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=8),
            nn.ReLU(inplace=True),
            
                        
            nn.ZeroPad2d(16),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=16),
            nn.ReLU(inplace=True),
            
                                    
            nn.ZeroPad2d(32),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=32),
            nn.ReLU(inplace=True),
            
                                    
            nn.ZeroPad2d(64),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True, dilation=64),
            nn.ReLU(inplace=True),
            
                                   

            nn.ZeroPad2d(1),
            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(19, 19, kernel_size=1, stride=1, padding=0, bias=True),
            
            nn.Upsample(size=(CONFIG['cityscapes']['output_shape'][0]+1,CONFIG['cityscapes']['output_shape'][1]+1), mode='bilinear'),                           #Change to 1025x2049 if you have enough memory
            
            nn.Conv2d(19, 19, kernel_size=16, stride=1, padding=7, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Softmax(dim=1)
            
            
            
                
    )
        
    def forward(self, x):
        x = self.features(x)
      #  x = x.view(x.size(0), 128 * 6 * 6)
      #  x = self.classifier(x)
        return x
    
#Assigning pretrained weights to the model. 
        
Net=CAN()

with open('/home/soumya/Downloads/dilation_tensorflow/data/pretrained_dilation_cityscapes.pickle', 'rb') as f:
    w=pickle.load(f)

#layers=[0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 64, 66] #add 62,64,66
    
layers=[0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29, 31, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 62, 64] #add 62,64,66


att=['conv1_1/kernel:0', 'conv1_2/kernel:0', 'conv2_1/kernel:0', 'conv2_2/kernel:0', 'conv3_1/kernel:0', 'conv3_2/kernel:0', 'conv3_3/kernel:0', 
     'conv4_1/kernel:0', 'conv4_2/kernel:0', 'conv4_3/kernel:0', 'conv5_1/kernel:0', 'conv5_2/kernel:0', 'conv5_3/kernel:0', 
     'fc6/kernel:0', 'fc7/kernel:0', 'final/kernel:0', 'ctx_conv1_1/kernel:0', 'ctx_conv1_2/kernel:0', 'ctx_conv2_1/kernel:0',
     'ctx_conv3_1/kernel:0', 'ctx_conv4_1/kernel:0', 'ctx_conv5_1/kernel:0', 'ctx_conv6_1/kernel:0', 'ctx_conv7_1/kernel:0', 
     'ctx_fc1/kernel:0', 'ctx_final/kernel:0', 'ctx_upsample/kernel:0']
bia=['conv1_1/bias:0', 'conv1_2/bias:0', 'conv2_1/bias:0', 'conv2_2/bias:0', 'conv3_1/bias:0', 'conv3_2/bias:0', 'conv3_3/bias:0', 
     'conv4_1/bias:0', 'conv4_2/bias:0', 'conv4_3/bias:0', 'conv5_1/bias:0', 'conv5_2/bias:0', 'conv5_3/bias:0', 
     'fc6/bias:0', 'fc7/bias:0', 'final/bias:0', 'ctx_conv1_1/bias:0', 'ctx_conv1_2/bias:0', 'ctx_conv2_1/bias:0',
     'ctx_conv3_1/bias:0', 'ctx_conv4_1/bias:0', 'ctx_conv5_1/bias:0', 'ctx_conv6_1/bias:0', 'ctx_conv7_1/bias:0', 
     'ctx_fc1/bias:0', 'ctx_final/bias:0']
#S=[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
for L in enumerate(layers):
    Net.features[L[1]].weight=nn.Parameter(torch.from_numpy(w[att[L[0]]]).permute(3,2,0,1))
    if L[1] != 64: #66
       Net.features[L[1]].bias=nn.Parameter(torch.from_numpy(w[bia[L[0]]]))
       
# preparing input image to evaluate the result. 
       
#def semanticmap(input_image_path):
    # Choose between 'cityscapes' and 'camvid'



#    with tf.Session() as sess:

        # Choose input shape according to dataset characteristics
#input_h, input_w, input_c = CONFIG[dataset]['input_shape']
#    input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
#    model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

#    sess.run(tf.global_variables_initializer())

        # Save both graph and weights
#    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
#    saver.save(sess, path.join(checkpoint_dir, 'dilation'))

    # Restore both graph and weights from TF checkpoint
#    with tf.Session() as sess:

#    saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
#    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

#    graph = tf.get_default_graph()
#    model = graph.get_tensor_by_name('softmax:0')
#     model = tf.reshape(model, shape=(1,)+CONFIG[dataset]['output_shape'])
#
#        # Read and predict on a test image
#    input_image = cv2.imread(input_image_path)
#    input_tensor = graph.get_tensor_by_name('input_placeholder:0')
#    predicted_image = predict(input_image, input_tensor, model, dataset, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
#        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        
#        cv2.imwrite(output_image_path, predicted_image)
#    return predicted_image


#def predict(image, input_tensor, model, ds, sess):

def semanticmap(image):
    
   dataset = 'cityscapes'
   
   M=CONFIG[dataset]['mean_pixel']
    
   S=image.size()
#    f_M=torch.stack((torch.from_numpy(np.matlib.repmat(M[0],S[2],S[3])),torch.from_numpy(np.matlib.repmat(M[1],S[2],S[3])),torch.from_numpy(np.matlib.repmat(M[2],S[2],S[3]))),2).unsqueeze(0).repeat(S[0],1,1,1)
   f_M=torch.stack((torch.from_numpy(np.matlib.repmat(M[0],S[2],S[3])).float(),torch.from_numpy(np.matlib.repmat(M[1],S[2],S[3])).float(),torch.from_numpy(np.matlib.repmat(M[2],S[2],S[3])).float()),2).unsqueeze(0).repeat(S[0],1,1,1).permute(0,3,1,2)
   image = image - f_M
   conv_margin = CONFIG[dataset]['conv_margin']
    
   input_dims = (S[0],) + CONFIG[dataset]['input_shape']
   batch_size, input_height, input_width, num_channels = input_dims
#   model_in = np.zeros(input_dims, dtype=np.float32)
#
#   image_size = [S[2],S[3],S[1]]
#   output_height = input_height - 2 * conv_margin
#   output_width = input_width - 2 * conv_margin
    #image = cv2.copyMakeBorder(image, conv_margin, conv_margin,
#                               conv_margin, conv_margin,
#                               cv2.BORDER_REFLECT_101)
#   m=torch.nn.ReplicationPad2d(conv_margin)
   m=torch.nn.ReflectionPad2d(conv_margin)
   image=m(image)
   
#   image = cv2.copyMakeBorder(image, margin[0], margin[1],
#                                      margin[2], margin[3],
#                                      cv2.BORDER_REFLECT_101)
   image=Variable(image.data,volatile=True)
   del f_M
   prob=Net(image)
#   n=torch.cat()
   out_pred=torch.max(prob,dim=1,keepdim=True) 
#   col_pred=torch.cat(prob.data,1)
#   col_pred=col_pred.permute(1,2,0)
#   col_pred=col_pred.numpy()
#   prediction = np.argmax(col_pred, axis=2)
#   color_image = CONFIG[dataset]['palette'][prediction.ravel()].reshape(256,512,3)
#   plt.imshow(color_image)
   return out_pred

#a=semanticmap('/home/soumya/Downloads/dilation_tensorflow/data/cityscapes.png')

a=io.imread('data/cityscapes_real.png')
T=torch.from_numpy(a).float()
Z1=T[:,:,0]
Z2=T[:,:,1]
Z3=T[:,:,2]
Z1=Z1.unsqueeze(2)
Z2=Z2.unsqueeze(2)
Z3=Z3.unsqueeze(2)
Z=torch.cat((Z3,Z2,Z1),2).permute(2,0,1).unsqueeze(0)   
#T=(torch.rand(2,3,256,512))
del Z1,Z2,Z3,T,a
output=semanticmap(Z)
m=CONFIG['cityscapes']['output_shape']
output=output[1]
output=output.data.view(1,m[0],m[1]).numpy()
color_image = CONFIG['cityscapes']['palette'][output.ravel()].reshape((m[0],m[1],3))
import scipy.misc
scipy.misc.imsave('data/pytorch_out.png', color_image)
