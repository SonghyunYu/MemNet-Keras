"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
import tensorflow as tf
import scipy
import scipy.io
from keras import backend as K
from keras.datasets import mnist
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Lambda, GlobalAveragePooling2D, multiply, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, add, SeparableConv2D, subtract, DepthwiseConv2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys, h5py
import numpy as np
import os 
from glob import glob
import tensorflow as tf
import keras.backend as K
import math, imageio



def RGB_to_Gray(Input):
    R = Input[:,:,0]
    G = Input[:,:,1]
    B = Input[:,:,2]
    output = (R*.2568) + (G*.5041) + (B*.0979) + 16
    
    return output

def shave(input, size):
    
    a = input.shape[0] - size
    b = input.shape[1] - size
    
    output = np.float64(input[size:a,size:b])
    return output

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator 
         
    
def psnr(img1, img2, max_val):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



# Subpixel layer.
def SubpixelConv2D(scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=2
    :return:
    
    
    ----Usage Example----
    # Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)

    scale = 4
    inputs = Input(shape=input_shape)
    
    x = Convolution2D(channels * scale ** 2, (3, 3), 
        activation='relu', 
        name='conv3x3')(inputs)

    out = SubpixelConv2D(input_shape, scale=scale)(x)

    model = Model(inputs=inputs, outputs=out)

"""
# upsample using depth_to_space
def subpixel_shape(input_shape):
    dims = [input_shape[0],
            input_shape[1],
            input_shape[2],
            int(input_shape[3] / (scale ** 2))]
    output_shape = tuple(dims)
    return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel)



def image_test(model_name, Test_DATA_PATH0, Test_DATA_PATH1, Output_PATH):
    
            generator_model = load_model(model_name, custom_objects={"tf": tf})
            test_list0 = glob(Test_DATA_PATH0 + '/*mat')
            test_list1 = glob(Test_DATA_PATH1 + '/*mat')
            test_len = test_list0.__len__()
            test_names0 = os.listdir(Test_DATA_PATH0)
                       
            # 모델 테스트.
            
            ps = 0
            for i in range(test_len):
                    origin = scipy.io.loadmat(test_list0[i])['imhigh'] # .mat 파일 중 이미지배열만 읽기.
                    test = scipy.io.loadmat(test_list1[i])['imlow'] # .mat 파일 중 이미지배열만 읽기
                    test = test / 127.5 - 1
                    
                    
                    x_test = test.reshape(1,test.shape[0], test.shape[1], 3) 
                    pred = generator_model.predict(x_test, batch_size = 1)
                    pred_img = np.zeros((origin.shape[0],origin.shape[1],3), dtype=float)
                    
                    pred_img[:,:,:] = pred[0,:,:,:]
                    
                    # rescale to range 0-1.
                    pred_img = pred_img*0.5 + 0.5
                    pred_img = np.clip(pred_img,0,1)
                    origin = origin / 255.
                    
                    y_origin = RGB_to_Gray(origin)
                    y_pred_img = RGB_to_Gray(pred_img)

                    tem_psnr =  psnr(y_origin, y_pred_img,1.0) 
                    ps += tem_psnr 
                    print(tem_psnr)
                    
                    pred_img = np.uint8(np.round(pred_img*255.)) 
                    
                    if(test_names0[i][-1]=='t'): # mat파일일 경우 bmp로 바꿔줌
                        test_names0[i] = test_names0[i][:-4] + '.png'
                    imageio.imwrite(Output_PATH + '/MemNet_' + test_names0[i] ,pred_img)
            
            ps = ps/test_len
            print(' Avg. PSNR      : ', ps)
            print('saved test images..')





class MyDataLoader():
    def __init__(self, patch_name, label_name):
        self.patch_name = patch_name
        self.label_name = label_name
        with h5py.File(self.patch_name, 'r') as hf:
            self.patch = np.array(hf.get('patch_all'))  
        with h5py.File(self.label_name, 'r') as hf:
            self.label = np.array(hf.get('label_all')) 
        self.patch = np.transpose(self.patch,(3,1,2,0))
        self.label = np.transpose(self.label,(3,1,2,0)) # patch, label 모두 저장.
        self.patch_number = self.patch.shape[0]
    
    def load_data(self, step, batch_size=16):
        batch_x = (self.patch[step:step+batch_size,:,:,:] / 127.5) - 1
        batch_y = (self.label[step:step+batch_size,:,:,:] / 127.5) - 1
        return batch_y, batch_x
        
        
        
    

class MemNet():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.height = 80                 # Low resolution height
        self.width = 80                  # Low resolution width
        self.shape = (self.height, self.width, self.channels)
        
        self.gf = 64

        optimizer = Adam(lr=0.0001, beta_1=0.9)

        # Configure data loader
        self.patch_name = 'E:/ImageDataset/Input_standard_test_images/DIV2K(Train,Valid)_for_SR/DIV2K_train_HR/patches/patch_all_jpg20_color_size'+str(self.height)+'.h5'
        self.label_name = 'E:/ImageDataset/Input_standard_test_images/DIV2K(Train,Valid)_for_SR/DIV2K_train_HR/patches/label_all_jpg20_color_size'+str(self.height)+'.h5'
        self.data_loader = MyDataLoader(patch_name=self.patch_name, label_name=self.label_name)
                                     

     
        # Build the generator

        self.generator = self.build_generator()
        self.generator.compile(loss='mae',
            optimizer=optimizer)

        # images
        img = Input(shape=self.shape)

        self.PATCH_NUMBER = self.data_loader.patch_number
        
        self.Test_DATA_PATH0 =  "E:/ImageDataset/Input_standard_test_images/LIVE1/test/imhigh"
        self.Test_DATA_PATH1 =  "E:/ImageDataset/Input_standard_test_images/LIVE1/test/imlow_q20"
        self.test_list0 = glob(self.Test_DATA_PATH0 + '/*mat')
        self.test_list1 = glob(self.Test_DATA_PATH1 + '/*mat')
        self.test_len = self.test_list0.__len__()
       

    def build_generator(self):
        
      
        ### recursive block ###
        def recursive_block():
            
            input_data = Input(shape=(None, None, self.gf))

            x = BatchNormalization()(input_data)
            x = Activation('relu')(x)
            x = Conv2D(self.gf,(3,3),padding='same', kernel_initializer='glorot_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(self.gf,(3,3),padding='same', kernel_initializer='glorot_normal')(x)
                 
            x = add([x, input_data])

            return Model(input_data,x)
            
        ### whole network
        model_recursive1 = recursive_block()
        model_recursive2 = recursive_block()
        model_recursive3 = recursive_block()
        model_recursive4 = recursive_block()
        model_recursive5 = recursive_block()
        model_recursive6 = recursive_block()


        input_data = Input(shape=(None, None, 3))

        # feature embedding
        x = Conv2D(self.gf,(3,3),padding='same', kernel_initializer='glorot_normal')(input_data)
        concat_x = x

        # recursive structure
        gate_input_list1 = []
        gate_input_list2 = []
        gate_input_list3 = []
        gate_input_list4 = []
        gate_input_list5 = []
        gate_input_list6 = []
        
        for i in range(6):
            x = model_recursive1(x)
            gate_input_list1.append(x)
        gate_input_list1.append(concat_x)
        gate_input = Concatenate()(gate_input_list1)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        output_gate1 = x
        
        for i in range(6):
            x = model_recursive2(x)
            gate_input_list2.append(x)
        gate_input_list2.append(concat_x)
        gate_input_list2.append(output_gate1)
        gate_input = Concatenate()(gate_input_list2)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        output_gate2 = x 
        
        for i in range(6):
            x = model_recursive3(x)
            gate_input_list3.append(x)
        gate_input_list3.append(concat_x)
        gate_input_list3.append(output_gate1)
        gate_input_list3.append(output_gate2)
        gate_input = Concatenate()(gate_input_list3)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        output_gate3 = x 
        
        for i in range(6):
            x = model_recursive4(x)
            gate_input_list4.append(x)
        gate_input_list4.append(concat_x)
        gate_input_list4.append(output_gate1)
        gate_input_list4.append(output_gate2)
        gate_input_list4.append(output_gate3)
        gate_input = Concatenate()(gate_input_list4)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        output_gate4 = x 
        
        for i in range(6):
            x = model_recursive5(x)
            gate_input_list5.append(x)
        gate_input_list5.append(concat_x)
        gate_input_list5.append(output_gate1)
        gate_input_list5.append(output_gate2)
        gate_input_list5.append(output_gate3)
        gate_input_list5.append(output_gate4)
        gate_input = Concatenate()(gate_input_list5)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        output_gate5 = x 
        
        for i in range(6):
            x = model_recursive6(x)
            gate_input_list6.append(x)
        gate_input_list6.append(concat_x)
        gate_input_list6.append(output_gate1)
        gate_input_list6.append(output_gate2)
        gate_input_list6.append(output_gate3)
        gate_input_list6.append(output_gate4)
        gate_input_list6.append(output_gate5)
        gate_input = Concatenate()(gate_input_list6)
        x = Conv2D(self.gf,(1,1),padding='same', kernel_initializer='glorot_normal')(gate_input)
        
        # Recon Net
        x = Conv2D(3,(3,3),padding='same', kernel_initializer='glorot_normal')(x)
        x = add([x, input_data])
        
        model_final = Model(input_data,x)
        
       
        return model_final

   
    
    def train_G(self, end_ep, batch_size=1, sample_interval=50, second_training=False, start_ep=0):
        if second_training: # L1 -> L2
            self.generator.load_weights('./saved_Pre_Generator/MemNet_29.34dB_3ep_7000it_.h5')
            self.generator.compile(loss='mse',optimizer=Adam(lr=0.0001, beta_1=0.9))
        
        start_time = datetime.datetime.now()
        max_psnr=0
        for ep in range(start_ep, end_ep):
            elapsed_time = datetime.datetime.now() - start_time
            print ("%d_ep / time: %s" % (ep, elapsed_time))
            step=0
            for it in range((self.PATCH_NUMBER//batch_size)):
                # ------------------
                #  Pre-Train Generator
                # ------------------
                # Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(step, batch_size)
                step = step + batch_size
                G_loss = self.generator.train_on_batch(imgs_lr, imgs_hr)
                
                
                
                # Plot the progress (time, PSNR), save the current best model.
                if it % sample_interval == 0:
                    max_psnr = self.get_PSNR_save_model(ep, it,sample_interval,max_psnr)
                    
                    

             
          
        
    def get_PSNR_save_model(self, ep, it,sample_interval, max_psnr):
       
        print(it, 'th iteration')
        ps = 0
        for i in range(self.test_len):
                origin = scipy.io.loadmat(self.test_list0[i])['imhigh'] # .mat 파일 중 이미지배열만 읽기.
                test = scipy.io.loadmat(self.test_list1[i])['imlow'] # .mat 파일 중 이미지배열만 읽기
                test = test / 127.5 - 1
                x_test = test.reshape(1,test.shape[0], test.shape[1], 3) 
                pred = self.generator.predict(x_test, batch_size = 1)
                pred_img = np.zeros((origin.shape[0],origin.shape[1],3), dtype=float)
                pred_img[:,:,:] = pred[0,:,:,:]

                # rescale to range 0-1.
                origin = origin / 255.
                pred_img = pred_img*0.5 + 0.5

                y_origin = RGB_to_Gray(origin)
                y_pred_img = RGB_to_Gray(pred_img)

                ps +=  psnr(y_origin, y_pred_img,1.0) 

        ps = ps/self.test_len
        print(' Val. PSNR      : ', ps)
        if ep==0 and it==0:
            max_psnr = ps; temp_ps = "%0.2f" % max_psnr
            self.generator.save('./saved_Pre_Generator/MemNet_' + str(temp_ps) + 'dB_' + str(ep) + 'ep_' + str(it) + 'it_.h5')
        elif max_psnr < ps:
            max_psnr = ps; temp_ps = "%0.2f" % max_psnr
            self.generator.save('./saved_Pre_Generator/MemNet_' + str(temp_ps) + 'dB_' + str(ep) + 'ep_' + str(it) + 'it_.h5')

        return max_psnr
                
   