# MemNet-Keras

Keras implementation of MemNet in the paper "MemNet: A Persistent Memory Network for Image Restoration"

It has the same network structures with MemNet, but there are some differences.
  - RGB input and output are used instead of luminance channel only
  - It uses only final output for loss function (MemNet uses multi-supervised learning)


Training data(.h5) and Test/Valid data(.mat) are made using MATLAB.

-Training data
   It contains 91 training images. 
-Valid data
   Set5 dataset
-Test data
   LIVE1 dataset

-training
   run first cell in main.ipynb.

-test
   run second cell in main.ipynb.


Results
  - PSNR at Q=10 in LIVE1 dataset: paper: 29.45 dB, implementation: 29.55 dB 
    (paper used BSD300 images for training, I used Div2K images (800 2K resolution images) for training).
