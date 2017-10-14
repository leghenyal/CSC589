#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:21:41 2017

@author: westonmillar
"""

import numpy as np#,sys
#import cv2
import matplotlib.pyplot as plt
#import scipy
from scipy import misc#, ndimage
#import skimage
#from skimage import data, io, filters, data_dir
#import skimage.io as io
#from skimage.io import imread_collection

def cross_correlation_2d(im,k):
    kx,ky=k.shape
    k2=kx*ky
    kx2,ky2=kx/2,ky/2
    karray=np.ravel(k)
    blank=np.zeros(im.shape)
    if im.ndim==3:
        ix,iy,colors=im.shape
    else:
        colors=1
        ix,iy=im.shape
    zeropad=np.zeros((ix+kx-1,iy+ky-1,colors))
    zeropad[kx2:kx2+ix,ky2:ky2+iy]=im
    for x in xrange(iy):
        for y in xrange(ix):
            channels=np.reshape(zeropad[y:y+kx,x:x+ky],(k2,colors))
            blank[y,x]=np.dot(karray,channels)
    #plt.imshow(blank)
    return blank

def convolve_2d(im,k):
    c2d=cross_correlation_2d(im,k)
    #plt.imshow(c2d)
    return c2d

def gaussian_blur_kernel_2d(w,h):
    w=int(w)
    h=int(h)
    x,y=np.mgrid[-w:w+1,-h:h+1]
    k=np.exp(-(x**2/float(w)+y**2/float(h)))/np.exp(-(x**2/float(w)+y**2/float(h))).sum()
    #plt.imshow(kernel)
    return k
    
def low_pass(im,slo):
    lp=convolve_2d(im,gaussian_blur_kernel_2d(slo,slo))
    #plt.imshow(lp)
    #misc.imsave('lopass.png',lp)
    return lp

def high_pass(im,shi):
    hp=im-low_pass(im,shi)
    #plt.imshow(hp)
    #misc.imsave('hipass.png',hp)
    return hp


#while True:
#    choice=raw_input('Generate a Hybrid Image? [y/n] ')
#
#    if choice=='y':
#        iminput1=raw_input('Image to blur (with extension):')
#        iminput2=raw_input('Image to sharpen (with extension):')
#        lo=int(input('Low-pass filter size:'))
#        hi=int(input('High-pass filter size:'))
#        balance=float(input('Image balance (between 0 and 1):'))
#        
#        im1=misc.imread(iminput1).astype(np.float32)/255.0
#        im2=misc.imread(iminput2).astype(np.float32)/255.0
#        lp=low_pass(im1,lo)*2*(1-balance)
#        hp=high_pass(im2,hi)*2*balance
#        hybrid=((lp+hp)*255).clip(0,255).astype(np.uint8)
#        plt.imshow(hybrid)
#
#    elif choice=='n':
#        print 'Thank you for using the generator!'
#        break
#
#    else:
#        print "Invalid input."
        
iminput1=raw_input('Image to blur (with extension):')
iminput2=raw_input('Image to sharpen (with extension):')
lo=int(input('Low-pass filter size:'))#13
hi=int(input('High-pass filter size:'))#5
balance=float(input('Image balance (between 0 and 1):'))#0.7

im1=misc.imread(iminput1).astype(np.float32)/255.0
im2=misc.imread(iminput2).astype(np.float32)/255.0
lp=low_pass(im1,lo)*2*(1-balance)
hp=high_pass(im2,hi)*2*balance
hybrid=((lp+hp)*255).clip(0,255).astype(np.uint8)
plt.imshow(hybrid)
#misc.imsave('subfish.png',hybrid)