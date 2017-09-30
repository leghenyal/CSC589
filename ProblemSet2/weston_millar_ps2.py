#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 14:15:26 2017

@author: westonmillar
"""


import numpy as np,sys
import cv2
import matplotlib.pyplot as plt
import scipy
from scipy import misc, ndimage
import skimage
from skimage import data, io, filters, data_dir
import skimage.io as io
from skimage.io import imread_collection

im1=misc.imread('peppers.png')
im1b=skimage.filters.gaussian(im1,3)
#misc.imsave('blurtest.png', im1_blur)
im1f=np.fft.fft2(im1)
im1fs=np.fft.fftshift(im1f)
im1fm=20*np.log(np.abs(im1fs))
plt.subplot(121),plt.imshow(im1b,cmap='gray')
plt.axis('off')
plt.subplot(122),plt.imshow(im1fm,cmap='gray')
plt.axis('off')
plt.show()

im2=misc.imread('lowcontrast.jpg')
h,bins=np.histogram(im2.flatten(),256,[0,256])
cdf=h.cumsum()
cdf_n=np.ma.masked_equal(cdf,0)
cdf_n2=(cdf_n-cdf_n.min())*255/(cdf_n.max()-cdf_n.min())
cdf2=np.ma.filled(cdf_n2,0).astype('uint8')
im2c=cdf2[im2]
plt.subplot(121),plt.imshow(im2,cmap='gray')
plt.axis('off')
plt.subplot(122),plt.imshow(im2c,cmap='gray')
plt.axis('off')
plt.show()

im3=misc.imread('einstein.png',flatten=1)
g=np.array([1,4,6,4,1])
b=np.array([1,1,1])
s=np.array([-1,0,1])
im3gv=ndimage.convolve1d(im3,g,axis=1)
im3gh=ndimage.convolve1d(im3,g,axis=0)
im3bv=ndimage.convolve1d(im3,b,axis=1)
im3bh=ndimage.convolve1d(im3,b,axis=0)
im3sv=ndimage.convolve1d(im3,s,axis=1)
im3sh=ndimage.convolve1d(im3,s,axis=0)
#misc.imsave('zgaussv.png',im3gv)
#misc.imsave('zgaussh.png',im3gh)
plt.subplot(131),plt.imshow(im3,cmap='gray')
plt.axis('off')
plt.subplot(132),plt.imshow(im3gv,cmap='gray')
plt.axis('off')
plt.subplot(133),plt.imshow(im3gh,cmap='gray')
plt.axis('off')
plt.show()
plt.subplot(131),plt.imshow(im3,cmap='gray')
plt.axis('off')
plt.subplot(132),plt.imshow(im3bv,cmap='gray')
plt.axis('off')
plt.subplot(133),plt.imshow(im3bh,cmap='gray')
plt.axis('off')
plt.show()
plt.subplot(131),plt.imshow(im3,cmap='gray')
plt.axis('off')
plt.subplot(132),plt.imshow(im3sv,cmap='gray')
plt.axis('off')
plt.subplot(133),plt.imshow(im3sh,cmap='gray')
plt.axis('off')
plt.show()

#im4=misc.imread('zebra.png')
#im4f=skimage.img_as_float(im4)
#sx=ndimage.sobel(im4f,0)
#sy=ndimage.sobel(im4f,1)
#im4m_built_in=np.hypot(sx,sy)

im4flat=misc.imread('zebra.png',flatten=1)
im4sv=ndimage.convolve1d(im4flat,s,axis=1)
im4sh=ndimage.convolve1d(im4flat,s,axis=0)
im4m_custom=np.hypot(im4sh,im4sv)

#plt.subplot(121),plt.imshow(im4m_built_in,cmap='gray')
#plt.axis('off')
#plt.subplot(122),plt.imshow(im4m_custom,cmap='gray')
#plt.axis('off')
#plt.show()

plt.imshow(im4m_custom,cmap='gray')
plt.axis('off')