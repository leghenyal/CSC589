#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:17:20 2017

@author: westonmillar
"""

import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage

im1=misc.imread('apple.jpg', flatten=1)
im2=misc.imread('orange.jpg', flatten=1)
mask=misc.imread('mask.jpg', flatten=1).astype(float)/255

#Code for kernel, interpolate, decimate, pyramids, and construct fns
#sourced from GeneratePyramid.py

#Binomial 5-tap filter kernel
kernel = (1.0/256)*np.array([[1, 4,  6,  4,  1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4,  6,  4,  1]])

def interpolate(image):
    """
    Interpolates an image with upsampling rate r=2.
    """
    image_up = np.zeros((2*image.shape[0], 2*image.shape[1]))
    # Upsample
    image_up[::2, ::2] = image
    # Blur (we need to scale this up since the kernel has unit area)
    # (The length and width are both doubled, so the area is quadrupled)
    #return sig.convolve2d(image_up, 4*kernel, 'same')
    return ndimage.filters.convolve(image_up,4*kernel, mode='constant')

def decimate(image):
    """
    Decimates at image with downsampling rate r=2.
    """
    # Blur
    #image_blur = sig.convolve2d(image, kernel, 'same')
    image_blur = ndimage.filters.convolve(image,kernel, mode='constant')
    # Downsample
    return image_blur[::2, ::2]     

# here is the constructions of pyramids 
def pyramids(image):
    """
    Constructs Gaussian and Laplacian pyramids.
    Parameters :
        image  : the original image (i.e. base of the pyramid)
    Returns :
        G   : the Gaussian pyramid
        L   : the Laplacian pyramid
    """
    # Initialize pyramids
    G = [image, ]
    L = []
    # Build the Gaussian pyramid to maximum depth
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        G.append(image)   
    # Build the Laplacian pyramid
    for i in range(len(G) - 1):
        L.append(G[i] - interpolate(G[i + 1]))
    return G[:-1], L

def construct_pyramid(img, G):
    """ 
    for display purposes 
    parameters: 
        image: the original image 
        pyramid: the result from pyramids function 
    returns: 
        composite_image: the image containing the breakdown of the pyramid
    """ 
    rows, cols = img.shape
    composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)
    composite_image[:rows, :cols] = G[0]
    i_row = 0
    for p in G[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    return composite_image

def blend(im1l,im2l,mask):
    blended=[] #empty array for final image
    for i in range(0,len(im1l)): 
        left=(mask[i]*im1l[i]) #multiply mask with first image for left half
        right=((1-mask[i])*(im2l[i])) #inverse mask with second for right half
        blended.append(left+right) #add them together
    return blended

def reconstruct(L):  
    for i in range(len(L)-1,0,-1):
        lupscale=interpolate(L[i]) #start with the top 
        lupscale2 = L[i-1] #take next level
        lsum=lupscale+lupscale2 #sum laplacians 
        del L[-1]
        del L[-1] #delete the 2 used pyramid levels
        L.append(lsum) #append pyramid with collapsed segmen
    return lsum

im1=misc.imread('apple.jpg',flatten=1)
im2=misc.imread('orange.jpg',flatten=1)
mask=misc.imread('mask.jpg',flatten=1).astype(float)/255

im1g,im1l=pyramids(im1)
im2g,im2l=pyramids(im2)
maskg,non=pyramids(mask)

im1gcomp=construct_pyramid(im1,im1g)
im2gcomp=construct_pyramid(im2,im2g) 
im1lcomp=construct_pyramid(im1,im1l)
im2lcomp=construct_pyramid(im2,im2l) 

reconstructed=reconstruct(blend(im2l,im1l,maskg))

plt.subplot(121),plt.imshow(im1gcomp)
plt.axis('off')
plt.subplot(122),plt.imshow(im1lcomp)
plt.axis('off')
plt.show()
plt.subplot(121),plt.imshow(im2gcomp)
plt.axis('off')
plt.subplot(122),plt.imshow(im2lcomp)
plt.axis('off')
plt.show()
plt.imshow(reconstructed)
plt.axis('off')
plt.show()

#misc.imsave('appleorange.png',reconstructed)