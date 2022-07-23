from concurrent.futures import thread
import cv2
import numpy as np
from torch import sigmoid
from matplotlib import pyplot as plt
import math

def createGaussianKernel(kernel_size,sigma=1):
    h=kernel_size//2     #h = int(kernel_size/2)
    s = 2.0 * sigma * sigma
    # sum = 0
    kernel=np.zeros((kernel_size,kernel_size),np.float64)
    for i in range(kernel_size):
        for j in range(kernel_size):
            r = np.sqrt(np.square(i-h) +np.square(j-h))
            kernel[i,j] = (np.exp(-(r * r) / s)) / (math.pi * s)
    kernel = kernel/kernel.sum()
    return kernel

def convolute(image, kernel):
    #lật ma trận kernel
    r,c = kernel.shape
    flip_kernel = np.zeros((r,c), np.float64)
    h = 0
    if ((r==c) and (r%2)):
        h = int((r-1)/2)   #h=1
        for i in range(-h, h+1):    #i -1 -> 1
            for j in range(-h, h+1):   #j -1 -> 1
                flip_kernel[i+h, j+h] = kernel[-i+h, -j+h]
    m, n = image.shape
    padding_image = np.zeros((m+2*h, n+2*h), np.float64)
    padding_image[h:-h, h:-h] = image

    convolute_image = np.zeros((m, n), np.float64)
    for i in range(m):
        for j in range(n):
            convolute_image[i, j] = (flip_kernel * padding_image[i: i+r, j: j+r]).sum()
    return convolute_image

def SobelFilter(image,dx,dy):
    X = np.array([[1,0,-1],[2,0,-2],[1,0,-1]],np.float64)
    Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float64)
    sobel_x = convolute(image,X)
    sobel_y = convolute(image,Y)
    if dx==1 and dy==0:
        return sobel_x
    if dx==0 and dy==1:
        return sobel_y
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)
    alpha = 0.5
    beta = 0.5
    gamma = 0
    sobel_xy = abs_grad_x*alpha + abs_grad_y*beta + gamma

    return sobel_xy

def RobertsFilter(image,dx,dy):
    X = np.array([[0, 0, 0],[0, 1, 0],[0, 0,-1]],np.float64)
    Y = np.array([[0, 0, 0],[0, 0, 1],[0,-1, 0]],np.float64)
    Roberts_x = convolute(image,X)
    Roberts_y = convolute(image,Y)
    if dx==1 and dy==0:
        return Roberts_x
    if dx==0 and dy==1:
        return Roberts_y
    abs_grad_x = cv2.convertScaleAbs(Roberts_x)
    abs_grad_y = cv2.convertScaleAbs(Roberts_y)
    alpha = 0.5
    beta = 0.5
    gamma = 0
    Roberts_xy = abs_grad_x*alpha + abs_grad_y*beta + gamma

    return Roberts_xy

def PrewittFilter(image,dx,dy):
    X = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],np.float64)
    Y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],np.float64)
    Prewitt_x = convolute(image,X)
    Prewitt_y = convolute(image,Y)
    if dx==1 and dy==0:
        return Prewitt_x
    if dx==0 and dy==1:
        return Prewitt_y
    abs_grad_x = cv2.convertScaleAbs(Prewitt_x)
    abs_grad_y = cv2.convertScaleAbs(Prewitt_y)
    alpha = 0.5
    beta = 0.5
    gamma = 0
    Prewitt_xy = abs_grad_x*alpha + abs_grad_y*beta + gamma

    return Prewitt_xy

def my_canny(img, min_val, max_val):
    
    #2. Noise Reduction
    kernel_gauss = createGaussianKernel(5,1)
    smooth_img = convolute(img,kernel_gauss)
    
    #3. Finding Intensity Gradient of the Image
    Gx = SobelFilter(img,dx=1,dy=0)
    Gy = SobelFilter(img,dx=0,dy=1)

    img_sobel  = SobelFilter(img,dx=1,dy=1)
    edge_gradient = np.abs(Gx) + np.abs(Gy)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    
    # round angle to 4 directions   
    angle = np.abs(angle)
    angle[angle <= 22.5] = 0
    angle[angle >= 157.5] = 0
    angle[(angle > 22.5) * (angle < 67.5)] = 45
    angle[(angle >= 67.5) * (angle <= 112.5)] = 90
    angle[(angle > 112.5) * (angle <= 157.5)] = 135
    
    #4. Non-maximum Suppression
    keep_mask = np.zeros(smooth_img.shape, np.uint8)
    for y in range(1, edge_gradient.shape[0]-1):
        for x in range(1, edge_gradient.shape[1]-1):
            area_grad_intensity = edge_gradient[y-1:y+2, x-1:x+2] # 3x3 area
            area_angle = angle[y-1:y+2, x-1:x+2] # 3x3 area
            current_angle = area_angle[1,1]
            current_grad_intensity = area_grad_intensity[1,1]
            
            if current_angle == 0:
                if current_grad_intensity > max(area_grad_intensity[1,0], area_grad_intensity[1,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 45:
                if current_grad_intensity > max(area_grad_intensity[2,0], area_grad_intensity[0,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 90:
                if current_grad_intensity > max(area_grad_intensity[0,1], area_grad_intensity[2,1]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
            elif current_angle == 135:
                if current_grad_intensity > max(area_grad_intensity[0,0], area_grad_intensity[2,2]):
                    keep_mask[y,x] = 255
                else:
                    edge_gradient[y,x] = 0
    
    #5. Hysteresis Thresholding    
    canny_mask = np.zeros(smooth_img.shape, np.uint8)
    canny_mask[(keep_mask>0) * (edge_gradient>min_val)] = 255
    
    return img_sobel,smooth_img,edge_gradient,canny_mask

def canny_auto(img):
    sigma = 0.3
    meadian = np.median(img)
    lower = int(max(0,(1.0 - sigma)*meadian))
    upper = int(min(255,(1.0 + sigma)*meadian))
    auto_candy = cv2.Canny(img, lower,upper)
    return auto_candy


