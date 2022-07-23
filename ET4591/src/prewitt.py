import cv2
from cv2 import imshow
from matplotlib.image import imsave
import numpy as np


def funclib(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    prewitt_x = np.array([  [1,1,1],
                            [0,0,0],
                            [-1,-1,-1]])
    prewitt_y = np.array([  [-1,0,1],
                            [-1,0,1],
                            [-1,0,1]])
    lib_prewittx = cv2.filter2D(img_gaussian, -1, prewitt_x)
    lib_prewitty = cv2.filter2D(img_gaussian, -1, prewitt_y)
    lib_prewitt = lib_prewittx + lib_prewitty

    return lib_prewittx, lib_prewitty, lib_prewitt

def Convolution(img,mask):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp   =  img[i-1, j-1]    * mask[0, 0]\
                   +  img[i, j-1]      * mask[0, 1]\
                   +  img[i+1, j - 1]  * mask[0, 2]\
                   +  img[i-1, j]      * mask[1, 0]\
                   +  img[i, j]        * mask[1, 1]\
                   +  img[i+1, j]      * mask[1, 2]\
                   +  img[i - 1, j+1]  * mask[2, 0]\
                   +  img[i, j + 1]    * mask[2, 1]\
                   +  img[i + 1, j + 1]* mask[2, 2]
            img_new[i, j]= temp
    img_new = img_new.astype(np.float64)
    return img_new

prewitt_x = np.array(([[-1,0,1],[-1,0,1],[-1,0,1]]),dtype="float") #kiểu dữ liệu
prewitt_y = np.array(([[-1,-1,-1],[0,0,0],[1,1,1]]),dtype="float")

img = cv2.imread('cameraman.jpg',0)

x = Convolution(img,prewitt_x)
y = Convolution(img,prewitt_y)
z = x+y

cv2.imshow("Ảnh gốc",img)
cv2.imshow("x",x/255)
cv2.imwrite('prex.jpg',x)
cv2.imshow("y",y/255)
cv2.imwrite('prey.jpg',y)
cv2.imshow("xy",z/255)
cv2.imwrite("prexy.jpg",z)
cv2.waitKey()