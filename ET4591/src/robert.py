from cgitb import small
import cv2
import numpy as np   # Thư viện numy để làm việc dữ liệu kiểu mảng
import matplotlib.pyplot as plt # import thư viện matplotlib để vẽ ảnh
from scipy import ndimage

# img = cv2.imread("cameraman.jpg",0).astype('float64')s
def funclib(img):
    roberts_cross_v = np.array( [[1, 0 ],
                                [0,-1 ]] )
    
    roberts_cross_h = np.array( [[ 0, 1 ],
                                [ -1, 0 ]] )
    
    img/=255.0
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )
    
    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    edged_img*=255
    cv2.imshow("output",edged_img)



# Định nghĩa hàm Tich_chap() để lọc Trung bình, Trung bình có trọng số và lọc Gaussian
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

# Định nghĩa Robert Cross theo hướng chéo 1
kernel_x = np.array((   [0, 0, 0],
                        [0, 1, 0],
                        [0, 0,-1]),dtype="float")

# Định nghĩa Robert Cross theo hướng chéo 2
kernel_y = np.array((   [0, 0, 0],
                        [0, 0, 1],
                        [0,-1, 0]),dtype="float")

image = cv2.imread('cameraman.jpg', 0)

img_Robert_Cross1 = Convolution(image, kernel_x) #Gọi hàm tích chập
img_Robert_Cross2 = Convolution(image, kernel_y) #Gọi hàm tích chập
img_Robert_Cross = img_Robert_Cross1 + img_Robert_Cross2

cv2.imshow("Ảnh gốc",image)
cv2.imshow("img_Robert_Cross1",img_Robert_Cross1/255)
cv2.imwrite('ro1.jpg',img_Robert_Cross1)
cv2.imshow("img_Robert_Cross2",img_Robert_Cross2/255)
cv2.imwrite('ro2.jpg',img_Robert_Cross2)
cv2.imshow("img_Robert_Cross",img_Robert_Cross/255)
cv2.imwrite('ro12.jpg',img_Robert_Cross)

cv2.waitKey(0)