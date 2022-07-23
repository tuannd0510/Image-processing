import streamlit as st
import cv2
import numpy as np
import PIL
from Canny import *
from LapLace import *

def main():
    st.title("XỬ LÝ ẢNH")
    activities = ["Canny","Sobel","Laplace","Robert & Prewitt"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == 'Canny':
        img_file = st.file_uploader("Up file here",type=['jpg','png'])
        if img_file is not None:
            st.header("Original Image")
            img_show = np.asarray(PIL.Image.open(img_file))
            st.image(img_show)


            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
            img_sobel,smooth_img,edge_gradient,canny_mask = my_canny(img_show,100,200)

            st.header("Bước 1 : Giảm nhiễu")
            col1,col2 = st.columns(2)
            with col1:
                st.image(img_show,caption=" Chuyển Qua Ảnh Xám")
            with col2:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Khử Nhiễu Cho Ảnh")

            st.header("Bước 2 : Tính Gradient và hướng gradient")
            col3,col4 = st.columns(2)
            with col3:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Ảnh đã khử nhiễu")
            with col4:
                st.image((img_sobel/255), clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc sobel")

            st.header("Bước 3 : Non-maximum Suppression")
            col5,col6 = st.columns(2)
            with col5:
                st.image((img_sobel/255), clamp=True, channels='RGB',caption=" Ảnh sau bộ lọc sobel")
            with col6:
                st.image(edge_gradient/255, clamp=True, channels='RGB',caption=" Ảnh qua Non-maximum Suppression")

            st.header("Bước 4 : Lọc ngưỡng")
            col7,col8 = st.columns(2)
            with col7:
                st.image(edge_gradient/255, clamp=True, channels='RGB',caption=" Ảnh sau Non-maximum Suppression")
            with col8:
                st.image(canny_mask/255, clamp=True, channels='RGB',caption = " Ảnh sau khi lọc ngưỡng")

            st.header("SO SÁNH VỚI SỬ DỤNG HÀM OPENCV")
            col9,col10 = st.columns(2)
            with col9:
                st.image(canny_mask/255, clamp=True, channels='RGB',caption=" Code bộ lọc Canny by me")
            with col10:
                canny_opencv = cv2.Canny(img_show, 100, 200)
                st.image(canny_opencv, clamp=True, channels='RGB',caption=" Code bộ lọc Canny use OpenCV")

            st.header("Canny với hai ngưỡng tự động")
            img_auto = canny_auto(img_show)
            st.image(img_auto, clamp=True, channels='RGB',caption="Bộ lọc Canny lấy ngưỡng tự động")

    if choice == 'Sobel':
        img_file = st.file_uploader("Up file here",type=['jpg','png'])
        if img_file is not None:
            st.header("Original Image")
            img_show = np.asarray(PIL.Image.open(img_file))
            st.image(img_show)

            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
            st.header("Bước 1 : Giảm nhiễu")
            kernel_gauss = createGaussianKernel(5,1)
            smooth_img = convolute(img_show,kernel_gauss)

            col1,col2 = st.columns(2)
            with col1:
                st.image(img_show,caption=" Chuyển Qua Ảnh Xám")
            with col2:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Khử Nhiễu Cho Ảnh")
            
            st.header("Bước 2 : Chuyển đổi hình ảnh màu xám với bộ lọc Sobel x")
            col3,col4 = st.columns(2)
            Gx = SobelFilter(smooth_img,dx=1,dy=0)
            with col3:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Ảnh đã khử nhiễu")
            with col4:
                st.image((Gx/255), clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc sobel x")

            st.header("Bước 3 : Chuyển đổi hình ảnh màu xám với bộ lọc Sobel y")
            col5,col6 = st.columns(2)
            Gy = SobelFilter(smooth_img,dx=0,dy=1)
            with col5:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Ảnh đã khử nhiễu ")
            with col6:
                st.image(Gy/255, clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc sobel y")
            
            st.header("Bước 4 : Tính toán độ lớn của Gradient")
            col7,col8 = st.columns(2)
            Gxy = SobelFilter(smooth_img,dx=1,dy=1)
            with col7:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Ảnh đã khử nhiễu ")
            with col8:
                st.image(Gxy/255, clamp=True, channels='RGB',caption = " Ảnh qua bộ lọc Sobel xy")

            st.header("SO SÁNH VỚI SỬ DỤNG HÀM OPENCV")
            col9,col10 = st.columns(2)
            with col9:
                st.image(Gx/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobelx by me")
            with col10:
                Sobelx_opencv = cv2.Sobel(img_show,cv2.CV_64F,1,0)
                st.image(Sobelx_opencv/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobelx use OpenCV")
            col11,col12 = st.columns(2)
            with col11:
                st.image(Gy/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobely by me")
            with col12:
                Sobely_opencv = cv2.Sobel(img_show,cv2.CV_64F,0,1)
                st.image(Sobely_opencv/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobely use OpenCV")
            col13,col14 = st.columns(2)
            with col13:
                st.image(Gxy/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobelxy by me")
            with col14:
                Sobelxy_opencv = cv2.Sobel(img_show,cv2.CV_64F,1,1)
                st.image(Sobelxy_opencv/255, clamp=True, channels='RGB',caption=" Code bộ lọc Sobelxy use OpenCV")

    if choice == 'Robert & Prewitt':
        img_file = st.file_uploader("Up file here",type=['jpg','png'])
        if img_file is not None:
            st.header("Original Image")
            img_show = np.asarray(PIL.Image.open(img_file))
            st.image(img_show)

            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
            st.header("Bước 1 : Giảm nhiễu")
            kernel_gauss = createGaussianKernel(5,1)
            smooth_img = convolute(img_show,kernel_gauss)

            col1,col2 = st.columns(2)
            with col1:
                st.image(img_show,caption=" Chuyển Qua Ảnh Xám")
            with col2:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Khử Nhiễu Cho Ảnh")
            
            st.header("Bước 2 : Chuyển đổi hình ảnh màu xám với bộ lọc x")
            col3,col4 = st.columns(2)
            GxP = PrewittFilter(smooth_img,dx=1,dy=0)
            GxR = RobertsFilter(smooth_img,dx=1,dy=0)
            with col3:
                st.image((GxP/255), clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc Prewitt x")
            with col4:
                st.image((GxR/255), clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc Roberts x")

            st.header("Bước 3 : Chuyển đổi hình ảnh màu xám với bộ lọc y")
            col5,col6 = st.columns(2)
            GyP = PrewittFilter(smooth_img,dx=0,dy=1)
            GyR = RobertsFilter(smooth_img,dx=0,dy=1)
            with col5:
                st.image(GyP/255, clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc Prewitt y")
            with col6:
                st.image(GyR/255, clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc Roberts y")
            
            st.header("Bước 4 : Tính toán độ lớn của Gradient")
            col7,col8 = st.columns(2)
            GxyP = PrewittFilter(smooth_img,dx=1,dy=1)
            GxyR = RobertsFilter(smooth_img,dx=1,dy=1)
            with col7:
                st.image(GxyP/255, clamp=True, channels='RGB',caption = " Ảnh qua bộ lọc Prewitt xy")
            with col8:
                st.image(GxyR/255, clamp=True, channels='RGB',caption = " Ảnh qua bộ lọc Roberts xy")

        pass
    

    if choice == 'Laplace':
        img_file = st.file_uploader("Up file here",type=['jpg','png'])
        if img_file is not None:
            st.header("Original Image")
            img_show = np.asarray(PIL.Image.open(img_file))
            st.image(img_show)
        

            img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2GRAY)
            st.header("Bước 1 : Giảm nhiễu")
            kernel_gauss = createGaussianKernel(5,1)
            smooth_img = convolute(img_show,kernel_gauss)

            col1,col2 = st.columns(2)
            with col1:
                st.image(img_show,caption=" Chuyển Qua Ảnh Xám")
            with col2:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Khử Nhiễu Cho Ảnh")

            st.header("Bước 2 : Chuyển đổi hình ảnh màu xám với bộ lọc Laplace")
            col3,col4 = st.columns(2)
            img_Lap = LaplaceFilter(smooth_img,1)
            with col3:
                st.image(smooth_img/255, clamp=True, channels='RGB',caption=" Ảnh đã khử nhiễu")
            with col4:
                st.image((img_Lap/255), clamp=True, channels='RGB',caption=" Ảnh qua bộ lọc Laplace")
            
            st.header("SO SÁNH VỚI SỬ DỤNG HÀM OPENCV")
            col5,col6 = st.columns(2)
            img_LapOpencv =cv2.Laplacian(img_show,cv2.CV_64F)
            with col5:
                st.image((img_Lap/255), clamp=True, channels='RGB',caption="Code bộ lọc Laplace by me ")
            with col6:
                st.image(img_LapOpencv/255, clamp=True, channels='RGB',caption=" Code bộ lọc Laplace use OpenCV ")


            st.header("CÁC BIẾN THỂ KHÁC CỦA LAPLACE")
            col7,col8,col9 = st.columns(3)
            Laplacian1 = LaplaceFilter(smooth_img,1)
            Laplacian2 = LaplaceFilter(smooth_img,2)
            Laplacian3 = LaplaceFilter(smooth_img,3)
            with col7:
                st.image((Laplacian1/255), clamp=True, channels='RGB',caption="Laplace chuẩn")
            with col8:
                st.image(Laplacian2/255, clamp=True, channels='RGB',caption="Biến thể 1")
            with col9:
                st.image(Laplacian3/255, clamp=True, channels='RGB',caption="Biến thể 2")

if __name__== '__main__':
    main()