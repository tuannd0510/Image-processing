from mimetypes import init
import cv2  
from datetime import datetime
import sys
from matplotlib.pyplot import text
from pip import main
import pytesseract
from PIL import Image

"""
argv[1] - id member
argv[2] - check id 
"""

now = datetime.now()
vidcap = cv2.VideoCapture(0)

def capimg():
    if vidcap.isOpened():
        ret, frame = vidcap.read()
        if ret:
            if(sys.argv[2] == "1"):
                cv2.imshow("Frame",frame)            
                img = "plate-"+ str(sys.argv[1])+"-"+ str(now.strftime("%Y%m%d-%H%M%S")) + ".jpg"
                cv2.imwrite(img,frame)
        else:
            print("Error : Failed to capture frame")
    else:
        print("Cannot open camera")
    return img

def img2text(img):

    # text = pytesseract.image_to_string(Image.open(img))
    text = "1"
    print(text)

if __name__ == "__main__":
    a = capimg()
    img2text(a)
