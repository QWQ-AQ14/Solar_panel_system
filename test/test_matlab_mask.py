# 利用matlab获取HSV的阈值 并且得到mask图像
from tkinter import filedialog
import cv2
import numpy as np
from skimage.color import rgb2hsv,rgb2gray

 ## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def get_matlab_mask(img):
    # 创建mask图像
    dst = np.ones_like(img)
    # matlab原图像通道顺序不一致
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_hsv = rgb2hsv(img)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    # % Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.598
    channel1Max = 0.714
    # % Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.113
    channel2Max = 0.517
    # % Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.341
    channel3Max = 0.988

    mask_h = np.logical_and(h>=channel1Min, h <= channel1Max)
    mask_s= np.logical_and(s>=channel2Min, s <= channel2Max)
    mask_v = np.logical_and(v >= channel3Min, v <= channel3Max)
    mask = np.logical_and(mask_h,mask_s, mask_v)
    dst[mask == True] = (255, 255, 255)

    return dst

def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ftypes = [
        ("JPG", "*.jpg;*.JPG;*.JPEG"),
        ("PNG", "*.png;*.PNG"),
        ("GIF", "*.gif;*.GIF"),
        ("All files", "*.*")
    ]
    file_path = filedialog.askopenfilename(filetypes=ftypes)
    image_src = cv_imread(file_path)
    mask = get_matlab_mask(image_src)
    viewImage(mask)



