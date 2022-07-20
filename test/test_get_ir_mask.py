
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import rgb2hsv

from utils.a_commonUtils import *
from utils.m_ir_vis_reg import undistort
from utils.m_module_match import get_region
# 通过掩码图像获取轮廓
def get_mask_contours(mask):
    # mask = cv2.imread('../resultimgs/mask_0719.jpg')
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, 3)  #
    # viewImage(mask)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # viewImage(mask)
    # Now you can finally find contours.
    contours,hierarchy = cv2.findContours(mask[:,:,1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = []
    for i,contour in enumerate(contours):
        ares = cv2.contourArea(contour)  # 计算包围形状的面积
        if ares < 100000:  # 过滤面积小于50000的形状
            continue
        # print(ares)
        rect = cv2.minAreaRect(contour)  # 检测轮廓最小外接矩形，得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        # if rect[1][0] < rect[1][1]:
        #     continue
        final_contours.append(contour)

    return final_contours

def get_ir_mask(ir_img,vis_img):
    # 获取配准参数
    affmat = np.array([[3.6397, -0.0059, 810.2632],
                       [0.0017, 3.6523, 550.8806],
                       [0, 0, 1]])
    pos = [3.63,0,830,0,3.6341658,550]
    # 对可见光进行分割
    #   ----------------阈值分割
    visimg_mask  = cv2.imread('../resultimgs/mask_0719.jpg')
    vis_cnt = get_mask_contours(visimg_mask)
    trans_ir, VisImg, match_img = IR_img_changebig(ir_img,vis_img,pos)
    viewImage(trans_ir)
    viewImage(match_img)
    drawContours(trans_ir,vis_cnt)



if __name__ == '__main__':
    irimg_path = r"E:\xlq\学术文件\向罗巧项目材料\屋顶分布式项目素材\滨海浩光\DJI_0718_R.JPG"
    visimg_path = r"E:\xlq\学术文件\向罗巧项目材料\屋顶分布式项目素材\滨海浩光\DJI_0719.jpg"
    ir_img = cv_imread(irimg_path)
    vis_img = cv_imread(visimg_path)
    undistorted_visimg = undistort(vis_img,0.0019,0.0019,8,2)
    undistorted_irimg = undistort(ir_img, 0.017, 0.017, 19, 1)
    # cv2.imwrite('../resultimgs/undistorted_visimg.jpg',undistorted_visimg)
    get_ir_mask(undistorted_irimg, undistorted_visimg)
    cv2_imwrite('undistorted_visimg.jpg',undistorted_visimg)
    cv2_imwrite('undistorted_irimg.jpg', undistorted_irimg)
    # viewImage(undistorted_visimg)
    # viewImage(undistorted_irimg)






