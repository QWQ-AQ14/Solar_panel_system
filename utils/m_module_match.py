import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
from m_get_mosaic_img import pixel2latlon
from osgeo import gdal,osr

def get_point(M,pointAraay):
    pts = np.float32(pointAraay).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(M, pts)  # 映射后的坐标
    return target_point.reshape([-1,2])

def get_mask(img,hsv_range):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_min, h_max, s_min, s_max, v_min, v_max = 55, 135, 45, 255, 70, 255
    lower = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
    upper = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
    mask = cv2.inRange(imgHSV, lower, upper)
    # 滤波去除噪声, 小于5个像素的区域直接移除
    # viewImage(mask)
    #不腐蚀之前 可分割小组件
    kernel = np.ones((3, 3), np.uint8)
    #腐蚀 是缩小
    mask = cv2.erode(mask, kernel, 2)
    # viewImage(mask)
    #膨胀
    mask = cv2.dilate(mask, kernel, iterations=6)
    return mask

def get_region(mask):
    label_im = label(mask)
    regions = regionprops(label_im)
    return regions
# 计算欧式距离
def eucldist_forloop(coords1, coords2):

    """ Calculates the euclidean distance between 2 lists of coordinates. """
    dist = 0
    for (x, y) in zip(coords1, coords2):
        dist += (x - y)**2
    return dist**0.5

def plt_module(img,num,centroid):
    draw_img = img.copy()
    cv2.putText(draw_img, "#{}".format(num), (int(centroid[1]), int(centroid[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)
    plt.imshow(draw_img)
    plt.show()
    return draw_img

def get_target_module(target_pixel,sigle_region,mosaic_region):
    single_num = 0
    single_centroid = None
    # 计算目标像素在单幅图中的组串标号以及质心位置
    for num, x in enumerate(sigle_region):
        if x.area < 100000:
            continue
        bbox = x.bbox
        if target_pixel[0] > bbox[1] and target_pixel[0] < bbox[3] and target_pixel[1] > bbox[0] and target_pixel[1] < bbox[2]:
             single_num = num
             single_centroid = sigle_region[num].centroid
    if single_centroid is None:
        return 0,0
    #根据单幅图中的目标组件质心得出拼接图中的对应质心
    min_dis = 9999
    for num2, x2 in enumerate(mosaic_region):
        dis = eucldist_forloop(single_centroid, x2.centroid)
        if dis < min_dis:
            min_dis = dis
            mosaic_num = num2
    return sigle_region[single_num],mosaic_region[mosaic_num]

