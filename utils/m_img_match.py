import cv2
from m_get_xmp_info import get_xmp_info
import numpy as np
import math as m
import random
from time import *
import m_QPSO_ssim
# 红外与可见光的传感器尺寸
IR_SENSOR_WIDTH = 10.88
IR_SENSOR_HEIGHT = 8.7
VIS_SENSOR_WIDTH = 7.4
VIS_SENSOR_HEIGHT = 5.55
#桶型畸变参数
K1 = -0.3539
K2 = 0.19257
K3 = -0.00043
K4 = -0.00033
K5 = -0.06868
D_VIS = np.array([K1,K2,K3,K4,K5])
D_IR= np.array([-K1,K2,K3,K4,K5]) # 枕形畸变


#############################几何校正###################################
def computeR(Yaw, Pitch = 0, Roll = 0):

    a = Pitch * np.pi / 180
    b = Yaw * np.pi / 180
    g = Roll * np.pi / 180
    # Compute R matrix according to source.
    Rz = np.array(([m.cos(a), m.sin(a), 0],
                   [-1 * m.sin(a), m.cos(a), 0],
                   [0, 0, 1]))

    Ry = np.array(([m.cos(b), 0, -1 * m.sin(b)],
                   [0, 1, 0],
                   [m.sin(b), 0, m.cos(b)]))

    Rx = np.array(([1, 0, 0],
                   [0, m.cos(g), m.sin(g)],
                   [0, -1 * m.sin(g), m.cos(g)]))
    Ryx = np.dot(Rx, Ry)
    R = np.dot(Rz, Ryx)
    R[0, 0] = 1
    R[2, 2] = 1
    return R

def Geo_Correcition(img,f,DU,Yaw, Pitch = 0, Roll = 0):

    H, W = img.shape[:2]
    u0 = W / 2
    v0 = H / 2

    tmp1 = np.array(([1 / DU, 0, u0],
                     [0, 1 / DU, v0],
                     [0, 0, 1]))

    tmp2 = np.array(([f, 0, 0],
                     [0, f, 0],
                     [0, 0, 1]))

    # 报错信息
    try:
        K1 = np.dot(tmp1, tmp2)  # 3*3
        # K1的逆矩阵
        InvK1 = np.linalg.inv(K1)
    except:
        print("矩阵不存在逆矩阵")

    R = computeR(Pitch, Yaw, Roll)
    tmp3 = np.dot(K1, R)
    tmp4 = np.dot(tmp3, InvK1)

    # 像素重映射
    mapx = np.zeros(img.shape[:2], dtype=np.float32)
    mapy = np.zeros(img.shape[:2], dtype=np.float32)
    print('Geo_Correcition start...')
    for i in range(H):
        for j in range(W):
            arr = np.array([j, i, 1])
            dst = np.dot(tmp4, arr.T)
            mapx[i, j] = np.float32(dst[0])
            mapy[i, j] = np.float32(dst[1])
    print('Geo_Correcition over...')
    newimg = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)  # cv2.remap  像素重映射 双三样条插值
    # cv2.imwrite('res223R.jpg', newimg)
    return newimg
#############################几何校正###################################

#############################桶型校正###################################
def undistort(img,DU,F, num):
    H,W  = img.shape[:2]
    fu=F/DU
    fv=F/DU
    # 像素为单位的主点坐标,即图像物理光心在图像像素坐标系中的坐标
    u0 = W / 2
    v0 = H/ 2
    # 相机内参矩阵
    K = np.array(([fu, 0, u0],
                  [0, fv, v0],
                  [0, 0, 1]))
    if num == 1:#红外
        D = D_IR
    else:#可见
        D = D_VIS
    if img is None:
        return None
    dst = cv2.undistort(img,K,D)

    return dst
#############################桶型校正###################################

#############################图像匹配###################################
def image_matching(irimg_path,visimg_path):
    #读取图像
    ir_img = cv2.imread(irimg_path)
    vis_img = cv2.imread(visimg_path)

    # 获取红外与可见光的XMP信息
    ir_xmp = get_xmp_info(irimg_path)
    vis_xmp = get_xmp_info(visimg_path)

    # 计算图像单位像素的实际尺寸大小
    d_ir = ir_xmp['ImageWidth'] / IR_SENSOR_WIDTH
    d_vis = vis_xmp['ImageWidth'] / VIS_SENSOR_WIDTH

    #几何校正
    ir_img = Geo_Correcition(ir_img,d_ir,ir_xmp['FocalLength'],ir_xmp['Flight-Yaw'])
    vis_img = Geo_Correcition(vis_img,d_vis,vis_xmp['FocalLength'],vis_xmp['Flight-Yaw'])

    #桶型校正
    ir_img = undistort(ir_img, d_ir, ir_xmp['FocalLength'], 1)
    vis_img = undistort(vis_img, d_vis, vis_xmp['FocalLength'], 2)

    #校正后图像灰度化
    ir_img_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    vis_img_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    # -------------------------------------优化QPSO匹配-----------------------------------------
    # 初始值
    scale = ir_xmp['FocalLength'] * d_vis / vis_xmp['FocalLength'] * d_ir
    scale_x, scale_y = scale,scale
    a, b, tx, ty = 0, 0, 0, 0
    pos = [scale_x, a, tx, b, scale_y, ty]
    scale_range = 0.2
    offsetab_range = 0.1
    offset_range = 2
    pmax = [scale_x, a + offsetab_range, tx + offset_range, b + offsetab_range, scale_y, ty + offset_range]
    pmin = [scale_x - scale_range, a - offsetab_range, tx - offset_range, b - offsetab_range, scale_y - scale_range,
            ty - offset_range]
    pos = np.zeros((50, 6), dtype=np.float32)
    N = 50
    for i in range(6):
        for j in range(N):
            a = random.random()
            if (i == 0):
                pos[j, i] = pmin[i] + a * (pmax[i] - pmin[i])
            else:
                # pos[j, i] = random.randint(pmin[i], pmax[i])  # 整型
                pos[j, i] = random.uniform(pmin[i], pmax[i])  # 浮点型
    # 匹配优化
    maxinterations = 100
    yleft, yright, xleft, xright = 500, 2500, 1000, 3000
    print('[5]通过QPSO进行图像尺度参数优化...')
    begin_time = time()
    pa, mi = m_QPSO_ssim.QPSO(ir_img_gray, vis_img_gray, pos, maxinterations, pmax, pmin, yleft, yright, xleft, xright)
    end_time = time()
    run_time = end_time - begin_time
    print('QPSO算法运行时间：', run_time)  # 该循环程序运行时间： 1.4201874732
    fine_MATCH = m_QPSO_ssim.get_reg_img(ir_img_gray, vis_img_gray, pa)
    return pa,fine_MATCH
#############################图像匹配###################################
