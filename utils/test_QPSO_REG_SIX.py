import cv2
import matplotlib.image as mpimg
import numpy as np
import math as m
import random
from time import *

from utils.m_get_xmp_info import get_xmp_info
import utils.m_QPSO_ssim as m_QPSO_ssim
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
def get_reg_img(IrImg,VisImg,pa):
    # 获取变化后可见光图像的中心点坐标
    h1, w1 = VisImg.shape[:2]
    # 获取红外图像尺寸
    h2, w2 = IrImg.shape[:2]
    pa[2] = (w1 / pa[0] / 2 - w2 / 2 + pa[1]) * pa[0]
    pa[5] = (h1 / pa[4] / 2 - h2 / 2 + pa[5]) * pa[4]
    H = np.array([[pa[0], pa[1], pa[2]], [pa[3], pa[4], pa[5]], [0, 0, 1]], np.float32)
    image_output = cv2.warpPerspective(IrImg, H, (VisImg.shape[1], VisImg.shape[0]))
    match_img = cv2.addWeighted(image_output, 0.5, VisImg, 0.5, 0)
    # read.showimg(image_output)
    # read.showimg(match_img)
    return image_output, VisImg, match_img
def MI(x,y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    size = x.shape[-1]
    px = np.histogram(x, 256, (0, 255))[0] / size
    py = np.histogram(y, 256, (0, 255))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    nmi = (hx + hy)/hxy
    return nmi
def pv(p,im1,im2,yleft,yright,xleft,xright):
    # scale = p[0]
    # x = p[1]
    # y = p[2]
    # img1,img2 = m_ScaleImgReg.R_VIS_Match(im1,im2,scale,x,y)
    img1, img2,match_img = get_reg_img(im1,im2,p)
    #使用MI相似性度量
    result_mi = MI(img1[yleft:yright,xleft:xright],img2[yleft:yright,xleft:xright])
    #获取边缘图
    # edge_im1 = get_edge_img(img1[yleft:yright, xleft:xright])
    # edge_im2 = get_edge_img(img2[yleft:yright, xleft:xright])
    # 使用SSIM结构相似性评价
    # result_ssim = skimage.measure.compare_ssim(img1[yleft:yright,xleft:xright],img2[yleft:yright,xleft:xright])
    #专利
    # result_ssim = skimage.measure.compare_ssim(img2[yleft:yright, xleft:xright],match_img[yleft:yright,xleft:xright])
    # # result = tf.image.ssim(img1[yleft:yright, xleft:xright], img2[yleft:yright, xleft:xright],255)

    # --------------------vif-------------------
    #result_vif = skimage.measure.simple_metrics.compare_vifp(img1[yleft:yright,xleft:xright],img2[yleft:yright,xleft:xright])

    result = result_mi

    return result
def QPSO(R,F,pos,maxinterations,pmax,pmin,yleft,yright,xleft,xright):
    # 维数D 种群规模ps
    ps,D = pos.shape[:2]   #ps 30 D 3
    mi = []
    scale = []
    X = []
    # 参数
    ergrd = 1e-4    #最小误差
    ergrdep = 10    #迭代终止最大连续稳定迭代次数
    cnt2 = 0

    # 初始局部最优位置
    pbest = pos
    # 求初始全局最优位置
    p = pbest
    out = []
    # ----------------------------原始
    for i in range(ps):
        out.append(pv(pbest[i, :],R,F,yleft,yright,xleft,xright))

    pbestval = out # 每个粒子当前函数值
    gbestval = max(pbestval)
    idx = pbestval.index(gbestval) # 全局最优函数值
    gbest = pbest[idx,:] # 全局极值
    tr = []# 保存当前全局最优函数值
    tr.append(gbestval)
    mbest = sum(pbest) / ps # 求平均最优位置
    tempLists = []
    # 开始迭代
    for t in range(maxinterations):
        belt = (0.6 - 0.8) * (maxinterations - t) / maxinterations + 0.8
        for i in range(ps):
            # 更新位置
            mbest_x = []  ## 存储mbest与粒子位置差的绝对值
            for j in range(D):
                fi = random.uniform(0, 1)
                p[i,j] = fi * pbest[i,j] + (1 - fi) * gbest[j]
                mbest_x = []  ## 存储mbest与粒子位置差的绝对值
                mbest_x.append(abs(mbest[j] - pos[i][j]))  # article_loc = pos
                u = m.log(1 / random.uniform(0, 1))
                if random.random() >= 0.5:
                    pos[i,j] = p[i,j] + belt * u * abs(mbest[j] - pos[i,j])
                    # pos[i,j] = list(
                    #     np.array(p[j]) + np.array([belt * u * x for x in mbest_x]))
                else:
                    pos[i,j] = p[i,j] - belt * u * abs(mbest[j] - pos[i,j])
                if pos[i,j] > pmax[j]:
                    pos[i,j] = pmax[j]
                if pos[i,j] < pmin[j]:
                    pos[i,j] = pmin[j]
            out[i] = pv(pos[i,:],R,F,yleft,yright,xleft,xright) #求函数值(可以替换成pv_mi,pv_rmi,pv_rirmi,pv_pc,pv_,)
            # 更新pbest
            if pbestval[i] <= out[i]:
                pbestval[i] = out[i]
                pbest[i,:]=pos[i,:]
        # 更新gbest
        iterbestval = max(pbestval)
        idx1 = pbestval.index(iterbestval)
        if gbestval <= iterbestval:
            gbestval = iterbestval
            gbest = pbest[idx1,:]

        mbest = sum(pbest) / ps   #求平均最优位置
        # tr[t + 1] = gbestval
        tr.append(gbestval)
        te = t
        temp = gbest.T
        pa1 = temp.T
        # tempList = [temp[0],temp[1],temp[2],gbestval]
        # tempLists.append(tempList)
        # print(temp)
        # scale.append(temp[0])
        # mi.append(gbestval)
        X.append(t)
        tmp1 = abs(tr[t] - gbestval)
        if tmp1 > ergrd:
            cnt2 = 0
        elif tmp1 <= ergrd:
            cnt2 = cnt2 + 1
            if cnt2>=ergrdep:
                break

    # plt.plot(range(len(mi)), mi, 'r', range(len(mi)), [gbestval for i in range(len(mi))],
    #              'b')
    # plt.show()
    # plt.plot(range(len(scale)), scale, 'r', range(len(scale)), [temp[0] for i in range(len(scale))],
    #          'b')
    # plt.show()
    pa = gbest
    # List_to_csv(tempLists,'data.csv')
    print('total interations:{}'.format(t + 1))
    print('最后参数:{}'.format(pa))

    return pa,mi

#############################图像匹配###################################
def test_img_reg(irimg_path,visimg_path):
    #读取图像
    # ir_img = cv2.imread(irimg_path)
    # vis_img = cv2.imread(visimg_path)
    ir_img = mpimg.imread(irimg_path)
    vis_img = mpimg.imread(visimg_path)
    # 获取红外与可见光的XMP信息
    ir_xmp = get_xmp_info(irimg_path)
    vis_xmp = get_xmp_info(visimg_path)

    # 计算图像单位像素的实际尺寸大小  d_ir = 0.017  d_vis = 0.001824457593688363
    d_ir =  IR_SENSOR_WIDTH / ir_xmp['ImageWidth']
    d_vis = VIS_SENSOR_WIDTH / vis_xmp['ImageWidth']
    #几何校正
    # ir_img = Geo_Correcition(ir_img,d_ir,ir_xmp['FocalLength'],ir_xmp['Flight-Yaw'])
    # vis_img = Geo_Correcition(vis_img,d_vis,vis_xmp['FocalLength'],vis_xmp['Flight-Yaw'])

    #桶型校正
    # ir_img = undistort(ir_img, d_ir, ir_xmp['FocalLength'], 1)
    # vis_img = undistort(vis_img, d_vis, vis_xmp['FocalLength'], 2)

    #校正后图像灰度化
    ir_img_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    vis_img_gray = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    # -------------------------------------优化QPSO匹配-----------------------------------------
    # 初始值
    scale = (vis_xmp['FocalLength'] * d_ir) / (ir_xmp['FocalLength'] * d_vis)
    # 初始值
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
    yleft, yright, xleft, xright = 500,2500,500,2500
    print('[5]通过QPSO进行图像尺度参数优化...')
    begin_time = time()
    pa, mi =QPSO(ir_img_gray, vis_img_gray, pos, maxinterations, pmax, pmin, yleft, yright, xleft, xright)
    end_time = time()
    run_time = end_time - begin_time
    print('QPSO算法运行时间：', run_time)  # 该循环程序运行时间： 1.4201874732
    H = np.array([[pa[0], pa[1], pa[2]], [pa[3], pa[4], pa[5]], [0, 0, 1]], np.float32)
    image_output = cv2.warpPerspective(ir_img_gray, H, (vis_img_gray.shape[1], vis_img_gray.shape[0]))
    match_img = cv2.addWeighted(image_output, 0.5, vis_img_gray, 0.5, 0)
    return pa,match_img
#############################图像匹配###################################
