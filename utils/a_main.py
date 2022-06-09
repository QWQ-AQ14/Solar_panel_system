import cv2
from m_ir_vis_reg import image_matching
from m_get_mosaic_img import get_mosaic_img
from test_QPSO_REG_SIX import test_img_reg
from m_module_match import *
from time import *
import cv2

#############################test######################
irimg_path = r"E:/xlq/PycharmProjects/LearnCV/TIFF_MATCH/ImageRegistration/images/DJI_20210803111304_0324_T.JPG"
visimg_path = r"E:/xlq/PycharmProjects/LearnCV/TIFF_MATCH/ImageRegistration/images/DJI_20210803111305_0324_W.JPG"
mosaic_path = r"E:/xlq/PycharmProjects/LearnCV/TIFF_MATCH/ImageRegistration/images/GS.tif"
#############################test######################
begin_time = time()
# 1、红外与可见光图像配准获取配准参数
## 参数：红外与可见光的图像路径
## return：最佳配准参数以及融合图
best_para,best_match_img = test_img_reg(irimg_path,visimg_path)
# 2、对可见光和拼接图进行分割
#获取拼接子图
vis_mosaic_img = get_mosaic_img(visimg_path,mosaic_path)
# 分割
visimg = cv2.imread(visimg_path)
#   ----------------阈值分割
visimg_mask = get_mask(visimg, hsv_range=[55, 135, 45, 255, 70, 255])
# 差别主要在于亮度值v
mosaic_mask = get_mask(vis_mosaic_img[:,:,0:3], hsv_range=[55, 135, 45, 255, 90, 255])
# -------------通过regionprops函数获取分割区域-------------------
mosaic_list_of_region = get_region(mosaic_mask)
sigle_list_of_region = get_region(visimg_mask)

# 3.获取缺陷点信息
ir_defect_point=[(488.938353636101, 367.227431842042), (157.195520661313, 336.412105945077), (387.938964113797, 375.686286230821), (532.277680498913, 189.090517555979)]
best_para = np.array([[3.6295,0.0056,832.5334],
             [0.0159,3.5492,585.1608],
             [0,0,1]])
## 获取红外图像中对应的可见光中缺陷点信息
vis_defect_point = get_point(best_para,ir_defect_point)
## 遍历缺陷点
defect_lon_lat = []
for point in vis_defect_point:
    res_point = []
    single_region, mosaic_region = get_target_module(point, sigle_list_of_region,
                                                     mosaic_list_of_region)
    if single_region == 0:
        defect_lon_lat.append([0,0])
    else:
        # 获取拼接子图组串的经纬度坐标
        ds = gdal.Open(mosaic_path)
        lon, lat = pixel2latlon(ds, mosaic_region.centroid[1], mosaic_region.centroid[0])
        res_point.append(lon)
        res_point.append(lat)
        defect_lon_lat.append(res_point)

end_time = time()
run_time = end_time - begin_time
print('总共运行时间：', run_time)

