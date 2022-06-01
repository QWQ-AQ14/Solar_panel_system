import cv2

# 红外与可见光的传感器尺寸
IR_SENSOR_WIDTH = 10.88
IR_SENSOR_HEIGHT = 8.7
VIS_SENSOR_WIDTH = 7.4
VIS_SENSOR_HEIGHT = 5.55



def image_matching(irimg,visimg):
    # 获取红外与可见光的尺寸
    ir_h,ir_w = irimg.shape[:2]
    vis_h,vis_w = visimg.shape[:2]

    # 计算图像单位像素的实际尺寸大小
    d_ir = ir_h / IR_SENSOR_WIDTH
    d_vis = vis_h / VIS_SENSOR_WIDTH

