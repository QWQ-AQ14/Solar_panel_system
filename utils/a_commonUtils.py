import cv2
import numpy as np
from tkinter import filedialog
# 弹窗选取图像
 ## 读取图像，解决imread不能读取中文路径的问题

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
# 显示图像
def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def choseImage():
    ftypes = [
        ("JPG", "*.jpg;*.JPG;*.JPEG"),
        ("PNG", "*.png;*.PNG"),
        ("GIF", "*.gif;*.GIF"),
        ("All files", "*.*")
    ]
    file_path = filedialog.askopenfilename(filetypes=ftypes)
    image_src = cv_imread(file_path)
    return image_src

# 测试红外图像变换参数
def IR_img_changebig(IrImg,VisImg,affmat):
    H = np.array([[pos[0], pos[1], pos[2]], [pos[3], pos[4], pos[5]], [0, 0, 1]], np.float32)
    trans_ir = cv2.warpPerspective(IrImg, H, (VisImg.shape[1], VisImg.shape[0]))
    match_img = cv2.addWeighted(trans_ir, 0.5, VisImg, 0.5, 0)
    return trans_ir, VisImg, match_img