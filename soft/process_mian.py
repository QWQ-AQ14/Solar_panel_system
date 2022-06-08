
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import math
import numpy as np
import matplotlib.image as mpimg
# import GUI files
# #############################################
from process import *
from utils.m_ir_vis_reg import image_matching
# #############################################

# Main Window Class
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # code
        self.ir_filename = None
        self.vis_filename = None  # 获取图片路径
        self.tmp = None
        # self.filename = 'Snapshot ' + str(
        #     time.strftime("%Y-%b-%d at %H.%M.%S %p")) + '.png'  # Will hold the image address location
        self.ui.IR_BUTTON.clicked.connect(self.loadImageIR)  # type: ignore
        self.ui.VIS_BUTTON.clicked.connect(self.loadImageVIS)  # type: ignore
        self.ui.FUSION_BUTTON.clicked.connect(self.fusion)  # type: ignore
        self.ui.RES_BUTTON.clicked.connect(self.savePhoto)  # type: ignore

    def loadImageIR(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.ir_filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = mpimg.imread(self.ir_filename)
        self.setPhoto_IR(self.image)


    def loadImageVIS(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.vis_filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = mpimg.imread(self.vis_filename)
        self.setPhoto_VIS(self.image)


    def setPhoto_IR(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.IR_img = image
        frame = imutils.resize(image, width=640)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.IR.setPixmap(QtGui.QPixmap.fromImage(image))


    def setPhoto_VIS(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.VIS_img = image
        frame = imutils.resize(image, width=640)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.VIS.setPixmap(QtGui.QPixmap.fromImage(image))


    def setPhoto_RES(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        frame = imutils.resize(image, width=640)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.RES.setPixmap(QtGui.QPixmap.fromImage(image))

    def R_VIS_Match(self,scale):
        IRGaryImg = self.IR_img
        VISGrayimg = self.VIS_img
        VISGrayimg = cv2.resize(VISGrayimg, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        # cp_readImage.showImg(VISGrayimg)
        # 获取变化后可见光图像的中心点坐标
        h1, w1 = VISGrayimg.shape[:2]
        x1 = math.floor(h1 / 2)
        y1 = math.floor(w1 / 2)
        # 获取缩小后的可见光的中心点坐标
        h2, w2 = IRGaryImg.shape[:2]
        # x2 = round(h2/2)
        # y2 = round(w2/2)
        # 中心偏移量
        x, y = 0, 0
        mask = np.zeros(VISGrayimg.shape, dtype=np.uint8)
        mask[int(x1 - math.floor(h2 / 2) + math.floor(x)):int(x1 + math.floor(h2 / 2) + math.floor(x)),
        int(y1 - math.floor(w2 / 2) + math.floor(y)):int(y1 + math.floor(w2 / 2) + math.floor(y))] = IRGaryImg
        VISGrayimg = cv2.resize(VISGrayimg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # read.showimg(VISGrayimg)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # read.showimg(mask)
        dst = cv2.addWeighted(VISGrayimg, 0.5, mask, 0.5, 0)
        return dst

    def fusion(self):
        self.fusion_img = self.R_VIS_Match(3.63)
        self.setPhoto_RES(self.fusion_img)



    def savePhoto(self):
        """ This function will save the image"""
        # here provide the output file name
        # lets say we want to save the output as a time stamp
        # uncomment the two lines below

        # import time
        # filename = 'Snapshot '+str(time.strftime("%Y-%b-%d at %H.%M.%S %p"))+'.png'

        # Or we can give any name such as output.jpg or output.png as well
        # filename = 'Snapshot.png'

        # Or a much better option is to let user decide the location and the extension
        # using a file dialog.

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]

        cv2.imwrite(filename, self.fusion_img)
        print('Image saved as:', self.filename)



# Execute App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
