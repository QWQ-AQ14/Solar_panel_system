
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

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
        self.filename = None  # 获取图片路径
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
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.setPhoto_IR(self.image)


    def loadImageVIS(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.image = cv2.imread(self.filename)
        self.setPhoto_VIS(self.image)


    def setPhoto_IR(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.IR_img = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.IR.setPixmap(QtGui.QPixmap.fromImage(image))


    def setPhoto_VIS(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.VIS_img = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.VIS.setPixmap(QtGui.QPixmap.fromImage(image))


    def setPhoto_RES(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """

        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.ui.RES.setPixmap(QtGui.QPixmap.fromImage(image))


    def fusion(self):
        self.fusion_img = cv2.addWeighted(self.IR_img, 0.5, self.VIS_img, 0.5, 0)
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
