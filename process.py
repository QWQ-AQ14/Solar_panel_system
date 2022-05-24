# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'process.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(812, 389)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.funsion_top = QtWidgets.QFrame(self.centralwidget)
        self.funsion_top.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.funsion_top.setFrameShadow(QtWidgets.QFrame.Raised)
        self.funsion_top.setObjectName("funsion_top")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.funsion_top)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.IR_BUTTON = QtWidgets.QPushButton(self.funsion_top)
        self.IR_BUTTON.setObjectName("IR_BUTTON")
        self.horizontalLayout_2.addWidget(self.IR_BUTTON)
        self.VIS_BUTTON = QtWidgets.QPushButton(self.funsion_top)
        self.VIS_BUTTON.setObjectName("VIS_BUTTON")
        self.horizontalLayout_2.addWidget(self.VIS_BUTTON)
        self.FUSION_BUTTON = QtWidgets.QPushButton(self.funsion_top)
        self.FUSION_BUTTON.setObjectName("FUSION_BUTTON")
        self.horizontalLayout_2.addWidget(self.FUSION_BUTTON)
        self.RES_BUTTON = QtWidgets.QPushButton(self.funsion_top)
        self.RES_BUTTON.setObjectName("RES_BUTTON")
        self.horizontalLayout_2.addWidget(self.RES_BUTTON)
        self.verticalLayout.addWidget(self.funsion_top)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.IR = QtWidgets.QLabel(self.frame_2)
        self.IR.setText("")
        self.IR.setObjectName("IR")
        self.horizontalLayout.addWidget(self.IR)
        self.VIS = QtWidgets.QLabel(self.frame_2)
        self.VIS.setText("")
        self.VIS.setObjectName("VIS")
        self.horizontalLayout.addWidget(self.VIS)
        self.RES = QtWidgets.QLabel(self.frame_2)
        self.RES.setText("")
        self.RES.setObjectName("RES")
        self.horizontalLayout.addWidget(self.RES)
        self.verticalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.IR_BUTTON.clicked.connect(self.IR.clear) # type: ignore
        self.VIS_BUTTON.clicked.connect(self.VIS.clear) # type: ignore
        self.FUSION_BUTTON.clicked.connect(self.RES.clear) # type: ignore
        self.RES_BUTTON.clicked.connect(self.RES.clear) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.IR_BUTTON.setText(_translate("MainWindow", "输入红外图像"))
        self.VIS_BUTTON.setText(_translate("MainWindow", "输入可见光图像"))
        self.FUSION_BUTTON.setText(_translate("MainWindow", "图像融合"))
        self.RES_BUTTON.setText(_translate("MainWindow", "保存结果图"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
