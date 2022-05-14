# #############################################
# import
# #############################################
import sys
import os
from PySide2 import *
from PyQt5.QtWidgets import QApplication, QMainWindow,QGraphicsDropShadowEffect,QSizeGrip
from qt_material import *
# #############################################

# #############################################
# import GUI files
# #############################################
from system import *
# #############################################

# Main Window Class
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Load Style sheet, this will override and fonts selected in qt 重写样式
        # designer
        # apply_stylesheet(app, theme="dark_cyan.xml")
        # 移除系统默认的×
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # Set main background to transparent
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # # Shadow effect style
        # self.shadow = QGraphicsDropShadowEffect(self)
        # self.shadow.setBlurRadius(50)
        # self.shadow.setXOffset(0)
        # self.shadow.setYOffset(0)
        # self.shadow.setColor(QColor(0, 92, 157, 550))
        # # Apply shadow to central widget
        # self.ui.centralwidget.setGraphicsEffect(self.shadow)
        # Set window icon
        # This icon and title will not appear on our app main window because
        # we removed the title bar
        self.setWindowIcon(QtGui.QIcon(":/icons/icons/feather/airplay.svg"))
        # Set window title
        self.setWindowTitle("Solar panel System")
        # 拉动左下角自适应窗口
        QSizeGrip(self.ui.size_grip)

        # # Navigation Window Bar
        # Minimize window 最小化按钮
        self.ui.minimize_window_button.clicked.connect(
            lambda: self.showMinimized())

        # Close window 关闭窗口按钮事件
        self.ui.close_window_button.clicked.connect(
            lambda: self.close())

        # Restore/Maximize Window 自适应窗口按钮事件
        self.ui.restore_window_button.clicked.connect(
            lambda: self.restore_or_maximize_window())

        # STACKED PAGES NAVIGATION /////////////////////////
        # Using side menu buttons 设置按钮对应窗口的切换

        # navigate to CPU page
        self.ui.fusion_button.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(
                self.ui.fusion))

        # navigate to Battery page
        self.ui.detect_button.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(
                self.ui.detect))

        # navigate to system info page
        self.ui.position_button.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(
                self.ui.position))
        self.show()

    # update restore button icon on maximizing or minimizing window
    # 点击restore按钮 图标的变换
    def restore_or_maximize_window(self):
        # IF window is maximized
        if self.isMaximized():
            self.showNormal()
            # Change Icon
            self.ui.restore_window_button.setIcon(
                QtGui.QIcon(u":/icons/icons/feather/copy.svg"))
        else:
            self.showMaximized()
            # Change Icon
            self.ui.restore_window_button.setIcon(
                QtGui.QIcon(u":/icons/icons/feather/clipboard.svg"))


# Execute App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    sys.exit(app.exec_())
