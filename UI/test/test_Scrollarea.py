import sys, os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QSplitter, QVBoxLayout,
                             QGroupBox, QScrollArea, QRadioButton, QCheckBox,
                             QLabel)
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtCore import Qt


class DemoScrollArea(QWidget):
    def __init__(self, parent=None):
        super(DemoScrollArea, self).__init__(parent)

        # 设置窗口标题
        self.setWindowTitle('实战PyQt5: QScrollArea Demo!')
        # 设置窗口大小
        self.resize(480, 360)

        self.initUi()

    def initUi(self):
        mainLayout = QVBoxLayout(self)

        hSplitter = QSplitter(Qt.Horizontal)

        saLeft = QScrollArea(self)
        disp_img = QLabel(self)
        disp_img.setPixmap(QPixmap(os.path.dirname(__file__) + '/trp.jpg'))
        saLeft.setBackgroundRole(QPalette.Dark)
        saLeft.setWidget(disp_img)

        saRight = QScrollArea(self)
        # 滚动区域的Widget
        scrollAreaWidgetContents = QWidget()
        vLayout = QVBoxLayout(scrollAreaWidgetContents)
        vLayout.addWidget(self.createFirstExclusiveGroup())
        vLayout.addWidget(self.createSecondExclusiveGroup())
        vLayout.addWidget(self.createNonExclusiveGroup())
        scrollAreaWidgetContents.setLayout(vLayout)
        saRight.setWidget(scrollAreaWidgetContents)

        hSplitter.addWidget(saLeft)
        hSplitter.addWidget(saRight)
        mainLayout.addWidget(hSplitter)
        self.setLayout(mainLayout)

    def createFirstExclusiveGroup(self):
        groupBox = QGroupBox('Exclusive Radio Buttons', self)

        radio1 = QRadioButton('&Radio Button 1', self)
        radio1.setChecked(True)
        radio2 = QRadioButton('R&adio button 2', self)
        radio3 = QRadioButton('Ra&dio button 3', self)

        vLayout = QVBoxLayout(groupBox)
        vLayout.addWidget(radio1)
        vLayout.addWidget(radio2)
        vLayout.addWidget(radio3)
        vLayout.addStretch(1)

        groupBox.setLayout(vLayout)

        return groupBox

    def createSecondExclusiveGroup(self):
        groupBox = QGroupBox('E&xclusive Radio Buttons', self)
        groupBox.setCheckable(True)
        groupBox.setChecked(True)

        radio1 = QRadioButton('Rad&io button1', self)
        radio1.setChecked(True)
        radio2 = QRadioButton('Radi&o button2', self)
        radio3 = QRadioButton('Radio &button3', self)
        chkBox = QCheckBox('Ind&ependent checkbox', self)

        vLayout = QVBoxLayout(groupBox)
        vLayout.addWidget(radio1)
        vLayout.addWidget(radio2)
        vLayout.addWidget(radio3)
        vLayout.addWidget(chkBox)
        vLayout.addStretch(1)

        groupBox.setLayout(vLayout)

        return groupBox

    def createNonExclusiveGroup(self):
        groupBox = QGroupBox('No-Exclusive Checkboxes', self)
        groupBox.setFlat(True)

        chBox1 = QCheckBox('&Checkbox 1')
        chBox2 = QCheckBox('C&heckbox 2')
        chBox2.setChecked(True)
        tristateBox = QCheckBox('Tri-&state buttton')
        tristateBox.setTristate(True)
        tristateBox.setCheckState(Qt.PartiallyChecked)

        vLayout = QVBoxLayout(groupBox)
        vLayout.addWidget(chBox1)
        vLayout.addWidget(chBox2)
        vLayout.addWidget(tristateBox)
        vLayout.addStretch(1)

        groupBox.setLayout(vLayout)

        return groupBox


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DemoScrollArea()
    window.show()
    sys.exit(app.exec())