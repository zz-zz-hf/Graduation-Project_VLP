import sys

from PyQt5.QtWidgets import QSizePolicy

from scroll import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import os
from UI.uilt.GraphMatlab import Figure_Canvas,cal_err
import UI.uilt.EnvData as envdata
from UI.uilt.entity.VLPThreadEntity import VLPThreadEntity
from UI.uilt.Simlation import euler_to_rotation_matrix,genatate_camerapoints
from UI.uilt.EnvData import camera_matrix,dist_coeffs,colors
import cv2 as cv
from scipy import stats
from qt_material import apply_stylesheet


class Main_Window(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Main_Window, self).__init__()
        self.setupUi(self)
        self.init_UI()
        self.slot_init()


    def init_UI(self):
        self.scrollAreaWidgetContents_2.setLayout(self.verticalLayout) # 设置两个groupbox覆盖所有的scroll
        self.groupBox.setTitle("Graph")
        self.add_box2(self.groupBox_2)# groupbox_2添加按钮
        self.groupBox_2.setTitle("Button")
        self.groupBox_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)

        self.scrollAreaWidgetContents.setLayout(self.verticalLayout_2)
        self.page.setLayout(self.verticalLayout_3)

        # self.scrollAreaWidgetContents_1.setLayout(self.gridLayout)
        # self.comboBox.addItems(["文本1","文本2","文本3","文本4","文本5"])
        # self.comboBox.setCurrentIndex(3)
        """================================================="""

        self.verticalLayout_5.setContentsMargins(10,10,10,10)
        self.scrollAreaWidgetContents_3.setLayout(self.verticalLayout_5)
        # 设置标题label长度包裹内容
        title_vlp="VLP"
        self.label.setText(title_vlp)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # 设置仿真3d图像，绘制led和camera（动态刷新）坐标
        self.spin_camera = {}  # {index:[x,y,z]……}
        self.sim_camerapos = []
        self.init_led_camera(200)
        self.draw_led_camera(self.sim_camerapos)
        self.groupBox_5.setTitle('')
        self.groupBox_5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # 动态添加，删除组件  更新draw_led_camera
        self.doubleSpinBox.setSingleStep(0.1)
        self.doubleSpinBox_2.setSingleStep(0.1)
        self.doubleSpinBox_3.setSingleStep(0.1)
        self.doubleSpinBox.setRange(0,4.2)
        self.doubleSpinBox_2.setRange(0,2.7)
        self.doubleSpinBox_3.setRange(0,2.7)
        self.groupBox_3.setLayout(self.verticalLayout_7)
        self.pushButton_5.clicked.connect(self.add_component)
        self.pushButton_5.setText("+")

        # 获得组件数据
        self.pushButton_4.setText("开始仿真")
        self.pushButton_4.clicked.connect(self.start_simlation)




    def add_component(self):
        # 更新绘制 sim_camera
        self.updata_sim_camerapos()
        self.draw_led_camera(self.sim_camerapos)

        horizontalLayout = QtWidgets.QHBoxLayout()
        pushbutton = QtWidgets.QPushButton(self.widget)

        label_x=QtWidgets.QLabel(self.widget)
        label_x.setText("x:")
        spinbox_x=QtWidgets.QDoubleSpinBox(self.widget)
        spinbox_x.setRange(0,4.2)
        spinbox_x.setSingleStep(0.1)

        label_y=QtWidgets.QLabel(self.widget)
        label_y.setText("y:")
        spinbox_y=QtWidgets.QDoubleSpinBox(self.widget)
        spinbox_y.setRange(0,2.7)
        spinbox_y.setSingleStep(0.1)

        label_z = QtWidgets.QLabel(self.widget)
        label_z.setText("z:")
        spinbox_z = QtWidgets.QDoubleSpinBox(self.widget)
        spinbox_z.setRange(0, 2.7)
        spinbox_z.setSingleStep(0.1)

        horizontalLayout.addWidget(pushbutton)

        horizontalLayout.addWidget(label_x)
        horizontalLayout.addWidget(spinbox_x)
        horizontalLayout.addWidget(label_y)
        horizontalLayout.addWidget(spinbox_y)
        horizontalLayout.addWidget(label_z)
        horizontalLayout.addWidget(spinbox_z)

        self.verticalLayout_6.addLayout(horizontalLayout)
        layout_id = id(horizontalLayout)
        pushbutton.setText(str(layout_id))
        pushbutton.clicked.connect(lambda: self.delete_index(layout_id))
        print("add_component" + str(layout_id))



    def delete_index(self,layout_id):
        print("delete"+str(layout_id))
        self.updata_sim_camerapos(layout_id)

        # 删除子layout中的组件,然后删除layout
        for i in range(self.verticalLayout_6.count()):
            # 检测layout的id是否是
            widget_id=id(self.verticalLayout_6.itemAt(i))
            if widget_id==layout_id:

                for j in range(self.verticalLayout_6.itemAt(i).count()):# 删除子组件
                    self.verticalLayout_6.itemAt(i).itemAt(j).widget().deleteLater()
                self.verticalLayout_6.itemAt(i).deleteLater()
        # 更新绘制 sim_camera
        self.draw_led_camera(self.sim_camerapos)

    def updata_sim_camerapos(self,del_id=-1):
        """
        更新camera坐标数据
        :param del_id: 添加时不使用这个参数，删除坐标使用排除没有及时更新的位置
        :return:
        """
        print("updata_sim_camerapos")
        self.sim_camerapos.clear()
        self.spin_camera.clear()
        for i in range(self.verticalLayout_6.count()):
            pos=[]
            if id(self.verticalLayout_6.itemAt(i))==del_id or (i==self.verticalLayout_6.count()-1 and del_id!=-1):
                continue
            for j in range(self.verticalLayout_6.itemAt(i).count()):
                if isinstance(self.verticalLayout_6.itemAt(i).itemAt(j).widget(), QtWidgets.QDoubleSpinBox):
                    spin_value=self.verticalLayout_6.itemAt(i).itemAt(j).widget().value()
                    pos.append(spin_value)
                    # print(str(i)+" "+str(j)+" "+str(spin_value))
            self.spin_camera[i]=pos
        # {} -> []

        for ikey in list(self.spin_camera.keys()):
            self.sim_camerapos.append(self.spin_camera[ikey])

    def init_led_camera(self,min_width):
        self.LCFigure = Figure_Canvas()
        self.ax3d = self.LCFigure.fig.add_subplot(projection='3d')
        FigureLayout = QtWidgets.QGridLayout(self.groupBox_5)
        FigureLayout.addWidget(self.LCFigure, 0, 0, 1, 1)
        FigureLayout.setColumnMinimumWidth(0, min_width)
        FigureLayout.setRowMinimumHeight(0, min_width)
    def draw_led_camera(self,camera_poss):
        # 3D 图像 绘制led灯和standard_pos和cal_pos
        # ax3d = LCFigure.fig.add_subplot(projection='3d')
        self.ax3d.cla()
        x = [row[0] for row in envdata.LED_3D]
        y = [row[1] for row in envdata.LED_3D]
        z = [row[2] for row in envdata.LED_3D]
        self.ax3d.scatter(x, y, z, label='LED', c='k')  # LED
        # 绘制Camera点图
        if len(camera_poss) >= 1:
            x_c = [row[0] for row in camera_poss]
            y_c = [row[1] for row in camera_poss]
            z_c = [row[2] for row in camera_poss]
            self.ax3d.scatter(x_c, y_c, z_c, label='Camera', c='b', marker='*')  # pnp计算得到的点
        self.ax3d.set_xlim(0, 4.2)
        self.ax3d.set_ylim(0, 2.7)
        self.ax3d.set_zlim(0, 2.7)
        self.ax3d.legend(loc='right', bbox_to_anchor=(1.04, 1))  # 添加图例
        self.LCFigure.draw()

    def start_simlation(self):
        self.init_simlation(200)
        # 如果没有输入值则按照默认的值计算
        # x_angles=list(range(int(self.lineEdit_5.text()), int(self.lineEdit_6.text())+1))
        # y_angles = list(range(int(self.lineEdit_3.text()), int(self.lineEdit_4.text()) + 1))
        # z_angles = list(range(int(self.lineEdit.text()), int(self.lineEdit_2.text()) + 1))
        x_angles=[0]
        y_angles=[0]
        z_angles=[5,10,15,20,25]
        pnpts_list=[] #[index:[poss]……]
        for i in range(len(self.sim_camerapos)):
            pnpt_list=[]
            for anglex in x_angles:
                for angley in y_angles:
                    for anglez in z_angles:
                        R_gen= euler_to_rotation_matrix(anglex, angley, anglez)
                        t_gen=-np.array(self.sim_camerapos[i])
                        pixel_point_standard, world_point_standard = genatate_camerapoints((3264, 2464),
                                                                                           R_gen, t_gen)
                        if pixel_point_standard.shape[0] < 3:
                            print("控制点数量不足")
                            continue
                        (success, rpnp_vectorc, t_pnp) = cv.solvePnP(world_point_standard, pixel_point_standard,
                                                                      camera_matrix, dist_coeffs,
                                                                      flags=cv.SOLVEPNP_ITERATIVE)  # PnP求解
                        pnpt_list.append((-t_pnp).reshape(1,3).tolist()[0]) # t_pnp (3,1)
            pnpts_list.append(pnpt_list)
            self.draw_simlation(pnpts_list)

    def init_simlation(self,min_width):
        # 绘制四个图 3d 2d CDF 频率
        self.SimFigure3d = Figure_Canvas()
        self.SimFigure2d = Figure_Canvas()
        self.SimFigurecdf = Figure_Canvas()
        self.SimFigurefre = Figure_Canvas()

        FigureLayout = QtWidgets.QGridLayout(self.groupBox_4)

        FigureLayout.addWidget(self.SimFigure3d, 0, 0, 1, 1)
        FigureLayout.addWidget(self.SimFigure2d, 1, 0, 1, 1)
        FigureLayout.addWidget(self.SimFigurecdf, 2, 0, 1, 1)
        FigureLayout.addWidget(self.SimFigurefre, 3, 0, 1, 1)

        FigureLayout.setColumnMinimumWidth(0, min_width)
        FigureLayout.setRowMinimumHeight(0, min_width)
        FigureLayout.setRowMinimumHeight(1, min_width)
        FigureLayout.setRowMinimumHeight(2, min_width)
        FigureLayout.setRowMinimumHeight(3, min_width)

        self.simax3d = self.SimFigure3d.fig.add_subplot(projection='3d')
        self.simax2d = self.SimFigure2d.fig.add_subplot(111)
        self.simaxcdf = self.SimFigurecdf.fig.add_subplot(111)
        self.simaxfre = self.SimFigurefre.fig.add_subplot(111)




    def draw_simlation(self,standard_pnp):
        self.simax3d.cla()
        x = [row[0] for row in envdata.LED_3D]
        y = [row[1] for row in envdata.LED_3D]
        z = [row[2] for row in envdata.LED_3D]
        self.simax3d.scatter(x, y, z, label='LED', c='k')  # LED
        # 相机和pnp点
        index_pnp = 0
        for index in range(len(standard_pnp)):  # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            pnplists=np.array(standard_pnp[index])
            if index_pnp == 0:
                self.simax3d.scatter(pnplists[:, 0], pnplists[:, 1], pnplists[:, 2],
                             label='PnP calculate', c=envdata.colors_list[index], marker='x', alpha=0.4)  # pnp计算得到的点
            else:
                self.simax3d.scatter(pnplists[:, 0], pnplists[:, 1], pnplists[:, 2],
                             label=None, c=envdata.colors_list[index], marker='x', alpha=0.4)
            index_pnp = index_pnp + 1
        x_standard = [self.sim_camerapos[i][0] for i in range(len(standard_pnp))]
        y_standard = [self.sim_camerapos[i][1] for i in range(len(standard_pnp))]
        z_standard = [self.sim_camerapos[i][2] for i in range(len(standard_pnp))]
        self.simax3d.scatter(x_standard, y_standard, z_standard, label='Standard', c='r', marker='*')  # camera_standard
        self.simax3d.set_xlim(0, 4.2)
        self.simax3d.set_ylim(0, 2.7)
        self.simax3d.set_zlim(0, 2.7)
        self.simax3d.legend(loc='right', bbox_to_anchor=(1.04, 1))  # 添加图例
        self.SimFigure3d.draw()

        # 绘制2d图像
        self.simax2d.cla()
        index_pnp = 0
        for index in range(len(standard_pnp)):  # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            pnplists = np.array(standard_pnp[index])
            if index_pnp == 0:
                self.simax2d.scatter(pnplists[:, 0], pnplists[:, 1],
                                     label='PnP calculate', c=envdata.colors_list[index], marker='x',
                                     alpha=0.4)  # pnp计算得到的点
            else:
                self.simax2d.scatter(pnplists[:, 0], pnplists[:, 1],
                                     label=None, c=envdata.colors_list[index], marker='x', alpha=0.4)
            index_pnp = index_pnp + 1
        x_standard = [self.sim_camerapos[i][0] for i in range(len(standard_pnp))]
        y_standard = [self.sim_camerapos[i][1] for i in range(len(standard_pnp))]
        self.simax2d.scatter(x_standard, y_standard, label='Standard', c='r', marker='*')  # camera_standard
        self.simax2d.set_xlim(0, 4.2)
        self.simax2d.set_ylim(0, 2.7)
        self.simax2d.legend(loc='right', bbox_to_anchor=(1.04, 1))  # 添加图例
        self.SimFigure2d.draw()

        # 绘制CDF图和频率分布
        self.simaxcdf.cla()
        errors = []
        for i in range(len(standard_pnp)):
            standard_camera = np.array(self.sim_camerapos[i])
            for pnp_pos in standard_pnp[i]:  # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
                errors.append(float(cal_err(np.array(pnp_pos), standard_camera)))

        res = stats.relfreq(errors, numbins=25)  # 给定数据集的相对频率分布
        x_cdf = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
        y_cdf = np.cumsum(res.frequency)
        self.simaxcdf.plot(x_cdf, y_cdf, label='CDF')
        self.simaxcdf.legend()
        self.SimFigurecdf.draw()

        self.simaxfre.cla()
        self.simaxfre.bar(x_cdf, res.frequency, width=res.binsize, label='Histogram')
        self.simaxfre.legend()
        self.SimFigurefre.draw()

























    def add_box2(self,groupBox):
        radio1 = QtWidgets.QRadioButton('&Radio Button 1', self)
        radio1.setChecked(True)
        radio2 = QtWidgets.QRadioButton('R&adio button 2', self)
        radio3 = QtWidgets.QRadioButton('Ra&dio button 3', self)

        vLayout = QtWidgets.QVBoxLayout(groupBox)
        vLayout.addWidget(radio1)
        vLayout.addWidget(radio2)
        vLayout.addWidget(radio3)
        vLayout.setAlignment(Qt.AlignCenter)


        groupBox.setLayout(vLayout)

    def slot_init(self):
        self.pushButton.clicked.connect(self.btn1)
        # self.pushButton_2.clicked.connect(self.btn2)
        # self.pushButton_3.clicked.connect(self.btn3)
        # self.pushButton_4.clicked.connect(self.btn4)

        # self.comboBox.currentIndexChanged.connect(self.comboBox_change)

    def btn1(self):
        print("btn1")
        self.plot_graph(self.groupBox,100)

    def comboBox_change(self):
        print(self.comboBox.currentText())



    def btn2(self):
        print("btn2")

    def btn3(self):
        print("btn3")

    def btn4(self):
        print("btn4")




    def plot_graph(self,parents,min_width):

        LineFigure = Figure_Canvas()
        SurfFigure=Figure_Canvas()

        FigureLayout = QtWidgets.QGridLayout(parents)
        FigureLayout.addWidget(LineFigure,0,0,1,1)
        FigureLayout.addWidget(SurfFigure, 1, 0, 1, 1)
        FigureLayout.setColumnMinimumWidth(0,min_width)
        FigureLayout.setRowMinimumHeight(0,min_width)
        FigureLayout.setRowMinimumHeight(1, min_width)


        # 2D 点图
        ax = LineFigure.fig.add_subplot(111)
        x = [1, 2, 3, 4, 5]  # x 坐标
        y = [2, 3, 5, 7, 11]  # y 坐标
        ax.scatter(x,y)
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * 1.0)# 设置1：1固定x,y比例
        # 3D 图像
        ax3d = SurfFigure.fig.add_subplot(projection='3d')
        Xc, Zc = np.meshgrid(np.arange(-1, 1, 0.005), np.arange(0, 1, 0.005))  # 自变量网格坐标
        Yc = Xc ** 2
        ax3d.plot_surface(Xc, Yc, Zc, cmap='rainbow')  # 数据结构是二维[[]]









# #matlab画图对象
# class Figure_Canvas(FigureCanvas):
#     def __init__(self,parent=None,width=3.9,height=2.7,dpi=100):
#         self.fig=Figure(figsize=(width,height),dpi=100)
#         super(Figure_Canvas,self).__init__(self.fig)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 创建应用程序
    mainwindow = Main_Window()  # 创建主窗口
    apply_stylesheet(app, theme='light_cyan.xml')
    mainwindow.show()  # 显示窗口
    sys.exit(app.exec_())  # 程序执行循环