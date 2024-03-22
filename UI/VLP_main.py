import os
import sys

from PyQt5.QtWidgets import QSizePolicy, QFileDialog, QMessageBox

from VLP  import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib.pyplot as plt
import numpy as np

from uilt.MyThread import VLPWorkThread
from uilt.GraphMatlab import plot_graph,plot_mutil_graph,Figure_Canvas,cal_err
import UI.uilt.EnvData as envdata
from UI.uilt.Simlation import euler_to_rotation_matrix,genatate_camerapoints
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
        """==============设置布局================"""
        self.verticalLayout.setContentsMargins(20,0,0,0)
        self.verticalLayout_3.setContentsMargins(0,10,0,0)
        self.verticalLayout_3.setSpacing(15)
        self.verticalLayout_10.setContentsMargins(0, 10, 0, 0)
        self.verticalLayout_10.setSpacing(15)

        self.verticalLayout_2.setContentsMargins(20, 0, 0, 0)






        """==============无线光定位展示================"""

        self.label_loading.setScaledContents(True) # 设置加载中图片的自适应label的大小
        # 设置组件使用layout填充，可以自由拉伸
        self.centralwidget.setLayout(self.horizontalLayout)
        self.frame.setLayout(self.verticalLayout)
        self.frame_2.setLayout(self.verticalLayout_2)
        self.page.setLayout(self.verticalLayout_5)
        self.tab.setLayout(self.verticalLayout_4)
        self.scrollAreaWidgetContents_2.setLayout(self.verticalLayout_3) # 设置两个groupbox覆盖所有的scroll
        # 设置选项的groupbox组件
        self.init_groupbox2_VLPres()
        # 在未选择是隐藏输入图片组件
        self.img_filepath=""
        self.label_input_img.setVisible(False)
        # 设置单张图片的检测
        self.pushButton_3.setText("Start VLP")
        self.pushButton_3.clicked.connect(self.single_detected)
        # 设置label_xx水平充满，垂直只包裹内部
        self.label_result.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # 显示matlab等结果图片组件
        self.groupBox.setTitle("Graph-result")
        """==============无线光批量检测================="""
        # 设置田间layout实现自由拉伸
        self.tab_2.setLayout(self.verticalLayout_7)
        self.scrollAreaWidgetContents_3.setLayout(self.verticalLayout_10)

        self.init_groupbox3_VLPres()

        self.pushButton_5.setText("批量检测")
        self.pushButton_5.clicked.connect(self.multi_detected)
        """===============无线光仿真=================="""
        self.page_2.setLayout(self.verticalLayout_6)
        self.tab_3.setLayout(self.verticalLayout_8)

        self.verticalLayout_11.setContentsMargins(10, 10, 10, 10)
        self.scrollAreaWidgetContents_5.setLayout(self.verticalLayout_11)
        # 设置标题label长度包裹内容
        title_vlp = "\n无线光定位（VLP）是一种利用光波作为信息传输介质的室内定位技术。它通常使用发光二极管（LED）作为信号发射端，通过调制光信号来传输信息，接收端采用光电传感器（如光电二极管或图像传感器）检测信号，利用定位算法获得接收端的位置。"
        self.label_13.setText(title_vlp)
        # self.label_13.setMaximumWidth(700)
        self.label_13.setWordWrap(True)
        self.label_13.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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
        self.doubleSpinBox.setRange(0, 4.2)
        self.doubleSpinBox_2.setRange(0, 2.7)
        self.doubleSpinBox_3.setRange(0, 2.7)
        self.groupBox_6.setTitle("Simulation settings")
        self.groupBox_6.setTitle("Simulation result")
        self.groupBox_6.setLayout(self.verticalLayout_12)

        self.pushButton_6.setStyleSheet("#pushButton_6 {  background-color: #DCDCDC; color: gray; border-style:none; border-radius: 15px; }")
        self.pushButton_6.clicked.connect(self.add_component)
        self.pushButton_6.setText("+")

        # 获得组件数据
        self.init_simlation(200)
        self.pushButton_7.setText("开始仿真")
        self.pushButton_7.clicked.connect(self.start_simlation)
        """===============YOLOv5训练结果=================="""
        self.tab_4.setLayout(self.verticalLayout_9)
        self.scrollAreaWidgetContents_6.setLayout(self.verticalLayout_14)
        # 显示文件夹中的图片
        self.yolopreference_show()

    def yolopreference_show(self):
        yolopt_dir=r'E:\Code\Py_pro\YOLOV5\yolov5-master\runs\train\exp2'
        files= os.listdir(yolopt_dir)
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 获得图片名字 对象
                lable_name=filename.split('\.')[0]
                imgpath=os.path.join(yolopt_dir,filename)
                pixmap = QPixmap(imgpath)
                scaledPixmap = pixmap.scaled(QSize(500, 400), aspectRatioMode=Qt.KeepAspectRatio)

                vlayout = QtWidgets.QVBoxLayout()
                label_yoloname=QtWidgets.QLabel(self.verticalLayoutWidget_8)
                label_yoloimg=QtWidgets.QLabel(self.verticalLayoutWidget_8)

                label_yoloimg.setPixmap(scaledPixmap)
                label_yoloimg.setAlignment(Qt.AlignCenter)  # 设置居中显示
                label_yoloname.setText(lable_name)
                label_yoloname.setAlignment(Qt.AlignCenter)

                vlayout.addWidget(label_yoloimg)
                vlayout.addWidget(label_yoloname)
                self.verticalLayout_14.addLayout(vlayout)









    def add_component(self):
        # 更新绘制 sim_camera
        self.updata_sim_camerapos()
        self.draw_led_camera(self.sim_camerapos)

        horizontalLayout = QtWidgets.QHBoxLayout()
        pushbutton = QtWidgets.QPushButton(self.groupBox_6)
        pushbutton.setStyleSheet("QPushButton  {  background-color: #DCDCDC; color: gray; border-style:none; border-radius: 15px; }")


        label_x=QtWidgets.QLabel(self.groupBox_6)
        label_x.setText("x:")
        spinbox_x=QtWidgets.QDoubleSpinBox(self.groupBox_6)
        spinbox_x.setRange(0,4.2)
        spinbox_x.setSingleStep(0.1)

        label_y=QtWidgets.QLabel(self.groupBox_6)
        label_y.setText("y:")
        spinbox_y=QtWidgets.QDoubleSpinBox(self.groupBox_6)
        spinbox_y.setRange(0,2.7)
        spinbox_y.setSingleStep(0.1)

        label_z = QtWidgets.QLabel(self.groupBox_6)
        label_z.setText("z:")
        spinbox_z = QtWidgets.QDoubleSpinBox(self.groupBox_6)
        spinbox_z.setRange(0, 2.7)
        spinbox_z.setSingleStep(0.1)

        horizontalLayout.addWidget(pushbutton)

        horizontalLayout.addWidget(label_x)
        horizontalLayout.addWidget(spinbox_x)
        horizontalLayout.addWidget(label_y)
        horizontalLayout.addWidget(spinbox_y)
        horizontalLayout.addWidget(label_z)
        horizontalLayout.addWidget(spinbox_z)

        self.verticalLayout_13.addLayout(horizontalLayout)
        layout_id = id(horizontalLayout)
        pushbutton.setText('-')
        pushbutton.clicked.connect(lambda: self.delete_index(layout_id))
        print("add_component" + str(layout_id))
    
    def delete_index(self,layout_id):
        print("delete"+str(layout_id))
        self.updata_sim_camerapos(layout_id)

        # 删除子layout中的组件,然后删除layout
        for i in range(self.verticalLayout_13.count()):
            # 检测layout的id是否是
            widget_id=id(self.verticalLayout_13.itemAt(i))
            if widget_id==layout_id:

                for j in range(self.verticalLayout_13.itemAt(i).count()):# 删除子组件
                    self.verticalLayout_13.itemAt(i).itemAt(j).widget().deleteLater()
                self.verticalLayout_13.itemAt(i).deleteLater()
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
        for i in range(self.verticalLayout_13.count()):
            pos=[]
            if id(self.verticalLayout_13.itemAt(i))==del_id or (i==self.verticalLayout_13.count()-1 and del_id!=-1):
                continue
            for j in range(self.verticalLayout_13.itemAt(i).count()):
                if isinstance(self.verticalLayout_13.itemAt(i).itemAt(j).widget(), QtWidgets.QDoubleSpinBox):
                    spin_value=self.verticalLayout_13.itemAt(i).itemAt(j).widget().value()
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
                                                                      envdata.camera_matrix, envdata.dist_coeffs,
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

        FigureLayout = QtWidgets.QGridLayout(self.groupBox_7)

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
        


    def init_groupbox2_VLPres(self):
        self.groupBox_2.setTitle("VLP-choice")
        self.groupBox_2.setLayout(self.gridLayout_2)
        self.groupBox_2.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.gridLayout_2.setContentsMargins(150, 5, 150, 5)  # 设置VLP-choice左右的padding

        self.comboBox.addItems(["YOLOv5","YOLOv8"])
        self.comboBox.setCurrentIndex(0)
        self.comboBox_2.addItems(["SQPnP", "EPnP","P3P","PnP2"])
        self.comboBox_2.setCurrentIndex(0)
        self.toolButton.clicked.connect(self.btn5)
        # 设置label大小固定
        self.label_3.setFixedSize(40,40)
        self.label_7.setFixedSize(40, 40)
        self.label_8.setFixedSize(40, 35)

    def init_groupbox3_VLPres(self):
        self.groupBox_3.setTitle("VLP-choice")
        self.groupBox_3.setLayout(self.gridLayout_3)
        self.groupBox_3.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.gridLayout_3.setContentsMargins(150, 5, 150, 5)  # 设置VLP-choice左右的padding

        self.comboBox_4.addItems(["YOLOv5","YOLOv8"])
        self.comboBox.setCurrentIndex(0)
        self.comboBox_3.addItems(["SQPnP", "EPnP","P3P","PnP2"])
        self.comboBox_3.setCurrentIndex(0)
        self.muti_img_dir = ""
        self.toolButton_2.clicked.connect(self.btn6)
        # 设置label大小固定
        self.label_11.setFixedSize(40, 40)
        self.label_9.setFixedSize(40, 40)
        self.label_10.setFixedSize(40, 35)


    def slot_init(self):
        """
        按钮的插槽函数
        :return:
        """
        self.pushButton.clicked.connect(self.btn1)
        self.pushButton_2.clicked.connect(self.btn2)



    def btn1(self):
        #切换到page1
        print("page 1")
        self.stackedWidget.setCurrentIndex(0)
    def btn2(self):
        # 切换到page1
        print("page 2")
        self.stackedWidget.setCurrentIndex(1)

    def single_detected(self):
        print("开始对选中图片执行VLP程序")
        # if self.img_filepath!="":
        self.start_loading()
        script_path =r"E:\Code\Py_pro\YOLOV5\yolov5-master\detect.py"
        script_args = ['--weights', r'E:\Code\Py_pro\YOLOV5\yolov5-master\runs\train\exp2\weights\best.pt',
                       '--source', self.img_filepath,
                       '--data',r'E:\Code\Py_pro\YOLOV5\yolov5-master\data\VLP.yaml',
                       '--imgsz','1632',
                       '--save-txt','',
                       '--save-csv','',
                       '--save-conf','']

        self.vlp_thread = VLPWorkThread(['python', script_path] + script_args)# 线程执行延时任务
        self.vlp_thread.start()
        self.vlp_thread.signals.connect(self.resolve_oneimg_vlpsignals)  # 信号连接槽函数
        # else: # 弹窗警告 请选择待检测的文件
        #     reply = QMessageBox.warning(self,"警告","请选择待检测的文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)

    def start_loading(self):
        # label_loading gif图像开始
        movie_loading = QtGui.QMovie("../UI/image/loading.gif")  # 初始化loading
        self.label_loading.setMovie(movie_loading)  # loading出现
        self.label_loading.raise_()
        movie_loading.start()
    def stop_loading(self):
        # gif图像停止
        self.label_loading.setMovie(QtGui.QMovie(""))  # loading 消失
        self.label_loading.lower()

    def resolve_oneimg_vlpsignals(self, signals):
        # VLP程序执行结束之后的响应
        print(signals)
        self.stop_loading()
        tiff_path = self.vlp_thread.detect_newestdir + '\\' + self.vlp_thread.entityvlp.detectedres_filename[0] + '.tiff'
        self.set_show_inputimg(tiff_path)
        # 因为只有一个值，所以直接使用
        txt_path=self.vlp_thread.detect_newestdir + '\\' + "labels"+ '\\' +self.vlp_thread.entityvlp.detectedres_filename[0] +'.txt'
        with open(txt_path, "r",encoding='UTF-8') as f:
            txt_content = f.read()
        key=list(self.vlp_thread.entityvlp.pnpres.keys())[0]
        pnp_camerapos=self.vlp_thread.entityvlp.pnpres[key][0]
        txt_labelres="Detected txt:\n" + txt_content \
                     +"Camera Postion:\n" + str(pnp_camerapos) + "\n" \
                     +"Time:\n" +f"{self.vlp_thread.entityvlp.thread_time * 1000:.2f}" + " ms"

        self.label_result.setText(txt_labelres)  # 更新检测结果和matlab图片\

        plot_graph(self.groupBox,self.vlp_thread.entityvlp, 400)  # 绘制matlab图像



    def btn5(self):
        print("选取文件")
        self.img_filepath, filetype = QFileDialog.getOpenFileName(self,
                                                      "选取文件",
                                                      "./",
                                                      "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(self.img_filepath)
        self.set_show_inputimg(self.img_filepath)

    def set_show_inputimg(self, imgpath):
        # 显示imgpath下待检测的图片或者检测结果
        self.label_input_img.setVisible(True)
        pixmap=QPixmap(imgpath)
        scaledPixmap = pixmap.scaled(QSize(300, 200), aspectRatioMode=Qt.KeepAspectRatio)
        self.label_input_img.setPixmap(scaledPixmap)
        self.label_input_img.setAlignment(Qt.AlignCenter)# 设置居中显示

    def btn6(self):
        print("选取文件夹")
        self.muti_img_dir = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")
        print(self.muti_img_dir)

    def multi_detected(self):
        print("开始批量执行VLP程序")
        # if self.img_filepath!="":
        self.start_loading()
        script_path = r"E:\Code\Py_pro\YOLOV5\yolov5-master\detect.py"
        script_args = ['--weights', r'E:\Code\Py_pro\YOLOV5\yolov5-master\runs\train\exp2\weights\best.pt',
                       '--source', self.muti_img_dir,
                       '--data', r'E:\Code\Py_pro\YOLOV5\yolov5-master\data\VLP.yaml',
                       '--imgsz', '1632',
                       '--save-txt', '',
                       '--save-csv', '',
                       '--save-conf', '']

        self.vlp_thread_mutil = VLPWorkThread(['python', script_path] + script_args)  # 线程执行延时任务
        self.vlp_thread_mutil.start()
        self.vlp_thread_mutil.signals.connect(self.resolve_mutiimg_vlpsignals)  # 信号连接槽函数
        # else: # 弹窗警告 请选择待检测的文件
        #     reply = QMessageBox.warning(self,"警告","请选择待检测的文件",QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)

    def resolve_mutiimg_vlpsignals(self,signals):
        print(signals)
        self.stop_loading()

        plot_mutil_graph(self.groupBox_4, self.vlp_thread_mutil.entityvlp, 400)  # 绘制matlab图像



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 创建应用程序
    mainwindow = Main_Window()  # 创建主窗口

    apply_stylesheet(app, theme='light_cyan_500.xml')

    mainwindow.show()  # 显示窗口
    sys.exit(app.exec_())  # 程序执行循环
    # import UI.image.res