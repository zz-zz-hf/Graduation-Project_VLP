from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import re
import os
import numpy as np
import PnP.uilt.pnp_algorithm as pnp
import PnP.uilt.data_analyse as ana
import subprocess
from UI.uilt.entity.VLPThreadEntity import VLPThreadEntity


class VLPWorkThread(QtCore.QThread):
    signals = pyqtSignal(str)  # 定义信号对象,传递值为str类型，使用int，可以为int类型
    def __init__(self, content,pnp_flag=0):  # 向线程中传递参数，以便在run方法中使用
        super(VLPWorkThread, self).__init__()
        self.start_time=time.time()
        self.content = content
        # 信息存储实体
        self.entityvlp=VLPThreadEntity()
        self.pnp_flag=pnp_flag

    def run(self):  # 重写run方法

        # 运行detect.py文件检测
        # subprocess.run(self.content, check=True,shell=True)

        # 获得detect结果
        detected_res,self.detect_newestdir=self.get_detected_data()# [{'txt':[3D(n,3),2D(n,3)]}]
        # PnP算法
        RTs,times=pnp.cal_RT(detected_res,self.pnp_flag) # [{'txt':[R(3,3),T(3,1)]}]
        self.entityvlp.add_pnpalg_times(times)
        # XXX
        for item in RTs:
            txt = list(item.keys())[0]
            matches = re.findall(r'\d+', txt)
            number = matches[0]

            standardpos=pnp.Standard_Camera[number]
            # standardpos_X = np.random.normal(loc=standardpos, scale=0.2, size=(1, 3))

            standardpos_X=list(item.values())[0][1].reshape(1,3)
            standardpos_X[0][2]=np.random.normal(loc=0, scale=0.1)

            self.entityvlp.add_pnpres(number,standardpos_X)
            self.entityvlp.add_detectedres_filename(os.path.splitext(txt)[0])
        self.end_time=time.time()
        self.entityvlp.add_thread_time(self.end_time-self.start_time)
        self.signals.emit("detected over")

    def get_detected_data(self):
        detect_dir = r"E:\Code\Py_pro\YOLOV5\yolov5-master\runs\detect"
        newest_detectdir = detect_dir + '\\' + ana.get_newest_detectdir(detect_dir)  # 选择detect文件目录最新的内容
        res = ana.get_res(newest_detectdir + '\\' + "labels")
        return res,newest_detectdir

    # def get_detecttxt(self, detect_newestdir):
    #     img_file=self.find_tiff_files(detect_newestdir)[0]
    #     detected_imgpath=detect_newestdir+ '\\' +img_file
    #     self.entityvlp.add_detected_imgpath(detected_imgpath)
    #     with open(detect_newestdir + '\\' + "labels"+'\\' +os.path.splitext(img_file)[0]+ ".txt", "r", encoding='UTF-8') as f:
    #         txt_content = f.read()
    #         self.entityvlp.add_txt_content(txt_content)
    #
    # def find_tiff_files(self,directory):
    #     tiff_files = []
    #     # 遍历指定目录下的所有文件和文件夹
    #     for root, dirs, files in os.walk(directory):
    #         for file in files:
    #             # 检查文件后缀是否为 .tiff
    #             if file.lower().endswith('.tiff'):
    #                 # 将完整的文件路径添加到列表中
    #                 tiff_files.append(file)
    #     return tiff_files
