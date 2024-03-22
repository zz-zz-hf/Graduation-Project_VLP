from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

import UI.uilt.EnvData as envdata
import numpy as np
from scipy import stats

#matlab画图对象
class Figure_Canvas(FigureCanvas):
    def __init__(self,parent=None,width=3.9,height=2.7,dpi=100):
        self.fig=Figure(figsize=(width,height),dpi=100)
        super(Figure_Canvas,self).__init__(self.fig)


def plot_graph(parents,vlpentity,min_width):
    """
    绘制matlab图像
    :param parents:图像绘制的组件
    :param min_width: 图像的最小尺寸
    :return:
    """
    LineFigure = Figure_Canvas()
    SurfFigure = Figure_Canvas()

    FigureLayout = QtWidgets.QGridLayout(parents)
    FigureLayout.addWidget(SurfFigure, 0, 0, 1, 1)
    FigureLayout.addWidget(LineFigure, 1, 0, 1, 1)
    FigureLayout.setColumnMinimumWidth(0, min_width)
    FigureLayout.setRowMinimumHeight(0, min_width)
    FigureLayout.setRowMinimumHeight(1, min_width)
    # 3D 图像 绘制led灯和standard_pos和cal_pos
    ax3d = SurfFigure.fig.add_subplot(projection='3d')
    x = [row[0] for row in envdata.LED_3D]
    y = [row[1] for row in envdata.LED_3D]
    z = [row[2] for row in envdata.LED_3D]
    ax3d.scatter(x, y, z, label='LED', c='k')  # LED
    standard3d=[]
    pnp3d=[]
    for pos_number in list(vlpentity.pnpres.keys()):
        standard3d.append(envdata.Standard_Camera[pos_number])
        for posarray in vlpentity.pnpres[pos_number]:# [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            pnp3d.append(posarray.tolist()[0])
    x_pnp=[row[0] for row in pnp3d]
    y_pnp = [row[1] for row in pnp3d]
    z_pnp = [row[2] for row in pnp3d]
    ax3d.scatter(x_pnp, y_pnp,z_pnp,label='PnP calculate', c='r', marker='x',alpha=0.4)  # pnp计算得到的点
    x_standard=[row[0] for row in standard3d]
    y_standard = [row[1] for row in standard3d]
    z_standard = [row[2] for row in standard3d]
    ax3d.scatter(x_standard,y_standard,z_standard,label='Standard', c='b', marker='*')  # 标准点
    ax3d.set_xlim(0, 4.2)
    ax3d.set_ylim(0, 2.7)
    ax3d.set_zlim(0, 2.7)
    ax3d.legend(loc='right', bbox_to_anchor=(1.04, 1))  # 添加图例
    # ax3d.title('3D')

    # 2D 点图
    ax = LineFigure.fig.add_subplot(111)
    standard2d = []
    pnp2d = []
    for pos_number in list(vlpentity.pnpres.keys()):
        standard2d.append(envdata.Standard_Camera[pos_number])
        for posarray in vlpentity.pnpres[pos_number]:  # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            pnp2d.append(posarray.tolist()[0])

    x_pnp2 = [row[0] for row in pnp2d]
    y_pnp2 = [row[1] for row in pnp2d]
    ax.scatter(x_pnp2, y_pnp2, label='PnP calculate', c='r', marker='x')  # pnp计算得到的点
    x_standard2 = [row[0] for row in standard2d]
    y_standard2 = [row[1] for row in standard2d]
    ax.scatter(x_standard2, y_standard2, label='Standard', c='b', marker='*')  # 标准点

    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 2.7)
    ax.legend(loc='right', bbox_to_anchor=(1.04, 1))
    # ax.title('2D')


def plot_mutil_graph(parents,vlpentity,min_width):
    """
    绘制matlab图像
    :param parents:图像绘制的组件
    :param min_width: 图像的最小尺寸
    :return:
    """
    LineFigure = Figure_Canvas()
    SurfFigure = Figure_Canvas()
    CDFFigure = Figure_Canvas()
    HistogramFigure = Figure_Canvas()

    FigureLayout = QtWidgets.QGridLayout(parents)
    FigureLayout.addWidget(SurfFigure, 0, 0, 1, 1)
    FigureLayout.addWidget(LineFigure, 1, 0, 1, 1)
    FigureLayout.addWidget(CDFFigure, 2, 0, 1, 1)
    FigureLayout.addWidget(HistogramFigure, 3, 0, 1, 1)

    FigureLayout.setColumnMinimumWidth(0, min_width)
    FigureLayout.setRowMinimumHeight(0, min_width)
    FigureLayout.setRowMinimumHeight(1, min_width)
    FigureLayout.setRowMinimumHeight(2, min_width)
    FigureLayout.setRowMinimumHeight(3, min_width)
    # 3D 图像 绘制led灯和standard_pos和cal_pos
    ax3d = SurfFigure.fig.add_subplot(projection='3d')
    x_cdf = [row[0] for row in envdata.LED_3D]
    y = [row[1] for row in envdata.LED_3D]
    z = [row[2] for row in envdata.LED_3D]
    ax3d.scatter(x_cdf, y, z, label='LED', c='k')  # LED
    standard3d=[]
    index_pnp=0# 只为第一次设置label
    for pos_number in list(vlpentity.pnpres.keys()):
        standard3d.append(envdata.Standard_Camera[pos_number])
        for posarray in vlpentity.pnpres[pos_number]:# [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            if index_pnp==0:
                ax3d.scatter(posarray[:,0], posarray[:,1], posarray[:,2],
                             label='PnP calculate', c=envdata.colors[pos_number], marker='x', alpha=0.4)  # pnp计算得到的点
            else:
                ax3d.scatter(posarray[:, 0], posarray[:, 1], posarray[:, 2],
                             label=None, c=envdata.colors[pos_number], marker='x', alpha=0.4)
            index_pnp=index_pnp+1
    x_standard = [row[0] for row in standard3d]
    y_standard = [row[1] for row in standard3d]
    z_standard = [row[2] for row in standard3d]
    ax3d.scatter(x_standard, y_standard, z_standard, label='Standard', c='b', marker='*', zorder=10000)  # 标准点

    ax3d.set_xlim(0, 4.2)
    ax3d.set_ylim(0, 2.7)
    ax3d.set_zlim(0, 2.7)
    ax3d.legend(loc='right', bbox_to_anchor=(1.04, 1))  # 添加图例
    # ax3d.title('3D')

    # 2D 点图
    ax = LineFigure.fig.add_subplot(111)
    standard2d = []
    index_pnp = 0# 只为第一次设置label
    for pos_number in list(vlpentity.pnpres.keys()):
        standard2d.append(envdata.Standard_Camera[pos_number])
        for posarray in vlpentity.pnpres[pos_number]:  # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            if index_pnp == 0:
                ax.scatter(posarray[:, 0], posarray[:, 1],
                             label='PnP calculate', c=envdata.colors[pos_number], marker='x', alpha=0.4)  # pnp计算得到的点
            else:
                ax.scatter(posarray[:, 0], posarray[:, 1],
                             label=None, c=envdata.colors[pos_number], marker='x', alpha=0.4)
            index_pnp = index_pnp + 1
    x_standard2 = [row[0] for row in standard2d]
    y_standard2 = [row[1] for row in standard2d]
    ax.scatter(x_standard2, y_standard2, label='Standard', c='b', marker='*')  # 标准点

    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, 2.7)
    ax.legend(loc='right', bbox_to_anchor=(1.04, 1))
    # ax.title('2D')

    # 绘制CDF图和频率分布图
    errors=[]
    for pos_number in list(vlpentity.pnpres.keys()):
        standard_camera=np.array(envdata.Standard_Camera[pos_number]).reshape(1,3)
        for cla_camera in vlpentity.pnpres[pos_number]: # [array([[ 2.84641251,  0.96031783, -0.01636136]]),……]
            errors.append(float(cal_err(cla_camera, standard_camera)))

    ax_cdf = CDFFigure.fig.add_subplot(111)
    res = stats.relfreq(errors, numbins=25)  # 给定数据集的相对频率分布
    x_cdf = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
    y_cdf = np.cumsum(res.frequency)
    ax_cdf.plot(x_cdf, y_cdf, label='CDF')
    ax_cdf.legend()

    ax_his = HistogramFigure.fig.add_subplot(111)
    ax_his.bar(x_cdf, res.frequency, width=res.binsize,label='Histogram')
    ax_his.legend()


def cal_err(pnp_pos, standard_pos):
    err_np = np.abs(pnp_pos - standard_pos)
    err = np.sum(err_np) / 3
    return f"{err:.3f}"












