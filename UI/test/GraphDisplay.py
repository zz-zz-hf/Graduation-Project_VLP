from DataDisplayUI import Ui_MainWindow
from PyQt5.QtWidgets import QApplication,QMainWindow,QGridLayout
from PyQt5.QtCore import QTimer
import sys,time
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cbook as cbook


class Figure_Canvas(FigureCanvas):
    def __init__(self,parent=None,width=3.9,height=2.7,dpi=100):
        self.fig=Figure(figsize=(width,height),dpi=100)
        super(Figure_Canvas,self).__init__(self.fig)
        self.ax=self.fig.add_subplot(111)
    def test(self):
        x=[1,2,3,4,5,6,7]
        y=[2,1,3,5,6,4,3]
        self.ax.plot(x,y)

class ImgDisp(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(ImgDisp,self).__init__(parent)
        self.setupUi(self)
        self.Init_Widgets()
        self.timer=QTimer()
        self.timer.start(1)
        self.ts=time.time()
        self.timer.timeout.connect(self.UpdateImgs)
    def Init_Widgets(self):
        self.PrepareSamples()
        self.PrepareLineCanvas()
        self.PrepareBarCanvas()
        self.PrepareImgCanvas()
        self.PrepareSurfaceCanvas()
    def PrepareSamples(self):
        self.x = np.arange(-4, 4, 0.02)
        self.y = np.arange(-4, 4, 0.02)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = np.sin(self.x)
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.Z = np.sin(self.R)
    def PrepareLineCanvas(self):
        self.LineFigure = Figure_Canvas()
        self.LineFigureLayout = QGridLayout(self.LineDisplayGB)
        self.LineFigureLayout.addWidget(self.LineFigure)
        self.LineFigure.ax.set_xlim(-4, 4)
        self.LineFigure.ax.set_ylim(-1, 1)
        self.line = Line2D(self.x, self.z)
        self.LineFigure.ax.add_line(self.line)
    def PrepareBarCanvas(self):
        self.BarFigure = Figure_Canvas()
        self.BarFigureLayout = QGridLayout(self.BarDisplayGB)
        self.BarFigureLayout.addWidget(self.BarFigure)
        self.BarFigure.ax.set_xlim(-4, 4)
        self.BarFigure.ax.set_ylim(-1, 1)
        self.bar = self.BarFigure.ax.bar(np.arange(-4, 4, 0.5), np.sin(np.arange(-4, 4, 0.5)), width=0.4)
        self.patches = self.bar.patches
    def PrepareImgCanvas(self):
        self.ImgFigure = Figure_Canvas()
        self.ImgFigureLayout = QGridLayout(self.ImageDisplayGB)
        self.ImgFigureLayout.addWidget(self.ImgFigure)
        self.ImgFig = self.ImgFigure.ax.imshow(self.Z, cmap='bone')
        self.ImgFig.set_clim(-0.8,0.8)
    def PrepareSurfaceCanvas(self):
        self.SurfFigure = Figure_Canvas()
        self.SurfFigureLayout = QGridLayout(self.SurfaceDisplayGB)
        self.SurfFigureLayout.addWidget(self.SurfFigure)
        self.SurfFigure.ax.remove()
        self.ax3d = self.SurfFigure.fig.add_subplot(projection='3d')
        self.Surf = self.ax3d.plot_surface(self.X, self.Y, self.Z, cmap='rainbow')
    def UpdateImgs(self):
        dt=time.time()-self.ts
        self.LineUpdate(dt)
        self.BarUpdate(dt)
        self.ImgUpdate(dt)
        self.SurfUpdate(dt)
    def LineUpdate(self,dt):
        z=np.sin(self.x+dt)
        self.line.set_ydata(z)
        self.LineFigure.draw()
    def BarUpdate(self,dt):
        x=np.sin(np.arange(-4,4,0.5)+dt)
        for i in range(len(self.patches)):
            self.patches[i].set_height(x[i])
        self.bar.patches=self.patches
        self.BarFigure.draw()
    def ImgUpdate(self,dt):
        X=self.X+dt
        Y=self.Y+dt
        R=np.sqrt(X**2+Y**2)
        Z=np.sin(R)
        self.ImgFig.set_data(Z)
        self.ImgFigure.draw()
    def SurfUpdate(self,dt):
        X = self.X + dt
        Y = self.Y + dt
        R = np.sqrt(X ** 2 + Y ** 2)
        Z = np.sin(R)
        polys=self.Get3dVerts(self.X,self.Y,Z)
        self.Surf.set_verts(polys)
        self.SurfFigure.draw()
    def Get3dVerts(self,X,Y,Z):
        if Z.ndim != 2:
            raise ValueError("Argument Z must be 2-dimensional.")
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        rows, cols = Z.shape
        rcount = 50
        ccount = 50
        rstride = int(max(np.ceil(rows / rcount), 1))
        cstride = int(max(np.ceil(cols / ccount), 1))
        # evenly spaced, and including both endpoints
        row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
        col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]
        polys = []
        for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
            for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
                ps = [
                    # +1 ensures we share edges between polygons
                    cbook._array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1])
                    for a in (X, Y, Z)
                ]
                # ps = np.stack(ps, axis=-1)
                ps = np.array(ps).T
                polys.append(ps)
        return polys

if __name__=='__main__':
    app=QApplication(sys.argv)
    ui=ImgDisp()
    ui.show()
    sys.exit(app.exec_())

