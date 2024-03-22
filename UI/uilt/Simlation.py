import numpy as np
import cv2 as cv
from UI.uilt.EnvData import camera_matrix,dist_coeffs,LED_3D
def euler_to_rotation_matrix(x,y, z):
    """
    根据pitch、roll和yaw生成旋转矩阵的Python函数
    :param x: x轴旋转的角度（°）
    :param y:
    :param z:
    :return: 旋转矩阵(3,3)
    """
    # 将角度转换为弧度
    y = np.radians(y)
    x = np.radians(x)
    z = np.radians(z)
    # 计算旋转矩阵的分量
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(y), 0, -np.sin(y)],
                   [0, 1, 0],
                   [np.sin(y), 0, np.cos(y)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # 组合旋转矩阵
    R = Rz @ Ry @ Rx
    return R


def genatate_camerapoints(imgsize,R,t):
    """
    生成仿真图像坐标
    :param R: 旋转矩阵(3,3)
    :param t: 平移矩阵(3,1)
    :param imgsize (w,h)
    :return:在imgsize范围内的led像素坐标
    """
    led3d=np.array(LED_3D)
    world_3d=None
    pixel_points=None
    my_dist_coeffs=np.array([0,0,0,0,0],dtype=np.double)  #(k1,k2,p1,p3,k3)
    imgpts, jac = cv.projectPoints(led3d, R, t, camera_matrix, dist_coeffs)  # 实际世界坐标转化为像素坐标
    imgpts=imgpts.reshape(-1,2)
    isnone=-1
    for i in range(imgpts.shape[0]):
        if imgpts[i, 0]> imgsize[0] or imgpts[i, 1]> imgsize[1] or imgpts[i, 0]<0 or imgpts[i, 1]<0:
            pixel_points=np.delete(imgpts,i,0)
            world_3d=np.delete(led3d,i,0)
            isnone=1
    # pixel_points=imgpts[(np.logical_and(imgpts[:, 0] < imgsize[0], imgpts[:, 1] < imgsize[1], dtype=bool))]
    # pixel_points=pixel_points[(np.logical_and(pixel_points[:, 0] >0, pixel_points[:, 1] >0, dtype=bool))]
    if isnone==-1:
        world_3d=led3d
        pixel_points=imgpts
    return pixel_points,world_3d