import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import *



led3d=np.array([[2.713,0.485,2.710],[2.713,2.081,2.710],[0.330,0.485,2.710],[0.335,2.080,2.710],
                [ 2.908,2.085,2.710],[2.908,0.492,2.710],[1.523,0.485,2.710],[1.528,2.087,2.710]])
H=2710
focal_length=3.04 #焦距
dist_coeffs = np.array([0.2172,-0.6233,-0.0008,-0.0004,0.5242],dtype=np.double)  #(k1,k2,p1,p3,k3)
camera_matrix = np.array(
    [[1288.6255/focal_length, 0, 813.2959],
     [0,  1290.6448/focal_length, 819.7536],
     [0, 0, 1]], dtype=np.double
) #相机的内参矩阵 但是f/dx更改为1/dx

camera_matrix_ = np.array(
    [[1288.6255, 0, 813.2959],
     [0,  1290.6448, 819.7536],
     [0, 0, 1]], dtype=np.double
)
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

def gen_projection_cpoints(imgsize,w_poss,R,t):
    """
    产生LED投影在相机坐标下对应的坐标
    :param w_poss: leds的世界坐标 一行是一个坐标
    :param R: 旋转矩阵
    :param t: 平移矩阵
    :return: w_poss在相机坐标系下的坐标 一行是一个坐标
    """
    pixel_points = None
    world_3d = None
    # 获得像素坐标imgpts(8,1,2)
    imgpts, jac = cv.projectPoints(w_poss, R, t, camera_matrix_, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2)
    isnone = -1
    for i in range(imgpts.shape[0]):
        if imgpts[i, 0] > imgsize[0] or imgpts[i, 1] > imgsize[1] or imgpts[i, 0] < 0 or imgpts[i, 1] < 0:
            pixel_points = np.delete(imgpts, i, 0)
            world_3d = np.delete(led3d, i, 0)
            isnone = 1
    if isnone == -1:
        world_3d = led3d
        pixel_points = imgpts

    # 将imgpts转换为(8,1,3)
    u_v = np.ones((pixel_points.shape[0],3))
    u_v[:, :2] = pixel_points
    camera_matrix_inv=np.linalg.inv(camera_matrix)

    return np.dot(camera_matrix_inv,u_v.T).T,world_3d

def cal_cameradistance(camera_poss):
    """
    计算相机坐标下的距离
    :param camera_poss:每一行是一个坐标点
    :return: 距离的集合数组
    """
    distances=[]
    for row in camera_poss:
        x=row[0]
        y=row[1]
        dis=math.sqrt(x*x+y*y)
        distances.append(dis)
    return distances

def cal_camerapos(w_poss,w_diss):
    """
    计算相机的位置（解二元二次方程，至少三个点确定一个位置）
    :param w_poss: 世界坐标下的led灯位置
    :param w_diss: 每一个led到达相机投影的距离
    :return: 相机位置
    """
    c_circle0=w_poss[0]
    c_circle1 = w_poss[1]
    c_circle2 = w_poss[2]
    r0=w_diss[0]
    r1 = w_diss[1]
    r2 = w_diss[2]

    # 解是纯虚数的情况需要优化修正
    x = Symbol('x')
    y = Symbol('y')
    solved_value = solve([(x-c_circle0[0])**2+(y-c_circle0[1])**2-r0**2,
                          (x-c_circle1[0])**2+(y-c_circle1[1])**2-r1**2,
                          ], [x, y])
    # 距离第三个圆越近越接近最优解

    x1,y1=solved_value[0][0],solved_value[0][1]
    x2, y2 = solved_value[1][0], solved_value[1][1]
    err1=(x1 - c_circle2[0]) ** 2 + (y1 - c_circle2[1]) ** 2 - r2 ** 2
    err2 = (x2 - c_circle2[0]) ** 2 + (y2 - c_circle2[1]) ** 2 - r2 ** 2
    res=[x2,y2]
    if abs(err1)<abs(err2):
        res=[x1,y1]
    res=[int(item) / 1000 for item in res]
    return res



if __name__ == '__main__':
    R_gen = euler_to_rotation_matrix(0, 0, 0)

    for i in np.linspace(1,4,31):
        for j in np.linspace(0,2,21):

            t_gen = np.array([[-i, -j, -0]], dtype=np.double) # 相机坐标
            c_poss,w_poss=gen_projection_cpoints((3264, 2464),led3d,R_gen,t_gen)
            camera_dis=cal_cameradistance(c_poss)
            world_dis=[dis*H/focal_length for dis in camera_dis]
            camera_pos=cal_camerapos(w_poss*1000,world_dis)
            print(-t_gen,camera_pos)





