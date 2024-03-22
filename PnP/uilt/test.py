import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

led3d=np.array([[2.713,0.485,2.710],[2.713,2.081,2.710],[0.330,0.485,2.710],[0.335,2.080,2.710],
                [ 2.908,2.085,2.710],[2.908,0.492,2.710],[1.523,0.485,2.710],[1.528,2.087,2.710]])

colors = [
    '#FF0000',  # 红色
    '#FFA500',  # 橙色
    '#FFFF00',  # 黄色
    '#FF4500',  # 橙红色
    '#556B2F', # 深橄榄绿色
    '#FF69B4',  # 热粉红色
    '#FFC0CB',  # 浅粉红色
    '#FFD700',  # 金色
    '#DAA520',  # 深金色
    '#20B2AA',  # 浅海绿色
    '#FF7F50',  # 珊瑚色
    '#9400D3',  # 紫色
    '#4B0082',  # 靛蓝色
    '#0000FF',  # 纯蓝色
    '#7CFC00'  # 草坪绿
]


def my_world_to_pixel(word_point, R, t, K):
    """
    :param word_point: (3*1)的ndarray格式数据
    :param R:(3*3)
    :param t:(3*1)
    :param K:(3*1)
    :return: (u,v)
    """
    # 使用齐次坐标计算
    word_point_nor=np.row_stack((word_point,np.array([[1]])))
    # 获得外部参数矩阵[[R,t],[0,1]]
    out_matrix=np.eye(4)
    out_matrix[:3, :3] = R
    out_matrix[:3,3]=t.flatten()
    # 获得内参矩阵的齐次
    in_matrix=np.eye(4)
    in_matrix=np.delete(in_matrix,3,0)
    in_matrix[:3, :3]=K

    camera_point=np.dot(out_matrix,word_point_nor) # 4*1

    pixel_point_nor=np.dot(in_matrix,camera_point) # 3*1
    pixel_point = pixel_point_nor[:2] / pixel_point_nor[2]

    return pixel_point


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

def tran_oula(rvec_matrix, translation_vector):
    """
    旋转矩阵转换为欧拉角
    :param rvec_matrix:
    :param translation_vector:
    :return:
    """
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    # 使用 OpenCV 的 decomposeProjectionMatrix 函数分解投影矩阵
    camera_,R,t,x_matrix,y_matrix,z_matrix,eulerAngles = cv.decomposeProjectionMatrix(proj_matrix)
    # 将欧拉角从弧度转换为角度
    pitch, roll,yaw = [_ for _ in eulerAngles]

    return pitch,roll,yaw

def genatate_camerapoints(imgsize,R,t):
    """
    生成仿真图像坐标
    :param R: 旋转矩阵(3,3)
    :param t: 平移矩阵(3,1)
    :param imgsize (w,h)
    :return:在imgsize范围内的led像素坐标
    """
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

def test_example():
    # 生成矩阵R,t
    R_gen = euler_to_rotation_matrix(30, 20, 10)
    t_gen = np.array([[-0.5, -0.2, -1.0]], dtype=np.double)
    world_points = np.array([[0.335, 2.080, 2.710],
                             [1.523, 0.485, 2.710],
                             [1.528, 2.087, 2.710],
                             [2.713, 0.485, 2.710]], dtype=np.double)  # 6_3 3 6 7 0

    imgpts, jac = cv.projectPoints(world_points, R_gen, t_gen, camera_matrix, dist_coeffs)  # 实际世界坐标转化为像素坐标
    (success, rpnp_vectorc, t_pnpc) = cv.solvePnP(world_points, imgpts, camera_matrix, dist_coeffs,
                                                  flags=cv.SOLVEPNP_SQPNP)  # PnP求解

    img_points = []
    for i in range(world_points.shape[0]):
        pixel_point = my_world_to_pixel(world_points[i].reshape(-1, 1), R_gen, t_gen, camera_matrix)
        img_points.append(pixel_point)
    img_points = np.array(img_points).reshape(-1, 2)
    (success, rpnp_vector, t_pnp) = cv.solvePnP(world_points, img_points, camera_matrix, dist_coeffs,
                                                flags=cv.SOLVEPNP_SQPNP)

    R_pnpc = np.array(cv.Rodrigues(rpnp_vectorc)[0])  # 将旋转向量转化为旋转矩阵
    R_pnp = np.array(cv.Rodrigues(rpnp_vector)[0])  # 将旋转向量转化为旋转矩阵

    # pitch, roll, yaw = tran_oula(R_pnp, t_pnp)  # 无用，旋转矩阵的分解并不是唯一的
    pos = np.dot(-np.linalg.inv(R_pnp), t_pnp)

    posc = np.dot(-np.linalg.inv(R_pnpc), t_pnpc)

    pos_standard = np.dot(-np.linalg.inv(R_gen), t_gen.reshape(3, 1))

    print(t_gen, R_gen)
    print(t_pnp.T, R_pnp)
    print(t_pnpc.T, R_pnpc)
def draw_allpic(data,is2d,is3d):
    """
    绘制相机坐标的2d和3d的坐标图
    :param data: {standard:[],calculate:[[pos1,pos2],……]}
    :param is2d: 是否绘制2d图
    :param is3d: 是否绘制3d图
    :return:
    """
    std_poss=data["standard"]
    cal_poss=data["calculate"]
    # 创建一个新的图形
    fig =plt.figure()
    if is2d==1 and is3d==1:
        plt.subplot(1,2,1)
        for i in range(len(std_poss)): #array(3,1)
            for cla_pos in cal_poss[i]:
                plt.scatter(cla_pos[0][0], cla_pos[1][0], color=colors[i], label='calculate',marker='x')
            plt.scatter(std_poss[i][0][0], std_poss[i][1][0], color=colors[i], label='Standard')
        plt.xlim(-0.5,4.2)
        plt.ylim(-0.5, 2.7)
        plt.title('Datasets in One Plot')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        # plt.subplot(1, 2, 2)
        ax = fig.add_subplot(122, projection='3d')  # 添加一个3D坐标轴
        # 绘制散点图
        for i in range(len(std_poss)): #array(3,1)
            for cla_pos in cal_poss[i]:
                ax.scatter(cla_pos[0][0], cla_pos[1][0],cla_pos[2][0], color=colors[i], label='calculate',marker='x')
            ax.scatter(std_poss[i][0][0], std_poss[i][1][0], std_poss[i][2][0], color=colors[i], label='Standard')
        ax.set_xlim(-0.5, 4.2)
        ax.set_ylim(-0.5, 2.7)
        ax.set_zlim(0,2.7)
        ax.set_title('3D Scatter Plot')  # 设置标题
        ax.set_xlabel('X Axis')  # 设置X轴标签
        ax.set_ylabel('Y Axis')  # 设置Y轴标签
        ax.set_zlabel('Z Axis')  # 设置Z轴标签

    plt.show()


    # if is3d:
    # if is2d==1:

camera_matrix = np.array(
    [[1288.6255, 0, 813.2959],
     [0,  1290.6448, 819.7536],
     [0, 0, 1]], dtype=np.double
)
# 设置相机的畸变矩阵
dist_coeffs = np.array([0.2172,-0.6233,-0.0008,-0.0004,0.5242],dtype=np.double)  #(k1,k2,p1,p3,k3)
# dist_coeffs = np.array([0,0,0,0,0],dtype=np.double)  #(k1,k2,p1,p3,k3)

if __name__ == '__main__':
    """
    旋转矩阵欧拉角：x<40 y<25 和平移矩阵的大小相关，平移矩阵越小，可调范围越大
    平移矩阵：(x,y,z) x+y+z<2
    1.2 1 0;1.6 0.5 0;1.8 0.25 0
    1 0.5 0.5;1.2 0.25 0.5
    0.2 0.5 1;0.6 0.25 1
    """
    pix_width, pix_height = 3264, 2464
    poss_standard=[]
    poss_pre=[]
    for j in range(1,3):
        tmpposs_standard = []
        tmpposs_pre = []
        for i in range(20):
            # 生成矩阵R,t
            R_gen=euler_to_rotation_matrix(0,0,5 + i * 5)
            t_gen=np.array([[-0.2-j*0.5,-0.2,-0]], dtype=np.double)
            pos_standard=np.dot(-np.linalg.inv(R_gen), t_gen.reshape(3, 1))

            pixel_point_standard,world_point_standard=genatate_camerapoints((pix_width, pix_height), R_gen, t_gen)

            if pixel_point_standard.shape[0]<3:
                print("控制点数量不足")
                continue
            (success, rpnp_vectorc, t_pnpc) = cv.solvePnP(world_point_standard, pixel_point_standard, camera_matrix, dist_coeffs,flags=cv.SOLVEPNP_ITERATIVE)  # PnP求解
            R_pnpc = np.array(cv.Rodrigues(rpnp_vectorc)[0])  # 将旋转向量转化为旋转矩阵
            pos_pnp = np.dot(-np.linalg.inv(R_pnpc), t_pnpc) #(3,1)
            print(-t_gen,-t_pnpc.T)
            print("====================< "+str(j)+" > "+" < "+str(i)+" >")
            tmpposs_pre.append(-t_pnpc)
            tmpposs_standard.append(-t_gen.T)
        poss_standard.append(np.mean(tmpposs_standard, axis=0))
        poss_pre.append(tmpposs_pre)

    draw_allpic({"standard":poss_standard,"calculate":poss_pre},1,1)

    print("pixel1")











