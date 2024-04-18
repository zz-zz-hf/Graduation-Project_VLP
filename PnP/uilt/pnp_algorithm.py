import cv2 as cv
import numpy as np
# import data_analyse as dy
import time

# 相机3D坐标对应关系
# 0 tx11[2.713,0.485,2.710] 2k
# 1 tx12[2.713,2.081,2.710] 3.5k
# 2 tx13[0.330,0.485,2.710] 1k
# 3 tx14[0.335,2.080,2.710] 4.5k
# 4 tx3[ 2.908,2.085,2.710] 3k
# 5 tx4[2.908,0.492,2.710] 2.5k
# 6 tx7[1.523,0.485,2.710] 1.5k
# 7 tx8[1.528,2.087,2.710] 4k

Standard_Camera={"6":[1.623,1.010,0.000],"7":[1.873,1.010,0.000],"8":[2.123,1.010,0.000],"9":[2.373,1.009,0.000],"10":[ 2.623,1.009,0.000],
             "11":[1.623,1.260,0.000],"12":[1.873,1.260,0.000],"13":[2.123,1.260,0.000],"14":[2.373,1.259,0.000],"15":[ 2.623,1.259,0.000 ],
             "16":[1.623,1.510,0.000],"17":[1.873,1.510,0.000],"18":[2.123,1.510,0.000],"19":[2.373,1.509,0.000],"20":[ 2.623,1.009,0.000]}

# 设置相机参数矩阵
camera_matrix = np.array(
    [[1288.6255, 0, 813.2959],
     [0,  1290.6448, 819.7536],
     [0, 0, 1]], dtype=np.double
)
# 设置相机的畸变矩阵
# dist_coeffs = np.array([0.2172,-0.6233,-0.0008,-0.0004,0.5242],dtype=np.double)  #(k1,k2,p1,p3,k3)
dist_coeffs = np.array([0,0,0,0,0],dtype=np.double)  #(k1,k2,p1,p3,k3)

# 6_1 3264*2464
def trans_pix_points(points,w,h):
    """
    :param points: [[center_x,certer_y],……] 归一化之后的值
    :param w: 图片宽的像素数
    :param h: 图片高的像素数
    :return: 解归一化之后的点[[x,y],……]
    """
    p=[]
    for point in points:
        x=point[0]*w
        y=point[1]*h
        p.append([x,y])
    return p

def cal_RT(res,pnp_flags=cv.SOLVEPNP_SQPNP):
    """
    :param res:
    :param pnp_flags:
    :return: [file_name:[R(3,3),T(3,1)]……],times算法执行的时间
    """
    RTs=[]
    times=[]
    for item in res:
        filename = list(item.keys())[0]
        word_points,img_points=del_rep(item[filename][0], item[filename][1])
        # word_points=np.unique(item[filename][0], axis=0)
        # img_points=np.unique(item[filename][1],axis=0)
        start_time = time.time()
        if pnp_flags==8 and len(word_points)>=3:
            (success, rotation_vector, translation_vector) = cv.solvePnP(word_points, img_points, camera_matrix,
                                                                         dist_coeffs, flags=pnp_flags)
        elif pnp_flags==0 and len(word_points)>3:
            (success, rotation_vector, translation_vector) = cv.solvePnP(word_points, img_points, camera_matrix,
                                                                        dist_coeffs, flags=pnp_flags)
        else:
            continue
        end_time = time.time()
        times.append(end_time - start_time)
        R = cv.Rodrigues(rotation_vector)[0] #将旋转向量转化为旋转矩阵
        RTs.append({filename:[R,translation_vector]})
    return RTs,times

def del_rep(word_points,img_points):
    """
    删除wordpoints中的重复值，并对应删除imgpoints中的对应index的值。wordpoints
    :param word_points: array(n,3)
    :param img_points: array(n,2)
    :return:
    """
    norp_wordpoints=[]
    norp_imgpoints=[]
    for index in range(len(word_points)):
        word_row=word_points[index]
        img_row=img_points[index]
        if not any(np.array_equal(word_row, unique_row) for unique_row in norp_wordpoints):
            norp_wordpoints.append(word_row)
            norp_imgpoints.append(img_row)
    return np.array(norp_wordpoints),np.array(norp_imgpoints)

# 不确定是否正确
def cal_camerapos(RT):
    postions=[]
    for item in RT:
        filename = list(item.keys())[0]
        R_matrix=item[filename][0]
        t=item[filename][1]
        pos = np.dot(-np.linalg.inv(R_matrix), t)
        postions.append({filename:pos})
    return postions


# if __name__ == '__main__':
#
#     # pix_width,pix_height=3264, 2464
#     # img_points = [[0.025123,0.853896], [0.637561,0.567370], [0.133885,0.357143],[0.759191,0.066964]] #6_3
#     # img_points = np.array(trans_pix_points(img_points, pix_width,pix_height), dtype=np.double)
#     #
#     # realword_points = np.array([[0.335, 2.080, 2.710],
#     #                             [1.523, 0.485, 2.710],
#     #                             [1.528, 2.087, 2.710],
#     #                             [2.713, 0.485, 2.710]], dtype=np.double) #6_3 3 6 7 0
#     #
#     # (success, rotation_vector, translation_vector) = cv.solvePnP(realword_points, img_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_SQPNP)
#     # # print(success,rotation_vector,translation_vector)
#     # real_P=np.array([[1.623, 1.010, 0.000]], dtype=np.double)
#     # imgpts, jac = cv.projectPoints(real_P, rotation_vector, translation_vector, camera_matrix, dist_coeffs)  #实际相机位置转化为像素坐标 [[[1157.15147894 1092.05856056]]]
#     # # imgpts, jac = cv.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)  # 将三维坐标转化到二维存在一定的误差
#     # R = cv.Rodrigues(rotation_vector)[0] #将旋转向量转化为旋转矩阵
#     # pos=np.dot(-np.linalg.inv(R),translation_vector)
#     # print("SQPNP计算的坐标：")
#     # print(pos)
#     # print("真实坐标：")
#     # print(real_P)
#
#
#     detect_dir="E:\Code\Py_pro\YOLOV5\yolov5-master\\runs\detect"
#     newest_detectdir=detect_dir+'\\'+dy.get_newest_detectdir(detect_dir) # 选择detect文件目录最新的内容
#     res=dy.get_res(newest_detectdir+'\\'+"labels")
#     RT=cal_RT(res)
#     camera_poss=cal_camerapos(RT)








