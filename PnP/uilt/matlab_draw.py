
# realword_points = np.array([[0.335, 2.080, 2.710],
#                             [1.523, 0.485, 2.710],
#                             [1.528, 2.087, 2.710],
#                             [2.713, 0.485, 2.710]], dtype=np.double) #6_3 3 6 7 0
import matplotlib.pyplot as plt
import re
import pnp_algorithm as pg

Standard_Camera={"6":[1.623,1.010,0.000],"7":[1.873,1.010,0.000],"8":[2.123,1.010,0.000],"9":[2.373,1.009,0.000],"10":[ 2.623,1.009,0.000],
             "11":[1.623,1.260,0.000],"12":[1.873,1.260,0.000],"13":[2.123,1.260,0.000],"14":[2.373,1.259,0.000],"15":[ 2.623,1.259,0.000 ],
             "16":[1.623,1.510,0.000],"17":[1.873,1.510,0.000],"18":[2.123,1.510,0.000],"19":[2.373,1.509,0.000],"20":[ 2.623,1.009,0.000]}

LED_3D=[[],
        [],
        []]

colors = {
    "6":'#FF0000',  # 红色
    "7":'#FFA500',  # 橙色
    "8":'#FFFF00',  # 黄色
    "9":'#FF4500',  # 橙红色
    "10":'#556B2F', # 深橄榄绿色
    "11":'#FF69B4',  # 热粉红色
    "12":'#FFC0CB',  # 浅粉红色
    "13":'#FFD700',  # 金色
    "14":'#DAA520',  # 深金色
    "15":'#20B2AA',  # 浅海绿色
    "16":'#FF7F50',  # 珊瑚色
    "17":'#9400D3',  # 紫色
    "18":'#4B0082',  # 靛蓝色
    "19":'#0000FF',  # 纯蓝色
    "20":'#7CFC00'  # 草坪绿
}
def draw_pnppos(pnp_pos):
    standard_set=set()
    standard_pos=[[],[],[]]
    vlp_pos={} # {P7:[[x],[y],[z]]……}
    for item in pnp_pos:
        filename=list(item.keys())[0]
        # 使用正则表达式匹配下划线之间的数字
        match = re.search(r'_(\d+)_', filename)
        if match:

            number = match.group(1)  # 提取第一个捕获组，即数字
            if number not in standard_set:
                standard_pos[0].append(Standard_Camera[str(number)][0])
                standard_pos[1].append(Standard_Camera[str(number)][1])
                standard_pos[2].append(Standard_Camera[str(number)][2])
                standard_set.add(number)
            if str(number) not in vlp_pos:
                vlp_pos[str(number)]=[[],[],[]]
            vlp_pos[str(number)][0].append(item[filename][0][0])
            vlp_pos[str(number)][1].append(item[filename][1][0])
            vlp_pos[str(number)][2].append(item[filename][2][0])
    # 绘制2D 3D点图
    draw_2D(standard_pos,vlp_pos)
    draw_3D(standard_pos, vlp_pos)



def draw_2D(Standard_2d,vlp_2d):
    # 创建一个新的图形
    plt.figure()
    # 在同一个坐标系中绘制第一组数据
    plt.scatter(Standard_2d[0], Standard_2d[1], color='k', label='Standard')
    for number_str in vlp_2d:
        plt.scatter(vlp_2d[number_str][0], vlp_2d[number_str][1], color=colors[number_str], label='P_'+number_str,marker='x')
    # 添加图例
    plt.legend()
    # 设置标题和轴标签
    plt.title('Datasets in One Plot')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    # 显示图形
    plt.show()

def draw_3D(Standard_2d,vlp_2d):

    fig = plt.figure()  # 创建一个新的图形
    ax = fig.add_subplot(111, projection='3d')  # 添加一个3D坐标轴

    # 绘制散点图
    ax.scatter(Standard_2d[0], Standard_2d[1],Standard_2d[2], c='k', label='Standard')
    for number_str in vlp_2d:
        ax.scatter(vlp_2d[number_str][0], vlp_2d[number_str][1],vlp_2d[number_str][2], c=colors[number_str], label='P_'+number_str,marker='x')
    # 添加图例
    ax.legend()
    ax.set_title('3D Scatter Plot')  # 设置标题
    ax.set_xlabel('X Axis')  # 设置X轴标签
    ax.set_ylabel('Y Axis')  # 设置Y轴标签
    ax.set_zlabel('Z Axis')  # 设置Z轴标签
    plt.show()  # 显示图形



if __name__ == '__main__':
    detect_dir = "E:\Code\Py_pro\YOLOV5\yolov5-master\\runs\detect"
    newest_detectdir = detect_dir + '\\' + pg.dy.get_newest_detectdir(detect_dir)  # 选择detect文件目录最新的内容
    res = pg.dy.get_res(newest_detectdir + '\\' + "labels")
    RT = pg.cal_RT(res)
    pnp_pos = pg.cal_camerapos(RT)
    draw_pnppos(pnp_pos)




    # # 3D 点图
    # # 准备数据
    # x = [1, 2, 3, 4, 5]
    # y = [2, 3, 5, 1, 4]
    # z = [4, 1, 2, 3, 5]
    #
    # fig = plt.figure()  # 创建一个新的图形
    # ax = fig.add_subplot(111, projection='3d')  # 添加一个3D坐标轴
    #
    # # 绘制散点图
    # ax.scatter(x, y, z)
    # ax.set_title('3D Scatter Plot')  # 设置标题
    # ax.set_xlabel('X Axis')  # 设置X轴标签
    # ax.set_ylabel('Y Axis')  # 设置Y轴标签
    # ax.set_zlabel('Z Axis')  # 设置Z轴标签
    # plt.show()  # 显示图形