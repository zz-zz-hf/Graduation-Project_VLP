import os
import re
import numpy as np

'''
工具类文件：
    获得YOLO模型detect结果的2D坐标
    需要提供detect_dir参数
'''

# 相机3D坐标对应关系
LED_3D=[[2.713,0.485,2.710],
         [2.713, 2.081, 2.710],
         [0.330, 0.485, 2.710],
         [0.335, 2.080, 2.710],
         [2.908, 2.085, 2.710],
         [2.908, 0.492, 2.710],
         [1.523, 0.485, 2.710] ,
         [1.528, 2.087, 2.710]]

def get_newest_detectdir(folder_path):
    # 初始化最大数字和对应的文件名
    max_number = -1
    max_filename = None

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 使用正则表达式匹配文件名中的数字
        match = re.search(r'\d+', filename)
        if match:
            # 提取数字
            number = int(match.group())
            # 如果当前数字大于已知的最大数字，则更新最大数字和文件名
            if number > max_number:
                max_number = number
                max_filename = filename
        else:
            max_filename=filename
    # 返回最大数字对应的文件名
    return max_filename

def analyze_onetxt(txt_filname):
    """
    :param txt_filname:
    :return: txt_filname文件下的2D和3D坐标
    """
    pos_3ds = []
    pos_2ds = []
    with open(txt_filname, "r", encoding='UTF-8') as f:
        file_content = f.read()
        # 初始化一个空列表来存储分割后的数据行

        # 按行分割文件内容
        for line in file_content.split('\n'):
            # 去除行尾的空白字符（包括换行符）
            line = line.strip()
            # 如果行不为空，按空格分割
            if line:
                # 分割数据并转换为浮点数（如果需要）
                split_line = [float(num) for num in line.split()]
                # 如果置信度低于0.36则舍去
                if split_line[-1]>=0.4:
                    pos_3ds.append(LED_3D[int(split_line [0])])
                    pos_2ds.append([split_line [1]*3264,split_line [2]*2464])
    return np.array(pos_3ds),np.array(pos_2ds)

def get_res(txt_dir):
    """
    :param txt_dir:
    :return: 返回一个[{文件名称:[array3D,array2D]},……]
    """
    res =[]
    # 遍历txt文件
    for filname in os.listdir(txt_dir):
        res3ds, res2ds = analyze_onetxt(txt_dir + '\\' + filname)
        res.append({filname:[res3ds,res2ds]})
    return res


if __name__ == '__main__':
    detect_dir="E:\Code\Py_pro\YOLOV5\yolov5-master\\runs\detect"
    newest_detectdir=detect_dir+'\\'+get_newest_detectdir(detect_dir) # 选择detect文件目录最新的内容
    res=get_res(newest_detectdir+'\\'+"labels")
