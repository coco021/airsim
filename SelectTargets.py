"""
process_whole_file 的作用是读取整个文件夹中的所有txt文件，然后将其转换为列表。
process_yolo_txt 的作用是读取单个txt文件
这样，我在主程序中就可以直接调用 process_whole_file 来获取整个文件夹的结果，只需要传入文件夹路径即可。
"""

import os


# 读取单个txt文件
def process_yolo_txt(txt_file_path, count):
    with open(txt_file_path, 'r') as file:
        result_txtfile = []
        # 创建一个空字典来存储结果
        result_oneitem = {}
        # 逐行读取TXT文件内容
        for line in file:
            # 将每行按空格分割成列表
            line_data = line.strip().split()
            # 这一行数据被收集到list中
            result_oneitem = {'count': count, 'category': line_data[0], 'x': line_data[1],
                              'y': line_data[2]}  # category, x, y
            result_txtfile.append(result_oneitem)  # append追加单个元素，所以是将整个字典加入。

    # 如果我读取完后就删除
    # os.remove(txt_file_path)
    return result_txtfile


# 定义要处理的TXT文件夹路径
txt_folder = "path/to/your/txt/folder"


def process_whole_file(txt_folder):
    result_folder = []
    # 遍历TXT文件夹中的每个TXT文件
    count = 0
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            # 构建完整的TXT文件路径
            txt_file_path = os.path.join(txt_folder, filename)
            # 处理每个TXT文件
            # 计数

            result_txtfile = process_yolo_txt(txt_file_path, count)
            result_folder.append(result_txtfile)

            count += 1
    return result_folder
