import time
import cv2
import os
import tempfile
import airsim
# 导入YOLOv5的运行函数
from detect import run
from SelectTargets import process_whole_file
# 计数，response（airsim的返回结果），临时文件名
# write_txt: 得到一个txt，一行中包含了拍摄它的相机的位置和姿态
def write_txt(count, response, temp_position):
    # (Vector3r: V)拍摄时相机的全局位置 (Quaternionr: Q)拍摄时相机的姿态信息  (uint64)时间戳
    position = response.camera_position
    orientation = response.camera_orientation
    # time_stamp = response.time_stamp
    # 以追加模式打开文件
    with open(temp_position, 'a') as f:
        f.write(f'{count} V {position.x_val} {position.y_val} {position.z_val} Q {orientation.w_val} {orientation.x_val} {orientation.y_val} {orientation.z_val}\n')




client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)  # 获取控制权
client.armDisarm(True)  # 解锁
client.takeoffAsync().join()  # 起飞
""" 
    起始(X=14433.351562,Y=-4720.406250,Z=531.579529)
    终止(-1635.984863, -4747.981445, 707.665283)
    起始-终止差值(-16,069.336425, -27.575195, -176.085754)然后z要变号
    缩放100(-160.69336, -0.27575, 1.76085)
    由于这里要变号，所以你想让它往上飞，那么z就要变小
"""
# 设置目的地
destination = airsim.Vector3r(-160.69336, -0.27575, -30)
client.moveToZAsync(-30, 5).join()
client.moveToPositionAsync(destination.x_val, destination.y_val, destination.z_val, 4)
vector_list = []


# 创建一个临时文件，用于存储位置信息
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True, dir=None) as temp_file:
    temp_position = temp_file.name
count = 0
#  循环拍照，每隔几秒拍摄照片
while True:
    timestamp1 = int(time.time())

    response = client.simGetImage("front_center", airsim.ImageType.Scene)

    # 将图像和姿态保存到临时文件中，然后放进模型里

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True, dir=None) as temp_file:
        # 使用temp_file.name来获取文件名
        temp_image = temp_file.name
        cv2.imwrite(temp_image, response)

    # 结果保存在"D:\yolov5\runs\detected_txt"
    results = run(source=temp_image)

    for filename in os.listdir(r"D:\yolov5\runs\detected_txt"):
        if filename.endswith(".txt"):
            # 构建完整的TXT文件路径
            txt_file_path = os.path.join(r"D:\yolov5\runs\detected_txt", filename)
            # 处理每个TXT文件
            result = process_whole_file(txt_file_path)

    # 获取飞机的位置信息:写入到列表中
    count += 1
    drone_vector = {'x_var': response.camera_position.x_val,
              'y_var': response.camera_position.y_val,
              'z_var': response.camera_position.z_val}

    vector_list.append(drone_vector)
    time.sleep(1)






    # 获取一次位置，如果到达目的地，则跳出循环
    pose = client.simGetVehiclePose()
    if pose.position.distance_to(destination) < 1.0:
        break



client.landAsync().join()
client.takeoffAsync().join()

# 降落、关闭连接
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

