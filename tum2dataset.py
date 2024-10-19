

import os
import shutil

dataDir = "/home/laijinxiang/edge-slam/rgbd_dataset_freiburg2_xyz"
imgDir = dataDir + "/rgb"
odomFile = dataDir + "/groundtruth.txt"

# 创建图像保存文件夹
newImgDir = dataDir + "/image"
if(not os.path.exists(newImgDir)):
    os.mkdir(newImgDir)

# 创建里程计文件
newOdomFile = dataDir + "/odometry.csv"
if(not os.path.exists(newOdomFile)):
    shutil.copy(odomFile, newOdomFile)

# 创建图像时间戳文件
imgTimestampFile = dataDir + "/image_timestamp.csv"
with open(imgTimestampFile, 'w') as time:
    # clear all
    pass

sortTimestamp = []
# 所有图片写入新文件夹
for root, dirs, files in os.walk(imgDir):
    for f in files:
        img = os.path.join(root, f)
        newImg = os.path.join(newImgDir, f)
        print(img)
        if not os.path.exists(newImg):
            shutil.copy(img, newImg)
            sortTimestamp.append(float(f[:-4]))

sortTimestamp.sort()
with open(imgTimestampFile, 'a') as time:
    for t in sortTimestamp:
        time.write(format(t, '.6f')+'\n')

