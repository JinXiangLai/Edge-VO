

import os
import shutil

dataDir = "/home/ht/Opencv_Ceres_Eigen_example/tum_dataset/rgbd_dataset_freiburg2_desk"
imgDir = [f"{dataDir}/depth", f"{dataDir}/rgb"] # "/rgb"
odomFile = dataDir + "/groundtruth.txt"

# 创建图像保存文件夹
newImgDir = [f"{dataDir}/depth_image", f"{dataDir}/image"] # "/image"
for newDir in newImgDir:
    if(not os.path.exists(newDir)):
        os.mkdir(newDir)

# 创建里程计文件
newOdomFile = dataDir + "/odometry.csv"
if(not os.path.exists(newOdomFile)):
    shutil.copy(odomFile, newOdomFile)

# 创建图像时间戳文件
imgTimestampFile = [f"{dataDir}/depth_image_timestamp.csv", f"{dataDir}/image_timestamp.csv"]
for file in imgTimestampFile:
    with open(file, 'w') as time:
        # clear all
        pass

for i in range(len(imgDir)):
    dir = imgDir[i]
    sortTimestamp = []
    # 所有图片写入新文件夹
    for root, dirs, files in os.walk(dir):
        for f in files:
            img = os.path.join(root, f)
            newImg = os.path.join(newImgDir[i], f)
            # print(img)
            if not os.path.exists(newImg):
                shutil.copy(img, newImg)
            # print("f: ", f)
            sortTimestamp.append(float(f[:-4]))

    sortTimestamp.sort()
    with open(imgTimestampFile[i], 'w') as time:
        for t in sortTimestamp:
            time.write(format(t, '.6f')+'\n')

