#!/usr/bin/bash
# --fp16，默认float
onnx_path="$1" # .onnx文件路径
output_path="$2" # .engine文件路径
export_layer_info_folder_path="$3" # 导出文件路径
/home/ht/Thirdparty/TensorRT-8.5.3.1/bin/trtexec \
--minShapes=kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x256,desc1:1x1x256 \
--optShapes=kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x256,desc1:1x512x256 \
--maxShapes=kpts0:1x1024x2,kpts1:1x1024x2,desc0:1x1024x256,desc1:1x1024x256 \
--onnx=$onnx_path --saveEngine=$output_path --profilingVerbosity=detailed --verbose \
--memPoolSize=workspace:2048 --dumpProfile  --separateProfileRun \
--exportLayerInfo="${export_layer_info_folder_path}/exportLayerInfo_fp32.txt" --exportProfile="${export_layer_info_folder_path}/exportProfile_fp32.txt"
