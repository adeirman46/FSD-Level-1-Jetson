# -*- coding: utf-8 -*-
# @Time    : 2024/7/16 17:21
# @Author  : XianYang🚀
# @Email   : xy_mts@163.com
# @File    : 1.py
# ------❤❤❤------ #

#
from ultralytics import YOLO

model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4n.pt',task='multi')
model.export(format='engine')