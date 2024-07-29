import sys

sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO

number = 12  # input how many tasks in your work
# model = YOLO('runs/multi/yolopm11/weights/best.pt')  # Validate the model
# model.export(format='onnx')
#
#
model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.onnx')
model.predict(source="/home/irman/Documents/FSD-Level-1/vision_control/videos/jalan_tol_new.mp4", imgsz=(384,672), device=[0],
              name='v4_daytime', save=True, show=False,conf=0.5, iou=0.5, show_labels=True)
