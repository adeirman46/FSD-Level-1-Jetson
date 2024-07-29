import sys

sys.path.insert(0, "/home/xianyang/Desktop/YOLOv8-multi-task")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('runs/multi/yolop0/weights/best.pt',task='multi')  # build a new model from YAML
# model = YOLO('v4s.pt',task='multi')
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='datasets/testt.yaml', batch=10, epochs=100, imgsz=640,
            device=[0], name='yolopm', val=True, task='multi', classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            combine_class=[], single_cls=False)
