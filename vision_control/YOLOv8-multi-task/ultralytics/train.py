# import sys
# sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")
# # 现在就可以导入Yolo类了
# from ultralytics import YOLO

# # Load a model
# model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/yolo/v4s.pt', task='multi')  # build a new model from YAML
# # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# # Train the model
# model.train(data='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/yolo/datasets/bdd-multi.yaml', batch=8, epochs=300, imgsz=(640,640), device=[0], name='yolopm', val=True, task='detect',classes=[0,2,3,4,5,6,9,10,11],combine_class=[0,2,3,4,5,6,9],single_cls=True)

import sys
sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")
from ultralytics import YOLO

# Load a model
model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-s.yaml', task='multi')

# Freeze segmentation layers to prevent them from being trained
for name, param in model.model.named_parameters():
    if 'seg' in name:  # assuming segmentation layers have 'seg' in their names
        param.requires_grad = False

# Train the model
model.train(data='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/datasets/bdd-multi.yaml', batch=8, epochs=300, imgsz=(640, 640), device=[0], name='yolopm', val=True, task='multi', classes=[0, 2, 3, 4, 5, 6, 9, 10, 11], combine_class=[0, 2, 3, 4, 5, 6, 9], single_cls=True)
