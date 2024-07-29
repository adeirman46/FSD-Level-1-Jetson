import torch
from torch.autograd import Variable
from ultralytics import YOLO  # Replace with your YOLOv8 model definition

# Load your YOLOv8 model
model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.pt')

# Export to ONNX
dummy_input = Variable(torch.randn(1, 3, 416, 416))  # Example input shape
torch.onnx.export(model, dummy_input, "yolov8_multitask.onnx")
