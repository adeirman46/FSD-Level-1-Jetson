import sys
sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

from ultralytics import YOLO
import cv2

img = cv2.imread('/home/irman/Documents/FSD-Level-1/vision_control/videos/jalan_tol_new.png')


number = 3 #input how many tasks in your work
model = YOLO('/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.onnx', task='multi')  # Validate the model
results = model.predict(source=img, imgsz=(384,672), device=[0],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False, stream=True, save_txt=True)
# print('Pred results: ', results)
# model.export(format="onnx")
# for result in results:
#     if isinstance(result, list):
#         result_ = result[0]
#         boxes = result_.boxes
#         masks = result_.masks
#         probs = result_.probs
#         print("Masks: ", masks)
#         print("Bbox: ", boxes)

        # for box in boxes:
        #     x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        #     conf = box.conf[0]
        #     cls = int(box.cls[0])

        #     if conf >= args.conf:
        #         # Get the center of the bounding box
        #         cx = (x1 + x2) // 2
        #         cy = (y1 + y2) // 2

        #         # Get the depth value at the center of the bounding box
        #         depth_value = depth.get_value(cx, cy)[1]
        #         brake_cond = brake_control(depth_value)

        #         # Display depth information
        #         label = f'{model.names[cls]} {depth_value:.2f}m'
        #         brake_label = f'Brake condition: {brake_cond:.2f}%'
        #         cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #         cv2.putText(frame_rgb, brake_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #         cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)


