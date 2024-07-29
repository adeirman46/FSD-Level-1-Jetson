import pyzed.sl as sl
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import sys
import serial
import can 
from pynput import keyboard
import torch
import threading
import time

class CANSerialHandler:
    MAX_BUFF_LEN = 255
    MESSAGE_ID_RPM = 0x47
    MESSAGE_ID_SPEED = 0x55

    def __init__(self):
        self.running = True
        self.speed = 0
        self.bus = can.interface.Bus(interface='socketcan', channel='can0', bitrate=500000)
        self.port = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=1)
        self.obstacle_distance = float('inf')

    def write_ser(self, cmd1, cmd2, cmd3):
        cmd = f'{cmd1},{cmd2},{cmd3}\n'
        self.port.write(cmd.encode())

    def read_ser(self, num_char=MAX_BUFF_LEN):
        string = self.port.read(num_char)
        return string.decode()

    def brake_control(self):
        if 7 < self.obstacle_distance < 9:
            return 0
        elif 5 < self.obstacle_distance < 7:
            return 0.1
        elif 3 < self.obstacle_distance < 5:
            return 0.3
        elif self.obstacle_distance < 3:
            return 0.5
        return 0

    # def velocity_control(self):
    #     if self.obstacle_distance > 7:
    #         return 35
    #     elif 5 < self.obstacle_distance < 7:
    #         return 33
    #     elif 3 < self.obstacle_distance < 5:
    #         return 32
    #     elif self.obstacle_distance < 3:
    #         return 30
    #     return 30
    
    def velocity_control(self):
        if self.obstacle_distance > 7:
            return 45
        elif 5 < self.obstacle_distance < 7:
            return 43
        elif 3 < self.obstacle_distance < 5:
            return 42
        elif self.obstacle_distance < 3:
            return 40
        return 40

    def can_serial_thread(self):
        while self.running:
            # msg = self.bus.recv(0.05)  # 50ms timeout
            msg = self.bus.recv(1)  # 1s timeout
            if msg is not None and msg.arbitration_id == self.MESSAGE_ID_SPEED:
                third_byte_speed = msg.data[3]
                combined_data_speed = third_byte_speed
                self.speed = (0.2829 * combined_data_speed + 0.973)

            velocity = self.velocity_control()
            brake = self.brake_control()
            self.write_ser(str(velocity), str(brake), str(self.speed))

            print(f'SPEED = {self.speed:0.0f}, VELOCITY = {velocity}, BRAKE = {brake}, DISTANCE = {self.obstacle_distance:.2f}')

            time.sleep(max(0, 0.001 - (time.time() % 0.001)))

    def start(self):
        self.thread = threading.Thread(target=self.can_serial_thread)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.bus.shutdown()

class ZEDHandler:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30
        self.init_params.coordinate_units = sl.UNIT.METER
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.obj_param = sl.ObjectDetectionParameters()
        self.obj_param.enable_tracking = True
        self.objects = sl.Objects()

    def open(self):
        if self.zed.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera")
            return False
        if not self.zed.enable_object_detection(self.obj_param):
            print("Failed to enable object detection")
            return False
        return True

    def close(self):
        self.zed.close()

    def grab_frame(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            frame = self.image.get_data()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            return frame_rgb
        return None

    def get_obstacle_distance(self, frame):
        height, width = frame.shape[:2]
        rect_width = 500
        rect_height = height
        rect_x = (width - rect_width) // 2
        rect_y = 0

        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)
        bitwise_frame = cv2.bitwise_and(frame, mask)

        depth_values = self.depth.get_data()[bitwise_frame[:,:,0] > 0]
        valid_depths = depth_values[np.isfinite(depth_values)]
        
        if len(valid_depths) > 0:
            return np.min(valid_depths)
        return float('inf')

    def process_frame(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, iou=0.45, device=[0], imgsz=(384,672), show_labels=False, save=False, stream=True)
        
        combined_img = frame.copy()
        self.zed.retrieve_objects(self.objects)

        for obj in self.objects.object_list:
            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                bounding_box = obj.bounding_box_2d
                x1, y1 = map(int, bounding_box[0])
                x2, y2 = map(int, bounding_box[2])
                track_id = obj.id
                track_label = f'Track ID: {track_id}'
                cv2.putText(combined_img, track_label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for result in results:
            if isinstance(result, list):
                result_ = result[0]
                boxes = result_.boxes
                masks = result_.masks

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    if conf >= self.conf_threshold:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        depth_value = self.depth.get_value(cx, cy)[1]

                        if np.isfinite(depth_value):
                            label = f'{self.model.names[cls]} {depth_value:.2f}m'
                            cv2.putText(combined_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if masks is not None:
                    for i, mask in enumerate(masks):
                        mask = mask.data[0].numpy().astype(np.uint8)
                        color_mask = np.zeros_like(combined_img)
                        color = (0, 255, 0) if i == 0 else (0, 0, 255)
                        color_mask[mask > 0] = color
                        alpha = 0.5
                        cv2.addWeighted(color_mask, alpha, combined_img, 1 - alpha, 0, combined_img)

        return combined_img

def main():
    parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
    parser.add_argument('--model', type=str, default='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for object detection')
    args = parser.parse_args()

    can_serial = CANSerialHandler()
    zed_handler = ZEDHandler(args.model, args.conf)

    if not zed_handler.open():
        return

    can_serial.start()

    while can_serial.running:
        frame = zed_handler.grab_frame()
        if frame is not None:
            can_serial.obstacle_distance = zed_handler.get_obstacle_distance(frame)
            processed_frame = zed_handler.process_frame(frame)
            cv2.imshow("ZED + YOLOv8 - Combined Detection and Segmentation", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    can_serial.stop()
    zed_handler.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()