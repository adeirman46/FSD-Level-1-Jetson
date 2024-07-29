import pyzed.sl as sl
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import sys
import torch
from collections import deque
import time
from sort import Sort
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import threading
from datetime import datetime
import serial

sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

class UKFTracker:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.points = MerweScaledSigmaPoints(1, alpha=0.1, beta=2., kappa=0)
        self.ukf = UKF(dim_x=1, dim_z=1, dt=1, fx=self.fx, hx=self.hx, points=self.points)
        self.ukf.x = np.array([initial_value])
        self.ukf.P *= 1
        self.ukf.R *= measurement_variance
        self.ukf.Q *= process_variance

    @staticmethod
    def fx(x, dt):
        return x

    @staticmethod
    def hx(x):
        return x

    def update(self, measurement):
        self.ukf.predict()
        self.ukf.update(measurement)
        return self.ukf.x[0]

class VehicleControl:
    @staticmethod
    def brake_control(distance):
        if 7 < distance < 9:
            return 0
        elif 5 < distance < 7:
            return 0.1
        elif 3 < distance < 5:
            return 0.3
        elif distance < 3:
            return 0.5
        return 0

    @staticmethod
    def velocity_control(distance):
        if distance > 7:
            return 25
        elif 5 < distance < 7:
            return 23
        elif 3 < distance < 5:
            return 22
        else:
            return 20

class ZEDCamera:
    def __init__(self):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30
        self.init_params.coordinate_units = sl.UNIT.METER
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.objects = sl.Objects()

    def open(self):
        if self.zed.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera")
            exit(1)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)

    def enable_object_detection(self):
        obj_param = sl.ObjectDetectionParameters()
        obj_param.enable_tracking = True
        if not self.zed.enable_object_detection(obj_param):
            print("Failed to enable object detection")
            exit(1)

    def grab_frame(self):
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            return self.image.get_data(), self.depth
        return None, None

    def retrieve_objects(self):
        self.zed.retrieve_objects(self.objects)

    def close(self):
        self.zed.close()

class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame, iou=0.45):
        return self.model.predict(frame, conf=self.conf_threshold, iou=iou, device=[0], imgsz=(384,672), show_labels=False, save=False, stream=True)

class Visualizer:
    @staticmethod
    def draw_trapezoid(frame):
        height, width = frame.shape[:2]
        top_width = int(width * 0.05)
        bottom_width = int(width * 0.5)
        top_x = (width - top_width) // 2
        bottom_x = (width - bottom_width) // 2
        points = np.array([
            [top_x, 0],
            [top_x + top_width, 0],
            [bottom_x + bottom_width, height],
            [bottom_x, height]
        ], np.int32)
        points = points.reshape((-1, 1, 2))
        mask = np.zeros_like(frame)
        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        return cv2.bitwise_and(frame, mask)

    @staticmethod
    def draw_detection(frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    @staticmethod
    def draw_tracking(frame, tracked_object):
        x1, y1, x2, y2, track_id = tracked_object
        track_label = f'Track ID: {int(track_id)}'
        cv2.putText(frame, track_label, (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    @staticmethod
    def draw_segmentation(frame, mask, color_index):
        color_mask = np.zeros_like(frame)
        if color_index == 0:
            color_mask[:, :, 1] = mask * 255  # Green for the first mask
        elif color_index == 1:
            color_mask[:, :, 2] = mask * 255  # Red for the second mask
        
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
            cx = int(np.mean(mask_indices[1]))
            cy = int(np.mean(mask_indices[0]))
            return color_mask, cx, cy
        return color_mask, None, None

    @staticmethod
    def overlay_segmentation(frame, color_mask):
        alpha = 0.5
        frame[np.any(color_mask != 0, axis=-1)] = (1 - alpha) * frame[np.any(color_mask != 0, axis=-1)] + alpha * color_mask[np.any(color_mask != 0, axis=-1)]
        return frame

    @staticmethod
    def draw_distance_and_velocity(frame, distance, velocity):
        cv2.putText(frame, f'Distance: {distance:.2f}m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Desired Velocity: {velocity:.2f}km/h', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

class SerialHandler:
    MAX_BUFF_LEN = 255

    def __init__(self, port, baudrate):
        self.port = serial.Serial(port, baudrate)
        self.distance = 0
        self.velocity = 0
        self.running = True

    def write_ser(self, cmd1, cmd2, cmd3):
        cmd = f'{cmd1},{cmd2},{cmd3}\n'
        self.port.write(cmd.encode())

    def read_ser(self, num_char=MAX_BUFF_LEN):
        string = self.port.read(num_char)
        return string.decode()

    def send_data_thread(self):
        while self.running:
            self.write_ser(self.distance, self.velocity, 0)  # Assuming 0 for cmd3
            time.sleep(0.01)  # 10ms

    def start(self):
        self.thread = threading.Thread(target=self.send_data_thread)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.port.close()

class LoggingHandler:
    def __init__(self, serial_handler):
        self.serial_handler = serial_handler
        self.running = True

    def logging_thread(self):
        while self.running:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename = f'logs_{datetime.now().strftime("%Y-%m-%d")}.txt'
            with open(log_filename, 'a') as log_file:
                log_entry = (
                    f"{current_time}, "
                    f"DISTANCE = {self.serial_handler.distance:.2f}, "
                    f"DESIRED VELOCITY = {self.serial_handler.velocity:.2f}\n"
                )
                log_file.write(log_entry)
            time.sleep(0.1)  # 100ms

    def start(self):
        self.thread = threading.Thread(target=self.logging_thread)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

class MainApplication:
    def __init__(self, args):
        self.camera = ZEDCamera()
        self.detector = ObjectDetector(args.model, args.conf)
        self.sort_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        self.frame_buffer = deque(maxlen=args.buffer_size)
        self.ukf_trackers = {}
        self.serial_handler = SerialHandler(args.serial_port, args.baudrate)
        self.logging_handler = LoggingHandler(self.serial_handler)

    def run(self):
        self.camera.open()
        self.camera.enable_object_detection()
        self.serial_handler.start()
        self.logging_handler.start()

        try:
            while True:
                frame, depth = self.camera.grab_frame()
                if frame is None:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                self.frame_buffer.append((frame_rgb, depth))

                if len(self.frame_buffer) == self.frame_buffer.maxlen:
                    self.process_frame()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def process_frame(self):
        start_time = time.time()
        frame_to_process, depth_to_process = self.frame_buffer.popleft()
        
        bitwise_frame = Visualizer.draw_trapezoid(frame_to_process)
        bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
        
        results = self.detector.detect(bitwise_frame)
        
        plotted_img = []
        detections = []
        min_bbox_distance = float('inf')
        half_lane_distance = None
        
        for result in results:
            if isinstance(result, list):
                result_ = result[0]
                boxes = result_.boxes
                plot_args = dict({'line_width': None, 'boxes': True, 'conf': True, 'labels': False})
                plotted_img.append(result_.plot(**plot_args))

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    if conf >= self.detector.conf_threshold:
                        detections.append([x1, y1, x2, y2, conf])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        if 0 <= cx < depth_to_process.get_width() and 0 <= cy < depth_to_process.get_height():
                            depth_value = depth_to_process.get_value(cx, cy)[1]
                            
                            if np.isfinite(depth_value):
                                object_id = f"{cls}_{x1}_{y1}_{x2}_{y2}"
                                if object_id not in self.ukf_trackers:
                                    self.ukf_trackers[object_id] = UKFTracker(process_variance=0.01, measurement_variance=0.1, initial_value=depth_value)
                                
                                filtered_depth = self.ukf_trackers[object_id].update(depth_value)
                                min_bbox_distance = min(min_bbox_distance, filtered_depth)

                                Visualizer.draw_detection(frame_to_process, x1, y1, x2, y2)
                                cv2.putText(frame_to_process, f'{filtered_depth:.2f}m', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                plotted_img.append(result)

        detections_np = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = self.sort_tracker.update(detections_np)

        combined_img = frame_to_process.copy()
        self.camera.retrieve_objects()

        for tracked_object in tracked_objects:
            Visualizer.draw_tracking(combined_img, tracked_object)

        for i in range(1, len(plotted_img)):
            mask = plotted_img[i][0].to(torch.uint8).cpu().numpy()
            color_mask, cx, cy = Visualizer.draw_segmentation(combined_img, mask, i-1)
            
            if cx is not None and cy is not None:
                depth_value = depth_to_process.get_value(cx, cy)[1]
                if np.isfinite(depth_value):
                    half_lane_distance = depth_value
            
            combined_img = Visualizer.overlay_segmentation(combined_img, color_mask)

        if half_lane_distance is not None:
            effective_distance = min(min_bbox_distance, 2 * half_lane_distance)
        else:
            effective_distance = min_bbox_distance

        velocity = VehicleControl.velocity_control(effective_distance)
        Visualizer.draw_distance_and_velocity(combined_img, effective_distance, velocity)

        # Update serial handler values
        self.serial_handler.distance = effective_distance
        self.serial_handler.velocity = velocity

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("ZED + YOLOv8 - Combined Detection and Segmentation", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

    def cleanup(self):
        self.camera.close()
        self.serial_handler.stop()
        self.logging_handler.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
    parser.add_argument('--model', type=str, default='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for object detection')
    parser.add_argument('--buffer_size', type=int, default=5, help='Size of the frame buffer')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0', help='Serial port for communication')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate for serial communication')
    args = parser.parse_args()

    app = MainApplication(args)
    app.run()