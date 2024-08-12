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
import can
import queue
import os

sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

# import gi
# gi.require_version('Gst', '1.0')
# from gi.repository import Gst, GObject, GLib
# Gst.init(None)
# import os

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

# class VehicleControl:
#     def __init__(self, serial_handler):
#         self.mass = 1500  # Vehicle mass in kg
#         self.obstacle_distance = float('inf')
#         self.serial_handler = serial_handler

#     def brake_control(self, desired_velocity):
#         if self.obstacle_distance <= 3:
#             return self.serial_handler.actual_brake -50
#         else:
#             return self.serial_handler.actual_brake

#     def velocity_control(self):
#         max_velocity = 25  # Max speed in km/h
#         min_velocity = 20  # Min speed in km/h
#         safe_distance = 10  # Safe distance in meters

#         if self.obstacle_distance > safe_distance:
#             return max_velocity
#         elif self.obstacle_distance > 0:
#             # Linearly interpolate between min and max velocity based on distance
#             return int(min_velocity + (max_velocity - min_velocity) * (self.obstacle_distance / safe_distance))
#         return min_velocity

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
    def draw_distance_and_velocity(frame, distance, velocity, actual_velocity, actual_brake, desired_brake, state_brake):
        cv2.putText(frame, f'Distance: {distance:.2f}m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Desired Velocity: {velocity:.2f}km/h', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Actual Velocity: {actual_velocity:.2f}km/h', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Desired Brake: {desired_brake:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Actual Brake: {actual_brake:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'State Brake: {state_brake}', (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    @staticmethod
    def create_2d_plane(tracked_objects, depth_map, initial_max_distance=30, initial_plane_size=(800, 800)):
        initial_plane_height, initial_plane_width = initial_plane_size
        max_depth = initial_max_distance
        max_x_distance = 0

        # First pass: determine the maximum depth and x_distance
        for tracked_object in tracked_objects:
            if len(tracked_object) >= 5:
                x1, y1, x2, y2, _ = tracked_object[:5]
                cx, cy = int((x1 + x2) // 2), int(y2)
                
                if 0 <= cx < depth_map.get_width() and 0 <= cy < depth_map.get_height():
                    try:
                        depth = depth_map.get_value(cx, cy)[1]
                        if np.isfinite(depth):
                            max_depth = max(max_depth, depth)
                            x_distance = abs((cx - depth_map.get_width() / 2) * depth / depth_map.get_width())
                            max_x_distance = max(max_x_distance, x_distance)
                    except Exception as e:
                        print(f"Error processing tracked object for max depth: {e}")

        # Calculate the scaling factor
        depth_scale = max(1, max_depth / initial_max_distance)
        x_scale = max(1, (2 * max_x_distance) / initial_max_distance)
        scale = max(depth_scale, x_scale)

        # Create the scaled plane
        plane_height = initial_plane_height
        plane_width = initial_plane_width
        plane = np.zeros((plane_height, plane_width, 3), dtype=np.uint8)
        plane[:] = (30, 30, 30)  # Dark gray background

        # Draw grid lines
        grid_spacing = 1  # meters
        pixels_per_meter = plane_height / (initial_max_distance * scale)
        for i in range(0, plane_width, int(grid_spacing * pixels_per_meter)):
            cv2.line(plane, (i, 0), (i, plane_height), (50, 50, 50), 1)
        for i in range(0, plane_height, int(grid_spacing * pixels_per_meter)):
            cv2.line(plane, (0, i), (plane_width, i), (50, 50, 50), 1)

        # Draw center line
        cv2.line(plane, (plane_width // 2, 0), (plane_width // 2, plane_height), (100, 100, 100), 2)

        # Draw ego vehicle at the bottom center
        ego_y = plane_height - 20
        ego_x = plane_width // 2
        cv2.circle(plane, (ego_x, ego_y), 7, (0, 255, 0), -1)

        # Plot tracked objects
        for tracked_object in tracked_objects:
            if len(tracked_object) >= 5:
                x1, y1, x2, y2, track_id = tracked_object[:5]
                cx, cy = int((x1 + x2) // 2), int(y2)
                
                if 0 <= cx < depth_map.get_width() and 0 <= cy < depth_map.get_height():
                    try:
                        depth = depth_map.get_value(cx, cy)[1]
                        
                        if np.isfinite(depth):
                            x_distance = (cx - depth_map.get_width() / 2) * depth / depth_map.get_width()
                            
                            # Scale to plane size with wider x-axis representation
                            plane_x = int(ego_x + (x_distance / (initial_max_distance * scale * 0.5)) * (plane_width / 2))
                            plane_y = int(ego_y - (depth / (initial_max_distance * scale)) * (plane_height - 40))
                            
                            # Draw object on plane as a rectangle
                            rect_width = int((0.1 / (initial_max_distance * scale * 0.5)) * plane_width) # 2m width -> (1 -> 2)
                            rect_height = int((0.1 / (initial_max_distance * scale)) * plane_height) #4.5m long -> (1 -> 4.5 )
                            rect_width = max(rect_width, 30)  # Minimum width of 30 pixels
                            rect_height = max(rect_height, 60)  # Minimum height of 60 pixels
                            rect_x = plane_x - rect_width // 2
                            rect_y = plane_y - rect_height // 2
                            cv2.rectangle(plane, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 255), -1)
                            
                            # Add text label
                            cv2.putText(plane, f'{int(track_id)}', (plane_x - 10, plane_y + 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Error processing tracked object: {e}")

        # Add distance markers
        for i in range(grid_spacing, int(initial_max_distance * scale) + 1, grid_spacing):
            y = int(ego_y - i * pixels_per_meter)
            cv2.putText(plane, f'{i}m', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return plane

class SerialHandler:
    MAX_BUFF_LEN = 255
    MESSAGE_ID_RPM = 0x47
    MESSAGE_ID_SPEED = 0x55
    MESSAGE_ID_BRAKE = 0x21

    def __init__(self, port, baudrate):
        self.port = serial.Serial(port, baudrate, timeout=0.1)
        self.speed = 0
        self.actual_brake = 0
        self.desired_velocity = 0
        self.desired_brake = 0
        self.brake_state = 0
        self.obstacle_distance = float('inf')
        self.running = True
        self.lock = threading.Lock()
        
        try:
            self.bus = can.interface.Bus(channel='can0', bustype='socketcan', receive_own_messages=False)
            self.can_connected = True
        except (can.CanError, OSError):
            print("Failed to connect to CAN bus. Running without CAN.")
            self.can_connected = False

    def write_ser(self, desired_velocity, desired_brake, actual_velocity):
        cmd = f'{desired_velocity},{desired_brake},{actual_velocity}\n'
        try:
            self.port.write(cmd.encode())
        except serial.SerialException as e:
            print(f"Failed to write to serial port: {e}")

    def read_ser(self):
        try:
            if self.port.in_waiting:
                line = self.port.readline().decode().strip()
                data = line.split(', ')
                return float(data[3].split(': ')[1])
        except (ValueError, UnicodeDecodeError, IndexError, serial.SerialException) as e:
            print(f"Error reading from serial port: {e}")
        return None

    def process_can_message(self, msg):
        if msg.arbitration_id == self.MESSAGE_ID_SPEED:
            third_byte_speed = msg.data[3]
            with self.lock:
                self.speed = int(0.2829 * third_byte_speed + 0.973)
        elif msg.arbitration_id == self.MESSAGE_ID_BRAKE:
            brake_flag = msg.data[0]
            with self.lock:
                self.brake_state = 1 if brake_flag == 0x60 else 0

    def update_control(self):
        with self.lock:
            obstacle_distance = self.obstacle_distance
            speed = self.speed

        if obstacle_distance <= 3:
            self.desired_brake = 200
        else:
            self.desired_brake = 230

        max_velocity = 25
        min_velocity = 20
        safe_distance = 10

        if obstacle_distance > safe_distance:
            self.desired_velocity = max_velocity
        elif obstacle_distance > 0:
            self.desired_velocity = int(min_velocity + (max_velocity - min_velocity) * (obstacle_distance / safe_distance))
        else:
            self.desired_velocity = min_velocity

        self.write_ser(self.desired_velocity, self.desired_brake, speed)

        new_actual_brake = self.read_ser()
        if new_actual_brake is not None:
            self.actual_brake = new_actual_brake

        print(f'ACTUAL VELOCITY = {speed:.2f}, DESIRED VELOCITY = {self.desired_velocity:.2f}, '
              f'DESIRED BRAKE = {self.desired_brake:.2f}, ACTUAL BRAKE = {self.actual_brake:.2f}, '
              f'OBSTACLE DISTANCE = {obstacle_distance:.2f}, BRAKE STATE = {self.brake_state}')

    def can_thread(self):
        while self.running and self.can_connected:
            try:
                message = self.bus.recv(timeout=0.1)
                if message:
                    self.process_can_message(message)
            except can.CanError as e:
                print(f"CAN error: {e}")

    def control_thread(self):
        while self.running:
            self.update_control()
            time.sleep(0.01)  # 10ms control loop

    def start(self):
        if self.can_connected:
            self.can_thread = threading.Thread(target=self.can_thread)
            self.can_thread.start()
        self.control_thread = threading.Thread(target=self.control_thread)
        self.control_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'can_thread'):
            self.can_thread.join()
        self.control_thread.join()
        self.port.close()
        if self.can_connected:
            self.bus.shutdown()

    def update_obstacle_distance(self, distance):
        with self.lock:
            self.obstacle_distance = distance

    def get_desired_velocity(self):
        return self.desired_velocity

    def get_desired_brake(self):
        return self.desired_brake

    def get_actual_velocity(self):
        return self.speed

    def get_actual_brake(self):
        return self.actual_brake

    def get_brake_state(self):
        return self.brake_state

# class LoggingHandler:
#     def __init__(self, serial_handler):
#         self.serial_handler = serial_handler
#         self.running = True
#         self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
#         os.makedirs(self.log_dir, exist_ok=True)

    # def logging_thread(self):
    #     while self.running:
    #         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         log_filename = os.path.join(self.log_dir, f'logs_{datetime.now().strftime("%Y-%m-%d")}.txt')
    #         try:
    #             with open(log_filename, 'a') as log_file:
    #                 log_entry = (
    #                     f"{current_time}, "
    #                     f"ACTUAL VELOCITY = {self.serial_handler.speed:.2f}, "
    #                     f"DESIRED VELOCITY = {self.serial_handler.desired_velocity:.2f}, "
    #                     f"ACTUAL BRAKE = {self.serial_handler.actual_brake:.2f}, "
    #                     f"DESIRED BRAKE = {self.serial_handler.desired_brake:.2f}, "
    #                     f"OBSTACLE DISTANCE = {self.serial_handler.obstacle_distance:.2f},"
    #                     f"BRAKE STATE = {self.serial_handler.brake_state}\n"
    #                 )
    #                 log_file.write(log_entry)
    #                 log_file.flush()  # Force write to disk
    #         except Exception as e:
    #             print(f"Error writing to log file: {e}")
    #         time.sleep(0.1)  # 100ms

    # def start(self):
    #     self.thread = threading.Thread(target=self.logging_thread)
    #     self.thread.start()

    # def stop(self):
    #     self.running = False
    #     if self.thread:
    #         self.thread.join()

# class LoggingHandler:
#     def __init__(self, serial_handler):
#         self.serial_handler = serial_handler
#         self.running = True
#         self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
#         os.makedirs(self.log_dir, exist_ok=True)

#     def __del__(self):
#         self.running = False
#         if hasattr(self, 'thread'):
#             self.thread.join()
#         self.port.close()

#     def logging_thread(self):
#         while self.running:
#             current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             log_filename = os.path.join(self.log_dir, f'logs_{datetime.now().strftime("%Y-%m-%d")}.txt')
#             try:
#                 with open(log_filename, 'a') as log_file:
#                     log_entry = (
#                         f"{current_time}, "
#                         f"ACTUAL VELOCITY = {self.serial_handler.speed:.2f}, "
#                         f"DESIRED VELOCITY = {self.serial_handler.desired_velocity:.2f}, "
#                         f"ACTUAL BRAKE = {self.serial_handler.actual_brake:.2f}, "
#                         f"DESIRED BRAKE = {self.serial_handler.desired_brake:.2f}, "
#                         f"OBSTACLE DISTANCE = {self.serial_handler.obstacle_distance:.2f},"
#                         f"BRAKE STATE = {self.serial_handler.brake_state}\n"
#                     )
#                     log_file.write(log_entry)
#                     log_file.flush()  # Force write to disk
#             except Exception as e:
#                 print(f"Error writing to log file: {e}")
#             time.sleep(0.1)  # 1 second logging interval

#     def start(self):
#         self.thread = threading.Thread(target=self.logging_thread)
#         self.thread.start()

#     def stop(self):
#         self.running = False
#         if hasattr(self, 'thread'):
#             self.thread.join()

class LoggingHandler:
    def __init__(self, serial_handler):
        self.serial_handler = serial_handler
        self.running = True
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.thread = None

    def logging_thread(self):
        while self.running:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_filename = os.path.join(self.log_dir, f'logs_{datetime.now().strftime("%Y-%m-%d")}.txt')
            try:
                with open(log_filename, 'a') as log_file:
                    log_entry = (
                        f"{current_time}, "
                        f"ACTUAL VELOCITY = {self.serial_handler.speed:.2f}, "
                        f"DESIRED VELOCITY = {self.serial_handler.desired_velocity:.2f}, "
                        f"ACTUAL BRAKE = {self.serial_handler.actual_brake:.2f}, "
                        f"DESIRED BRAKE = {self.serial_handler.desired_brake:.2f}, "
                        f"OBSTACLE DISTANCE = {self.serial_handler.obstacle_distance:.2f},"
                        f"BRAKE STATE = {self.serial_handler.brake_state}\n"
                    )
                    log_file.write(log_entry)
                    log_file.flush()  # Force write to disk
            except Exception as e:
                print(f"Error writing to log file: {e}")
            time.sleep(0.1)  # 100ms logging interval

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
        self.plane_size = (800, 800)
        #self.serial_handler = SerialHandler(args.serial_port, args.baudrate)
        #self.logging_handler = LoggingHandler(self.serial_handler)
        
        # self.is_serial_available = True
        # self.serial_handler = None # let app run without serial
        # try:
        #     self.serial_handler = SerialHandler(args.serial_port, args.baudrate)
        # except:
        #     self.is_serial_available = False
        # if self.is_serial_available:
        #     self.logging_handler = LoggingHandler(self.serial_handler)

        self.serial_handler = None
        try:
            self.serial_handler = SerialHandler(args.serial_port, args.baudrate)
            self.logging_handler = LoggingHandler(self.serial_handler)
        except Exception as e:
            print(f"Failed to initialize serial communication: {e}")

        # self.pipeline_str = (
        #     "appsrc name=source !"
        #     "video/x-raw,format=BGR,width=640,height=480,framerate=30/1 !"
        #     "videoconvert !"
        #     "video/x-raw,format=I420,width=640,height=480,framerate=30/1 !"
        #     "videoconvert !"
        #     "shmsink socket-path=/tmp/shmvideo shm-size=2000000 wait-for-connection=false"
        # )
        # self.pipeline = Gst.parse_launch(self.pipeline_str)
        # self.src = self.pipeline.get_by_name("source")
        # self.pipeline.set_state(Gst.State.PLAYING)

    def run(self):
        self.camera.open()
        self.camera.enable_object_detection()
        # if self.is_serial_available:
        #     self.serial_handler.start()
        #     self.logging_handler.start()
        if self.serial_handler:
            self.serial_handler.start()
            self.logging_handler.start()
        #self.serial_handler.start()
        #self.logging_handler.start()

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
        overall_start_time = time.time()

        # Frame retrieval
        frame_retrieval_start = time.time()
        frame_to_process, depth_to_process = self.frame_buffer.popleft()
        frame_retrieval_time = time.time() - frame_retrieval_start

        # Preprocessing
        preprocess_start = time.time()
        bitwise_frame = Visualizer.draw_trapezoid(frame_to_process)
        bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
        preprocess_time = time.time() - preprocess_start
        
        # Object detection
        detection_start = time.time()
        # results = self.detector.detect(bitwise_frame)
        results = self.detector.detect(frame_to_process)
        detection_time = time.time() - detection_start

        # Post-processing and visualization
        postprocess_start = time.time()
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
        postprocess_time = time.time() - postprocess_start

        # Tracking
        tracking_start = time.time()
        detections_np = np.array(detections) if detections else np.empty((0, 5))
        tracked_objects = self.sort_tracker.update(detections_np)
        tracking_time = time.time() - tracking_start

        # Final visualization
        visualization_start = time.time()
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

        if self.serial_handler:
            self.serial_handler.update_obstacle_distance(effective_distance)

            desired_velocity = self.serial_handler.get_desired_velocity()
            desired_brake = self.serial_handler.get_desired_brake()
            actual_velocity = self.serial_handler.get_actual_velocity()
            actual_brake = self.serial_handler.get_actual_brake()
            brake_state = self.serial_handler.get_brake_state()

            Visualizer.draw_distance_and_velocity(combined_img, effective_distance, desired_velocity, 
                                                  actual_velocity, actual_brake, desired_brake, brake_state)
        
        try:
            if tracked_objects.size > 0 and depth_to_process is not None:
                plane_img = Visualizer.create_2d_plane(tracked_objects, depth_to_process)
                if plane_img is not None:
                    # Resize the plane image to a fixed size for display
                    display_size = (800, 800)  # or any other size you prefer
                    resized_plane_img = cv2.resize(plane_img, display_size, interpolation=cv2.INTER_AREA)
                    cv2.imshow("2D Obstacle Visualization", resized_plane_img)
            else:
                print("No tracked objects or depth data available for 2D plane visualization")
        except Exception as e:
            print(f"Error in creating 2D plane visualization: {e}")
        
        visualization_time = time.time() - visualization_start

        # Calculate and display FPS
        overall_end_time = time.time()
        overall_time = (overall_end_time - overall_start_time)
        fps = 1 / overall_time
        cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display timing information
        cv2.putText(combined_img, f'Frame Retrieval: {frame_retrieval_time*1000:.2f}ms', (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Preprocessing: {preprocess_time*1000:.2f}ms', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Detection: {detection_time*1000:.2f}ms', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Post-processing: {postprocess_time*1000:.2f}ms', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Tracking: {tracking_time*1000:.2f}ms', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Visualization: {visualization_time*1000:.2f}ms', (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined_img, f'Total: {overall_time*1000:.2f}ms', (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("ZED + YOLOv8 - Combined Detection and Segmentation", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        # try:
        #     combined_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        #     combined_img = cv2.resize(combined_img, (640, 480))
        #     height, width, channels = combined_imgf.shape
        #     stride = channels * width
        #     gstBuffer = Gst.Buffer.new_allocate(None, height * stride, None)
        #     gstBuffer.fill(0, combined_img.tobytes())
        #     self.src.emit("push-buffer", gstBuffer)
        # except Exception as e:
        #     print("exception:", e)


    # def cleanup(self):
    #     # try:
    #     #     os.remove("/tmp/shmvideo")
    #     # except:
    #     #     pass
    #     self.camera.close()
    #     if self.is_serial_available:
    #         self.serial_handler.stop()
    #         self.logging_handler.stop()
    #     cv2.destroyAllWindows()
    def cleanup(self):
        self.camera.close()
        if self.serial_handler:
            self.serial_handler.stop()
            self.logging_handler.stop()  # Stop the logging handler
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
    parser.add_argument('--model', type=str, default='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/runs/multi/yolopm4/weights/best.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold for object detection')
    parser.add_argument('--buffer_size', type=int, default=5, help='Size of the frame buffer')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0', help='Serial port for communication')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate for serial communication')
    args = parser.parse_args()

    app = MainApplication(args)
    app.run()