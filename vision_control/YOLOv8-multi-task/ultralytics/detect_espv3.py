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
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

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
            msg = self.bus.recv(1)  # 1s timeout
            if msg is not None and msg.arbitration_id == self.MESSAGE_ID_SPEED:
                third_byte_speed = msg.data[3]
                combined_data_speed = third_byte_speed
                self.speed = (0.2829 * combined_data_speed + 0.973)

            velocity = self.velocity_control()
            brake = self.brake_control()
            self.write_ser(str(velocity), str(brake), str(self.speed))

            print(f'ACTUAL VELOCITY = {self.speed:0.0f}, DESIRED VELOCITY = {velocity}, DESIRED BRAKE = {brake}, OBSTACLE DISTANCE = {self.obstacle_distance:.2f}')

            time.sleep(max(0, 0.001 - (time.time() % 0.001)))

    def start(self):
        self.thread = threading.Thread(target=self.can_serial_thread)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.bus.shutdown()

class UnscentedKalmanFilterWrapper:
    def __init__(self, process_variance, measurement_variance):
        # Define points for the UKF
        self.points = MerweScaledSigmaPoints(n=1, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, fx=self.fx, hx=self.hx, dt=1.0, points=self.points)
        self.ukf.P *= process_variance  # Initial uncertainty
        self.ukf.R *= measurement_variance  # Measurement noise
        self.ukf.Q *= process_variance  # Process noise

    def fx(self, x, dt):
        # State transition function: assume constant velocity model
        return x

    def hx(self, x):
        # Measurement function: assume direct observation of the state
        return x

    def update(self, measurement):
        self.ukf.predict()
        self.ukf.update(measurement)
        return self.ukf.x[0]

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

        # Dictionary to store UKF for each tracked object
        self.kalman_filters = {}

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

        # Draw object detection results
        for obj in self.objects.object_list:
            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                bounding_box = obj.bounding_box_2d
                x1, y1 = map(int, bounding_box[0])
                x2, y2 = map(int, bounding_box[2])
                track_id = obj.id
                track_label = f'Track ID: {track_id}'
                cv2.putText(combined_img, track_label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        plotted_img = []
        for result in results:
            if isinstance(result, list):
                result_ = result[0]
                boxes = result_.boxes
                masks = result_.masks
                plot_args = dict({'line_width': None, 'boxes': True, 'conf': True, 'labels': False})
                plotted_img.append(result_.plot(**plot_args))

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    if conf >= self.conf_threshold:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        depth_value = self.depth.get_value(cx, cy)[1]

                        if np.isfinite(depth_value):
                            # Apply Unscented Kalman filter
                            object_id = f"{cls}_{x1}_{y1}_{x2}_{y2}"  # Create a unique ID for each object
                            if object_id not in self.kalman_filters:
                                self.kalman_filters[object_id] = UnscentedKalmanFilterWrapper(process_variance=0.01, measurement_variance=0.1)
                            
                            filtered_depth = self.kalman_filters[object_id].update(depth_value)

                            label = f'{self.model.names[cls]} {filtered_depth:.2f}m'
                            cv2.putText(combined_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Update obstacle distance if this is the closest object
                            if filtered_depth < can_serial_handler.obstacle_distance:
                                can_serial_handler.obstacle_distance = filtered_depth

                if masks is not None:
                    mask_data = masks.data[0]
                    colored_mask = np.zeros_like(frame, dtype=np.uint8)

                    for mask in mask_data:
                        mask = mask.cpu().numpy()
                        mask = (mask * 255).astype(np.uint8)
                        mask_color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)[0].tolist()
                        colored_mask[mask == 255] = mask_color

                        mask_indices = np.where(mask == 255)
                        depths_in_mask = self.depth.get_data()[mask_indices]
                        valid_depths = depths_in_mask[np.isfinite(depths_in_mask)]

                        if len(valid_depths) > 0:
                            avg_depth = np.mean(valid_depths)

                            # Display the distance of the mask
                            mask_label_position = (mask_indices[1][0], mask_indices[0][0])
                            mask_label = f'{avg_depth:.2f}m'
                            cv2.putText(colored_mask, mask_label, mask_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            # Update obstacle distance if this is the closest segment
                            if avg_depth < can_serial_handler.obstacle_distance:
                                can_serial_handler.obstacle_distance = avg_depth

                        combined_img = cv2.addWeighted(combined_img, 1, colored_mask, 0.5, 0)

        # Reset the obstacle distance for the next frame
        can_serial_handler.obstacle_distance = float('inf')

        return combined_img, plotted_img

# Add keyboard event handling
def on_press(key):
    if key == keyboard.Key.esc:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Process YOLO model path and confidence threshold.')
    parser.add_argument('--model', type=str, default='yolov8s-seg.pt', help='Path to the YOLO model.')
    parser.add_argument('--conf', type=float, default=0.7, help='Confidence threshold for YOLO predictions.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    can_serial_handler = CANSerialHandler()
    zed_handler = ZEDHandler(args.model, args.conf)

    if not zed_handler.open():
        sys.exit(1)

    can_serial_handler.start()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    cv2.namedWindow('ZED YOLO', cv2.WINDOW_AUTOSIZE)

    try:
        while listener.running:
            frame = zed_handler.grab_frame()
            if frame is None:
                print("Failed to grab frame")
                continue

            combined_img, plotted_img = zed_handler.process_frame(frame)

            # Display the combined image with detection results
            cv2.imshow('ZED YOLO', combined_img)

            # Optional: Display plotted image if needed
            if plotted_img:
                cv2.imshow('YOLO Segmentation', plotted_img[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received, exiting...")

    can_serial_handler.stop()
    zed_handler.close()
    cv2.destroyAllWindows()
