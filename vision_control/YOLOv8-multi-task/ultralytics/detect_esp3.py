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

MAX_BUFF_LEN = 255
MESSAGE_ID_RPM = 0x47
MESSAGE_ID_SPEED  = 0x55

# flag to control the loop
running = True

# Function to handle key press events
def on_press(key):
    global running
    try:
        if key.char == 'q':
            running = False
    except AttributeError:
        pass

# Function to handle key release events
def on_release(key):
    pass

# Set up the CAN interface
bus = can.interface.Bus(interface='socketcan', channel='can0', bitrate=500000)

# Set up the keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
print(f"Listening for CAN messages with ID on can0. Press 'q' to quit.")


port = serial.Serial("/dev/ttyUSB0", baudrate=115200, timeout=1)

# write serial
def write_ser(cmd1, cmd2, cmd3):
    cmd = f'{cmd1},{cmd2},{cmd3}\n'
    port.write(cmd.encode())

# read serial
def read_ser(num_char=MAX_BUFF_LEN):
    string = port.read(num_char)
    return string.decode()

sys.path.insert(0, "/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/ultralytics")

number = 3  # input how many tasks in your work

# Parse command line arguments
parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
parser.add_argument('--model', type=str, default='/home/irman/Documents/zed-sdk/object detection/custom detector/python/pytorch_yolov8/YOLOv8-multi-task/runs/multi/yolopm/weights/best.pt', help='Path to the YOLOv8 model')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for object detection')
args = parser.parse_args()

# Load the YOLOv8 model
model = YOLO(args.model)

# Create a Camera object
zed = sl.Camera()

# Create an InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 video mode
init_params.camera_fps = 30  # Set FPS at 30
init_params.coordinate_units = sl.UNIT.METER  # Set the units to meters

# Open the camera
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit(1)

# Create Mat objects to hold the frames and depth
image = sl.Mat()
depth = sl.Mat()

# Define object detection parameters
obj_param = sl.ObjectDetectionParameters()
obj_param.enable_tracking = True  # Enable object tracking

# Enable object detection module
if not zed.enable_object_detection(obj_param):
    print("Failed to enable object detection")
    exit(1)

# Create objects to hold object detection results
objects = sl.Objects()

def brake_control(distance):
    # create speed profiling for car
    brake = 0  # initialize brake variable
    if 7 < distance < 9:
        brake = 0
    elif 5 < distance < 7:
        brake = 0.1
    elif 3 < distance < 5:
        brake = 0.3
    elif distance < 3:
        brake = 0.5
    return brake

def velocity_control(distance):
    velocity = 0  # initialize brake variable
    if distance > 7:
        velocity = 20
    elif 5 < distance < 7:
        velocity = 17
    elif 3 < distance < 5:
        velocity = 15
    elif distance < 3:
        velocity = 13
    return velocity

# Kalman Filter implementation
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_value
        self.estimate_error = 1

    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate

# Dictionary to store Kalman filters for each tracked object
kalman_filters = {}

while True:
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image and depth map
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Convert to OpenCV format
        frame = image.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Get the dimensions of the frame
        height, width = frame.shape[:2]

        # Calculate the coordinates for the rectangular mask
        rect_width = 500  # Adjust as needed for your specific centering
        rect_height = height  # Adjust as needed for your specific centering
        rect_x = (width - rect_width) // 2
        rect_y = (height - rect_height)

        # Create a black mask image
        mask = np.zeros_like(frame_rgb)

        # Draw a white filled rectangle on the mask
        cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

        # Perform bitwise AND operation between the frame and the mask
        bitwise_frame = cv2.bitwise_and(frame_rgb, mask)
        # Convert to RGB
        bitwise_frame = cv2.cvtColor(bitwise_frame, cv2.COLOR_BGR2RGB)
        # Resize to 1280x720
        bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
        # Run YOLOv8 inference
        results = model.predict(bitwise_frame, conf=args.conf, iou=0.45, device=[0], imgsz=(384,672), show_labels=False, save=False, stream=True)

        plotted_img = []
        for result in results:
            if isinstance(result, list):
                result_ = result[0]
                boxes = result_.boxes
                masks = result_.masks
                probs = result_.probs
                plot_args = dict({'line_width': None, 'boxes': True, 'conf': True, 'labels': False})
                plotted_img.append(result_.plot(**plot_args))

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0]
                    cls = int(box.cls[0])

                    if conf >= args.conf:
                        # Get the center of the bounding box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Get the depth value at the center of the bounding box
                        depth_value = depth.get_value(cx, cy)[1]

                        if np.isfinite(depth_value):
                            # Apply Kalman filter
                            object_id = f"{cls}_{x1}_{y1}_{x2}_{y2}"  # Create a unique ID for each object
                            if object_id not in kalman_filters:
                                kalman_filters[object_id] = KalmanFilter(process_variance=0.01, measurement_variance=0.1, initial_value=depth_value)
                            
                            filtered_depth = kalman_filters[object_id].update(depth_value)
                            
                            brake_value = brake_control(filtered_depth)
                            velocity_value = velocity_control(filtered_depth)

                            # Display depth information
                            label = f'{model.names[cls]} {filtered_depth:.2f}m'
                            brake_label = f'Brake condition: {brake_value:.2f}%'
                            velocity_label = f'Velocity: {velocity_value:.2f} m/s'
                            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            # cv2.putText(frame_rgb, brake_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.putText(frame_rgb, velocity_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            string = read_ser()
                            if string:
                                print(string.strip())
                        
                            msg = bus.recv(1.0)
                            if msg is not None and msg.arbitration_id == MESSAGE_ID_RPM:
                                third_byte_rpm = msg.data[1]
                                fifth_byte_rpm = msg.data[2]
                                combined_data_rpm = (third_byte_rpm << 8) | fifth_byte_rpm  # Combine the bytes
                                #speed = (0.001*combined_data)-9.25372+10
                                rpm = (combined_data_rpm*3.6)+990
                            elif msg is not None and msg.arbitration_id == MESSAGE_ID_SPEED:
                                third_byte_speed = msg.data[3]
                                fifth_byte_speed = msg.data[5]
                                combined_data_speed = (third_byte_speed) # Combine the bytes
                                #speed = (0.001*combined_data)-9.25372+10
                                speed = (0.2829*combined_data_speed + 0.973)
                                print(f'RPM : {rpm:0.0f}, SPEED = {speed:0.0f}')

                            write_ser(str(velocity_value), str(brake_value), str(speed))

            else:
                plotted_img.append(result)

        # Combine object detection and segmentation
        if len(plotted_img) > 1:
            combined_img = frame_rgb.copy()

            # Retrieve object detection results
            zed.retrieve_objects(objects)

            # Draw object detection results
            for obj in objects.object_list:
                if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                    bounding_box = obj.bounding_box_2d
                    x1, y1 = map(int, bounding_box[0])
                    x2, y2 = map(int, bounding_box[2])
                    track_id = obj.id

                    # Display tracking information
                    track_label = f'Track ID: {track_id}'
                    cv2.putText(combined_img, track_label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(combined_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Overlay segmentation masks
            for i in range(1, len(plotted_img)):
                mask = plotted_img[i][0].to(torch.uint8).cpu().numpy()
                color_mask = np.zeros_like(combined_img)
                
                if i == 1:
                    color_mask[:, :, 1] = mask * 255  # Green for the first mask
                elif i == 2:
                    color_mask[:, :, 2] = mask * 255  # Red for the second mask
                
                alpha = 0.5
                combined_img[np.any(color_mask != 0, axis=-1)] = (1 - alpha) * combined_img[np.any(color_mask != 0, axis=-1)] + alpha * color_mask[np.any(color_mask != 0, axis=-1)]

            # Display the combined image
            cv2.imshow("ZED + YOLOv8 - Combined Detection and Segmentation", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the camera
zed.close()
cv2.destroyAllWindows()

