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

MAX_BUFF_LEN = 255
MESSAGE_ID_RPM = 0x47
MESSAGE_ID_SPEED = 0x55

# flag to control the loops
running = True
speed = 0

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

# Function to handle CAN communication and serial writing
def can_serial_thread():
    global running, speed
    while running:
        # Read CAN messages
        msg = bus.recv(0.05)  # 50ms timeout
        if msg is not None and msg.arbitration_id == MESSAGE_ID_SPEED:
            third_byte_speed = msg.data[3]
            combined_data_speed = third_byte_speed
            speed = (0.2829 * combined_data_speed + 0.973)

        # Send constant velocity, brake, and current speed
        write_ser(str(40), str(0.0), str(speed))

        # Print speed (optional, for debugging)
        print(f'SPEED = {speed:0.0f}')

        # Sleep for remaining time to achieve 1ms cycle
        # time.sleep(max(0, 0.05 - (time.time() % 0.05)))
        time.sleep(max(0, 0.001 - (time.time() % 0.001)))

# Start the CAN and serial communication thread
can_serial_thread = threading.Thread(target=can_serial_thread)
can_serial_thread.start()

sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

number = 3  # input how many tasks in your work

# Parse command line arguments
parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
parser.add_argument('--model', type=str, default='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.pt', help='Path to the YOLOv8 model')
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

while running:
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

        # # Calculate the coordinates for the rectangular mask
        # rect_width = 500  # Adjust as needed for your specific centering
        # rect_height = height  # Adjust as needed for your specific centering
        # rect_x = (width - rect_width) // 2
        # rect_y = (height - rect_height)

        # # Create a black mask image
        # mask = np.zeros_like(frame_rgb)

        # # Draw a white filled rectangle on the mask
        # cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

        # # Perform bitwise AND operation between the frame and the mask
        # bitwise_frame = cv2.bitwise_and(frame_rgb, mask)
        # # Convert to RGB
        # bitwise_frame = cv2.cvtColor(bitwise_frame, cv2.COLOR_BGR2RGB)
        # # Resize to 1280x720
        # bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
        
        # Run YOLOv8 inference
        results = model.predict(frame_rgb, conf=args.conf, iou=0.45, device=[0], imgsz=(384,672), show_labels=False, save=False, stream=True)

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
                            # Display depth information
                            label = f'{model.names[cls]} {depth_value:.2f}m'
                            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

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

# Set running to False to stop the CAN/serial thread
running = False

# Wait for the CAN/serial thread to finish
can_serial_thread.join()

# Properly shut down the CAN bus
bus.shutdown()