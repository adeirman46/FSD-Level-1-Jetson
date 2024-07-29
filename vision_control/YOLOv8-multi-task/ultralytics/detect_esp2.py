import pyzed.sl as sl
import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import sys
import serial
import can 
from pynput import keyboard

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

# def speed_control(distance):
#     # create speed profiling for car
#     speed1 = 0  # initialize brake variable
#     speed2 = 0
#     if distance < 30 and distance > 20:
#         speed1 = 
#     elif 5 < distance < 7:
#         brake = 0.1
#     elif 3 < distance < 5:
#         brake = 0.3
#     elif distance < 3:
#         brake = 0.5
    
#     return brake

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
                        brake_value = brake_control(depth_value)
                        velocity_value = velocity_control(depth_value)
                        # speed1 = 144
                        # speed2 = 206

                        # Display depth information
                        label = f'{model.names[cls]} {depth_value:.2f}m'
                        brake_label = f'Brake condition: {brake_value:.2f}%'

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

                        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.putText(frame_rgb, brake_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

            else:
                plotted_img.append(result)

        # Retrieve object detection results
        zed.retrieve_objects(objects)

        # Display object tracking information
        for obj in objects.object_list:
            if obj.tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                bounding_box = obj.bounding_box_2d
                x1, y1 = map(int, bounding_box[0])
                x2, y2 = map(int, bounding_box[2])
                track_id = obj.id

                # Display tracking information
                track_label = f'Track ID: {track_id}'
                cv2.putText(frame_rgb, track_label, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the image
        cv2.imshow("ZED + YOLOv8 - rgb", cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the camera
zed.close()
cv2.destroyAllWindows()
