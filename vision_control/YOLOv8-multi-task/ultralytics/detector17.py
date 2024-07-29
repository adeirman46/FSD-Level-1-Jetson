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

sys.path.insert(0, "/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics")

number = 3  # input how many tasks in your work

# Parse command line arguments
parser = argparse.ArgumentParser(description="ZED YOLOv8 Object Detection")
parser.add_argument('--model', type=str, default='/home/irman/Documents/FSD-Level-1/vision_control/YOLOv8-multi-task/ultralytics/v4s.pt', help='Path to the YOLOv8 model')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for object detection')
parser.add_argument('--buffer_size', type=int, default=5, help='Size of the frame buffer')
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

zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)

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

# Create a frame buffer
frame_buffer = deque(maxlen=args.buffer_size)

# Initialize SORT tracker
sort_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)

def brake_control(distance):
    brake = 0
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
    if distance > 7:
        return 25
    elif 5 < distance < 7:
        return 23
    elif 3 < distance < 5:
        return 22
    else:
        return 20

def fx(x, dt):
    # State transition function for UKF
    return x

def hx(x):
    # Measurement function for UKF
    return x

class UKFTracker:
    def __init__(self, process_variance, measurement_variance, initial_value=0):
        self.points = MerweScaledSigmaPoints(1, alpha=0.1, beta=2., kappa=0)
        self.ukf = UKF(dim_x=1, dim_z=1, dt=1, fx=fx, hx=hx, points=self.points)
        self.ukf.x = np.array([initial_value])
        self.ukf.P *= 1
        self.ukf.R *= measurement_variance
        self.ukf.Q *= process_variance

    def update(self, measurement):
        self.ukf.predict()
        self.ukf.update(measurement)
        return self.ukf.x[0]

# Dictionary to store UKF trackers for each tracked object
ukf_trackers = {}

while True:
    # Grab an image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image and depth map
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Convert to OpenCV format
        frame = image.get_data()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Add frame to buffer
        frame_buffer.append((frame_rgb, depth))

        # Process the oldest frame in the buffer
        if len(frame_buffer) == args.buffer_size:
            start_time = time.time()
            
            frame_to_process, depth_to_process = frame_buffer.popleft()

            # Get the dimensions of the frame
            height, width = frame_to_process.shape[:2]

            # Define the trapezoid points
            top_width = int(width * 0.05)
            bottom_width = int(width * 0.5)
            top_x = (width - top_width) // 2
            bottom_x = (width - bottom_width) // 2
            points = np.array([
                [top_x, 0],  # Top-left
                [top_x + top_width, 0],  # Top-right
                [bottom_x + bottom_width, height],  # Bottom-right
                [bottom_x, height]  # Bottom-left
            ], np.int32)
            points = points.reshape((-1, 1, 2))

            # Create a black mask image
            mask = np.zeros_like(frame_to_process)

            # Draw a trapezoidal filled polygon on the mask
            cv2.fillConvexPoly(mask, points, (255, 255, 255))
            
            # Perform bitwise AND operation between the frame and the mask
            bitwise_frame = cv2.bitwise_and(frame_to_process, mask)
            
            # Resize to 1280x720
            bitwise_frame = cv2.resize(bitwise_frame, (1280, 720))
            
            # Run YOLOv8 inference
            results = model.predict(bitwise_frame, conf=args.conf, iou=0.45, device=[0], imgsz=(384,672), show_labels=False, save=False, stream=True)

            plotted_img = []
            detections = []  # Prepare a list for detections
            # Process the frame and get detections
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

                        if conf >= args.conf:
                            # Add detection to the list for SORT
                            detections.append([x1, y1, x2, y2, conf])

                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            # Ensure cx and cy are within the bounds of the depth map
                            if 0 <= cx < depth_to_process.get_width() and 0 <= cy < depth_to_process.get_height():
                                depth_value = depth_to_process.get_value(cx, cy)[1]
                                
                                if np.isfinite(depth_value):
                                    # Update UKF tracker and display information
                                    object_id = f"{cls}_{x1}_{y1}_{x2}_{y2}"
                                    if object_id not in ukf_trackers:
                                        ukf_trackers[object_id] = UKFTracker(process_variance=0.01, measurement_variance=0.1, initial_value=depth_value)
                                    
                                    filtered_depth = ukf_trackers[object_id].update(depth_value)
                                    velocity_value = velocity_control(filtered_depth)

                                    label = f'Distance: {filtered_depth:.2f}m'
                                    velocity_label = f'Velocity: {velocity_value:.2f}km/h'
                                    cv2.putText(frame_to_process, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                    cv2.putText(frame_to_process, velocity_label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                    cv2.rectangle(frame_to_process, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                else:
                                    print(f"Invalid depth value at ({cx}, {cy})")
                            else:
                                print(f"Center coordinates ({cx}, {cy}) out of depth map bounds")

                else:
                    plotted_img.append(result)

                # Convert detections to numpy array before passing to SORT
                detections_np = np.array(detections) if detections else np.empty((0, 5))

                # Update SORT tracker with detections
                tracked_objects = sort_tracker.update(detections_np)

            # Combine object detection and segmentation
            if len(plotted_img) > 1:
                combined_img = frame_to_process.copy()
                
                # Retrieve object detection results
                zed.retrieve_objects(objects)

                # Draw object detection results using SORT
                for (x1, y1, x2, y2, track_id) in tracked_objects:
                    track_label = f'Track ID: {int(track_id)}'
                    cv2.putText(combined_img, track_label, (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(combined_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Overlay segmentation masks
                for i in range(1, len(plotted_img)):
                    mask = plotted_img[i][0].to(torch.uint8).cpu().numpy()
                    color_mask = np.zeros_like(combined_img)
                    
                    if i == 1:
                        color_mask[:, :, 1] = mask * 255  # Green for the first mask
                    elif i == 2:
                        color_mask[:, :, 2] = mask * 255  # Red for the second mask

                    mask_indices = np.where(mask > 0)
                    if len(mask_indices[0]) > 0 and len(mask_indices[1]) > 0:
                        cx = int(np.mean(mask_indices[1]))
                        cy = int(np.mean(mask_indices[0]))

                        depth_value = depth_to_process.get_value(cx, cy)[1]
                        
                        if np.isfinite(depth_value):
                            depth_label = f'Half Lane Distance: {depth_value:.2f}m'
                            cv2.putText(combined_img, depth_label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    alpha = 0.5
                    combined_img[np.any(color_mask != 0, axis=-1)] = (1 - alpha) * combined_img[np.any(color_mask != 0, axis=-1)] + alpha * color_mask[np.any(color_mask != 0, axis=-1)]

                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(combined_img, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the combined image
                cv2.imshow("ZED + YOLOv8 - Combined Detection and Segmentation", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the camera
zed.close()
cv2.destroyAllWindows()
