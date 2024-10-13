import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
from PyQt6.QtCore import Qt, QTimer

class VideoStreamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Driver Assistance System")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label)

        info_panel = QWidget()
        info_layout = QHBoxLayout(info_panel)
        main_layout.addWidget(info_panel)

        self.distance_label = QLabel("Distance: N/A")
        self.speed_label = QLabel("Speed: N/A")
        self.brake_label = QLabel("Brake: N/A")
        for label in (self.distance_label, self.speed_label, self.brake_label):
            label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #2a2a2a; border-radius: 5px;")
            info_layout.addWidget(label)

        self.show()

    def update_frame(self, frame, distance=None, velocity=None, brake_status=None):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        if distance is not None:
            self.distance_label.setText(f"Distance: {distance:.2f}m")
        if velocity is not None:
            self.speed_label.setText(f"Speed: {velocity:.2f} km/h")
        if brake_status is not None:
            self.brake_label.setText(f"Brake: {brake_status}")

    def process_frame(self, frame):
        # Simulating distance calculation (replace with your actual logic)
        distance = np.random.uniform(0, 5)

        if distance < 2.5:
            # Add warning rectangle
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 255), 2)
            
            # Add warning text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Automatic Braking System Activated"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame, distance

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamGUI()
    sys.exit(app.exec())