import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

class VideoStreamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Driver Assistance System")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side (video and info panel)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, 7)  # Allocate 70% of the width

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.video_label, 9)  # Allocate 90% of the height

        info_panel = QWidget()
        info_layout = QHBoxLayout(info_panel)
        left_layout.addWidget(info_panel, 1)  # Allocate 10% of the height

        self.distance_label = QLabel("Distance: N/A")
        self.speed_label = QLabel("Speed: N/A")
        self.brake_label = QLabel("Brake: N/A")
        for label in (self.distance_label, self.speed_label, self.brake_label):
            label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #2a2a2a; border-radius: 5px;")
            info_layout.addWidget(label)

        # Right side (2D plane visualization)
        self.plane_label = QLabel()
        self.plane_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plane_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plane_label.setStyleSheet("background-color: #2a2a2a;")
        main_layout.addWidget(self.plane_label, 3)  # Allocate 30% of the width

        self.showMaximized()  # Start in full screen mode

    def update_frame(self, frame, distance=None, velocity=None, brake_status=None):
        # Update video frame
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

        # Update info panel
        if distance is not None:
            self.distance_label.setText(f"Distance: {distance:.2f}m")
        if velocity is not None:
            self.speed_label.setText(f"Speed: {velocity:.2f} km/h")
        if brake_status is not None:
            self.brake_label.setText(f"Brake: {brake_status}")

    def update_plane(self, plane_image):
        # Update 2D plane visualization
        h, w, ch = plane_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(plane_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.plane_label.setPixmap(pixmap.scaled(self.plane_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        # Implement any cleanup if necessary
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoStreamGUI()
    sys.exit(app.exec())