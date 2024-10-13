from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from plane_visualization import PlaneVisualizationWidget

class VideoStreamGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Advanced Driver Assistance System")
        self.setStyleSheet("background-color: #e6e9ed; color: #4676fa;")
        self.resize(1280, 720)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side (video and info panel)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, 7)  # Allocate 70% of the width

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background-color: #e6e9ed;")
        left_layout.addWidget(self.video_label, 9)  # Allocate 90% of the height

        # Info panel
        info_panel = QWidget()
        info_layout = QHBoxLayout(info_panel)
        left_layout.addWidget(info_panel, 1)  # Allocate 10% of the height

        self.distance_label = QLabel("Distance: N/A")
        self.speed_label = QLabel("Speed: N/A")
        self.brake_label = QLabel("Brake: N/A")
        for label in (self.distance_label, self.speed_label, self.brake_label):
            label.setStyleSheet("font-size: 16px; padding: 5px; background-color: #c0cffa; border-radius: 5px;")
            info_layout.addWidget(label)

        # Right side (Google Maps and 2D plane visualization)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        main_layout.addWidget(right_widget, 3)  # Allocate 30% of the width

        # Google Maps
        self.map_view = QWebEngineView()
        self.map_view.setUrl(QUrl("https://maps.google.com"))
        self.map_loading_label = QLabel('Loading Google Maps...')
        self.map_loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.map_loading_label.setFont(QFont('Arial', 16))
        self.map_loading_label.setStyleSheet("color: #4676fa; background-color: #c0cffa;")
        right_layout.addWidget(self.map_loading_label)
        right_layout.addWidget(self.map_view, 4)  # Allocate 70% of the height
        self.map_view.hide()
        self.map_view.loadFinished.connect(self.on_map_loaded)

        # 2D plane visualization
        self.plane_widget = PlaneVisualizationWidget()
        self.plane_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plane_widget.setStyleSheet("background-color: #c0cffa; border-radius: 5px;")
        right_layout.addWidget(self.plane_widget, 6)  # Allocate 30% of the height

        self.showMaximized()  # Start in full screen mode

    def on_map_loaded(self):
        self.map_loading_label.hide()
        self.map_view.show()

    def update_frame(self, frame, distance=None, velocity=None, brake_status=None):
        # Update video frame
        if frame is not None:
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

    def update_plane(self, tracked_objects, depth_map):
        self.plane_widget.update_data(tracked_objects, depth_map)

    def update_map(self, latitude, longitude):
        # Update the map view to center on the given coordinates
        url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
        self.map_view.setUrl(QUrl(url))

    def closeEvent(self, event):
        # Implement any cleanup if necessary
        event.accept()

if __name__ == "__main__":
    import sys
    
    app = QApplication(sys.argv)
    gui = VideoStreamGUI()
    gui.show()
    sys.exit(app.exec())