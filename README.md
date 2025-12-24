README

Urban Mobility Perception System
Detection, Tracking, and Distance Estimation using Camera and LiDAR

Overview
This project implements a complete urban mobility perception pipeline capable of detecting, tracking, and estimating the distance of road users such as cars, trucks, buses, motorcycles, and pedestrians. The system is designed with real-world constraints in mind, including missing sensor data and incomplete calibration, and is evaluated on the KITTI Tracking dataset.
The focus of this project is on robust system design and engineering decisions, rather than perfect numerical accuracy.

Key Features
* Object detection using YOLOv8
* Multi-object tracking with ByteTrack
* LiDAR-assisted distance estimation
* Robust handling of missing LiDAR frames
* Per-object distance caching and smoothing
* Real-time FPS monitoring
* Annotated video output (MP4)

 
Project Structure

urban-mobility-perception/
¦
+-- data/
¦   +-- kitti/
¦       +-- image_02/0000/
¦       +-- velodyne/0000/
¦       +-- calib/0000.txt
¦
+-- utils/
¦   +-- calibration.py
¦   +-- lidar_loader.py
¦
+-- fusion/
¦   +-- lidar_camera_fusion.py
¦   +-- distance_estimation.py
¦
+-- output/
¦   +-- output.mp4
¦
+-- main.py
+-- README.md


Setup Instructions
1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate
2. Install Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy

How to Run
python main.py
The system will:
* Display annotated frames in real time
* Save the output video to output/output.mp4
Press ESC to stop execution.

Distance Estimation Strategy
The KITTI Tracking dataset does not provide LiDAR–camera extrinsic calibration for each sequence. To handle this limitation:
* LiDAR forward-axis distance is used as an approximate depth measure
* Distance is updated only when LiDAR data is available
* Last known distance is cached per object and reused during vision-only frames
* Distance values are smoothed using an exponential moving average
This approach prioritizes stability and clarity over unrealistic precision.

Output Visualization
Each detected object is annotated with:
* Class label (e.g., car, truck)
* Tracking ID
* Estimated distance (meters)
* Bounding box
* FPS counter (top-left)

Limitations
* Distance estimation is approximate due to missing calibration
* No full 3D object localization
* CPU-only execution

 Future Improvements
* Full LiDAR–camera calibration integration
* Per-object LiDAR point clustering
* Velocity and trajectory estimation
* GPU acceleration
* Risk-level or collision warning overlays

Dataset & Tools
* Dataset: KITTI Tracking Benchmark
* Detection: YOLOv8
* Tracking: ByteTrack
* Language: Python
* Libraries: PyTorch, OpenCV, Ultralytics

 Conclusion
This project demonstrates a realistic perception pipeline that reflects real-world challenges and engineering trade-offs. The system is robust, interpretable, and suitable for demonstration and further extension.


