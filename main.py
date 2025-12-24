import os
import cv2
import time

from ultralytics import YOLO

from utils.calibration import read_calib_file, get_matrices
from utils.lidar_loader import load_lidar
from fusion.lidar_camera_fusion import project_lidar_to_image
from fusion.distance_estimation import estimate_distance_from_lidar


#Configr
SEQ = "0000"
BASE = "data/kitti"
IMG_DIR = f"{BASE}/image_02/{SEQ}"
LIDAR_DIR = f"{BASE}/velodyne/{SEQ}"
CALIB_PATH = f"{BASE}/calib/{SEQ}.txt"
OUTPUT_VIDEO = "output/output.mp4"



def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    class_names = model.names

    # Load calibration
    calib = read_calib_file(CALIB_PATH)
    P2, R0, Tr = get_matrices(calib)

    images = sorted(os.listdir(IMG_DIR))

    # FPS
    prev_time = time.time()

    # Per-object cached distances
    track_distances = {}

 
    first_frame = cv2.imread(os.path.join(IMG_DIR, images[0]))
    h, w, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO, fourcc, 10, (w, h)
    )


    for img_name in images:
        img_path = os.path.join(IMG_DIR, img_name)
        lidar_path = os.path.join(
            LIDAR_DIR, img_name.replace(".png", ".bin")
        )

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Load LiDAR if available
        lidar = None
        if os.path.exists(lidar_path):
            lidar = load_lidar(lidar_path)

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            classes=[0, 2, 3, 5, 7],  # person, car, motorcycle, bus, truck
            tracker="bytetrack.yaml",
            conf=0.4
        )

        # Draw detections
        for box in results[0].boxes:
            if box.id is None:
                continue

            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, "obj")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            # Update distance if LiDAR exists
            if lidar is not None:
                _, depths = project_lidar_to_image(lidar, P2, R0, Tr)
                new_distance = estimate_distance_from_lidar(depths)

                if new_distance is not None:
                    if track_id in track_distances:
                        track_distances[track_id] = (
                            0.7 * track_distances[track_id] + 0.3 * new_distance
                        )
                    else:
                        track_distances[track_id] = new_distance

            # Label
            label = f"{class_name} | ID {track_id}"
            if track_id in track_distances:
                label += f" | {track_distances[track_id]:.1f} m"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Show + save
        cv2.imshow("Urban Mobility Perception", frame)
        video_writer.write(frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
