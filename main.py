import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# 1. Initialize MediaPipe HandLandmarker (Tasks API)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=lambda result, output_image, timestamp_ms: set_hand_result(result)
)

latest_hand_result = None

def set_hand_result(result):
    global latest_hand_result
    latest_hand_result = result

detector = vision.HandLandmarker.create_from_options(options)

# Hand skeletal joint connection pairs (0-20)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),        # Thumb
    (0,5), (5,6), (6,7), (7,8),        # Index Finger
    (5,9), (9,10), (10,11), (11,12),   # Middle Finger
    (9,13), (13,14), (14,15), (15,16), # Ring Finger
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky & Palm base
]

def draw_xray_skeleton(frame, keypoints_data, confidences):
    """Draws accurate body skeleton connections."""
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for person_kpts, person_conf in zip(keypoints_data, confidences):
        for p1_idx, p2_idx in skeleton_connections:
            if person_conf[p1_idx] > 0.5 and person_conf[p2_idx] > 0.5:
                pt1 = tuple(map(int, person_kpts[p1_idx]))
                pt2 = tuple(map(int, person_kpts[p2_idx]))
                
                cv2.line(frame, pt1, pt2, (0, 0, 0), 8, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, (255, 255, 0), 4, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        for idx, (x, y) in enumerate(person_kpts):
            if person_conf[idx] > 0.5:
                center = (int(x), int(y))
                cv2.circle(frame, center, 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, center, 4, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, center, 2, (255, 255, 255), -1, cv2.LINE_AA)

def draw_hand_fingers(frame, hand_result):
    """Draws glowing wireframes for all 21 finger joints per hand."""
    if not hand_result or not hand_result.hand_landmarks:
        return

    h, w, _ = frame.shape

    for hand_landmarks in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for p1_idx, p2_idx in HAND_CONNECTIONS:
            pt1 = pts[p1_idx]
            pt2 = pts[p2_idx]

            cv2.line(frame, pt1, pt2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 1, (255, 255, 255), -1, cv2.LINE_AA)

def main():
    model = YOLO("yolov8n-pose.pt")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Running Full Body & Finger Wireframe Visualizer... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Run YOLO Body Pose
        body_results = model(frame, conf=0.45, verbose=False)
        if len(body_results) > 0 and body_results[0].keypoints is not None:
            kpts_data = body_results[0].keypoints.xy.cpu().numpy()
            confs = body_results[0].keypoints.conf.cpu().numpy()
            draw_xray_skeleton(frame, kpts_data, confs)

        # 2. Run MediaPipe Finger Tracking (Async Stream)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        detector.detect_async(mp_image, timestamp_ms)

        if latest_hand_result:
            draw_hand_fingers(frame, latest_hand_result)

        cv2.imshow("Full Body & Finger Vision Overlay", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()