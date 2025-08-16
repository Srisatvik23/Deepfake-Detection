import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# ==== SETTINGS ====
INPUT_DIR = "data"
OUTPUT_DIR = "processed"
FRAMES_PER_VIDEO = 10  # number of frames to extract per video

# Create output folders if they don't exist
categories = ["fake", "real"]
for cat in categories:
    os.makedirs(os.path.join(OUTPUT_DIR, cat), exist_ok=True)

# MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Loop through categories
for cat in categories:
    video_folder = os.path.join(INPUT_DIR, cat)
    output_folder = os.path.join(OUTPUT_DIR, cat)

    # Process each video
    for video_file in tqdm(os.listdir(video_folder), desc=f"Processing {cat}"):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            continue

        # Calculate which frames to grab
        frame_indices = [int(i * total_frames / FRAMES_PER_VIDEO) for i in range(FRAMES_PER_VIDEO)]

        frame_count = 0
        save_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frame_indices:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x1 = int(bboxC.xmin * iw)
                        y1 = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)

                        face_img = frame[y1:y1 + h, x1:x1 + w]
                        if face_img.size != 0:
                            save_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_f{frame_count}_p{i}.jpg")
                            cv2.imwrite(save_path, face_img)
                            save_count += 1

            frame_count += 1

        cap.release()

print("✅ Face extraction complete.")










