import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

SMILE_PATH = "smile_videos"
FRAMES_PATH = "smile_frames"

# Load your smile detection model
with open('smile_model.pkl', 'rb') as file:
    gbm_model = pickle.load(file)

# Initialize MediaPipe face landmark detector
mp_face_mesh = mp.solutions.face_mesh

POINT_INDEXES = list(mp_face_mesh.FACEMESH_LIPS)  # Adjust as needed
POINT_INDEXES = {element for tup in POINT_INDEXES for element in tup}

def getCoords(face_mesh, idx):
    landmark = face_mesh[idx]
    x, y, z = landmark.x, landmark.y, landmark.z
    return [x, y, z]

def getAllLandmarksCoords(face_mesh):
    coords = []
    for idx in POINT_INDEXES:
        coords.append(getCoords(face_mesh, idx))
    coords = np.array(coords)
    return coords

# Ensure the frames path exists
if not os.path.exists(FRAMES_PATH):
    os.makedirs(FRAMES_PATH)

for video_path in os.listdir(SMILE_PATH):
    input_video_path = os.path.join(SMILE_PATH, video_path)
    video = cv2.VideoCapture(input_video_path)

    # Create a directory for frames of this video
    video_frames_path = os.path.join(FRAMES_PATH, os.path.splitext(video_path)[0])
    if not os.path.exists(video_frames_path):
        os.makedirs(video_frames_path)

    frame_count = 0
    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for _ in range(601):
            ret, frame = video.read()
            if not ret:
                break

            try:
                # Convert the frame to RGB and process with face mesh
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                # Check if landmarks are detected
                if results.multi_face_landmarks:
                    face_mesh_coords = getAllLandmarksCoords([landmark.landmark for landmark in results.multi_face_landmarks][0])
                    normalized_mesh = (face_mesh_coords - face_mesh_coords.min(axis=0)) / (face_mesh_coords.max(axis=0) - face_mesh_coords.min(axis=0))

                    # Predict smile
                    is_smiling = gbm_model.predict(normalized_mesh.reshape(1, -1))
                    for x, y, _ in face_mesh_coords:
                        # Ensure the coordinates are integers
                        x_coord = int(x * frame.shape[1])
                        y_coord = int(y * frame.shape[0])
                        # Draw the circle
                        cv2.circle(frame, (x_coord, y_coord), 1, (255, 255, 255), -1)

                    # Display the result on the frame
                    text = 'Smile' if is_smiling[0] else 'No Smile'
                    color = (0, 255, 0) if is_smiling[0] else (0, 0, 255)
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                # Save the frame
                frame_filename = os.path.join(video_frames_path, f"frame_{frame_count:05d}.png")
                cv2.imwrite(frame_filename, frame)
                frame_count += 1

            except Exception as e:
                print(e)

    video.release()
    print(f'Frames saved for video: {video_path}')

cv2.destroyAllWindows()
print('Processing complete!')
