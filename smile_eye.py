import mediapipe as mp
import numpy as np
import pickle
import cv2

# Load your smile detection model
with open('smile_model.pkl', 'rb') as file:
    smile_model = pickle.load(file)


# Load your eye contact detection model
with open('eye_contact_model.pkl', 'rb') as file:
    eye_contact_model = pickle.load(file)

# Initialize MediaPipe face landmark detector
mp_face_mesh = mp.solutions.face_mesh

SMILE_INDEXES =  list(mp_face_mesh.FACEMESH_LIPS)
SMILE_INDEXES = {element for tup in SMILE_INDEXES for element in tup}

EYE_INDEXES =  list(mp_face_mesh.FACEMESH_IRISES) + list(mp_face_mesh.FACEMESH_LEFT_EYE) + list(mp_face_mesh.FACEMESH_RIGHT_EYE)
EYE_INDEXES = {element for tup in EYE_INDEXES for element in tup}

def getCoords(face_mesh , idx):
  landmark = face_mesh[idx]
  x , y , z = landmark.x , landmark.y , landmark.z
  return [x , y , z]

def getAllLandmarksCoords(face_mesh , landmark_type):
    assert(landmark_type == 'lips' or landmark_type == 'eyes')

    coords = []
    if landmark_type == 'lips':
        POINT_INDEXES = SMILE_INDEXES 
    else:
        POINT_INDEXES = EYE_INDEXES
  
    for idx in POINT_INDEXES:
        coords.append(getCoords(face_mesh , idx))
    coords = np.array(coords)
    return coords

def normalizeMesh(face_mesh_coords):
    normalized_mesh = (face_mesh_coords - face_mesh_coords.min(axis=0)) / (face_mesh_coords.max(axis=0) - face_mesh_coords.min(axis=0))
    return normalized_mesh.reshape(1 , -1)


def isSmiling(landmarks):
    prediction = eye_contact_model.predict(landmarks)
    return float(prediction[0])

def hasEyeContact(landmarks):
    prediction = smile_model.predict(landmarks)
    return float(prediction[0])

def smile_eye_detection(video_path):

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    smile_count = 0
    eyes_count = 0
    with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        for i in range(total_frames):
            print(i , sep= " ")
            ret , frame = video.read()
            if not ret:
                break
            try:
                # Convert the frame to RGB and process with face mesh
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                # Check if landmarks are detected
                if results.multi_face_landmarks:
                    face_mesh_coords = [landmark.landmark for landmark in results.multi_face_landmarks][0]
                    lips_normalized_mesh = normalizeMesh(getAllLandmarksCoords(face_mesh_coords , landmark_type='lips'))
                    eyes_normalized_mesh = normalizeMesh(getAllLandmarksCoords(face_mesh_coords , landmark_type='eyes'))

                    smile_count += isSmiling(lips_normalized_mesh)
                    eyes_count += hasEyeContact(eyes_normalized_mesh)
            except Exception as e:
                print(e)
    
    return (smile_count / total_frames) * 100 , (eyes_count / total_frames) * 100


        
print(smile_eye_detection("38ddd6fc-eca7-4c19-a0e6-88fcba09fd4f.mp4"))

        


