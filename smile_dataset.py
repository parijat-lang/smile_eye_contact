import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

POINT_INDEXES =  list(mp_face_mesh.FACEMESH_LIPS)
POINT_INDEXES = {element for tup in POINT_INDEXES for element in tup}

def getCoords(face_mesh , idx):
  landmark = face_mesh[idx]
  x , y , z = landmark.x , landmark.y , landmark.z
  return [x , y , z]

def getAllLandmarksCoords(face_mesh):
  coords = []
  for idx in POINT_INDEXES:
    coords.append(getCoords(face_mesh , idx))
  coords = np.array(coords)
  return coords


landmarks = []
labels = []
 
with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    try:
        for i , label in enumerate(pd.read_csv('labels.txt' , sep=" ")['label'].tolist()):
            img = cv2.imread("files/file{:04d}.jpg".format(i+1))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            # Check if landmarks are detected
            if results.multi_face_landmarks:
                face_mesh_coords = getAllLandmarksCoords([landmark.landmark for landmark in results.multi_face_landmarks][0])
                normalized_mesh = (face_mesh_coords - face_mesh_coords.min(axis=0)) / (face_mesh_coords.max(axis=0) - face_mesh_coords.min(axis=0))
                landmarks.append(normalized_mesh)
                labels.append(label)
    except Exception as e:
       print(e)

facemesh_images_old , labels_old = np.load('smile_images.npy') , np.load('smile_labels.npy')

landmarks = np.array(landmarks)
labels = np.array(labels)

print(landmarks.shape)
print(labels.shape)

facemesh_images = np.concatenate((landmarks , facemesh_images_old) , axis=0)
labels = np.concatenate((labels , labels_old[: , 1]) , axis=0)

print(facemesh_images.shape)
print(labels.shape)

np.save('smile_images.npy' , facemesh_images)
np.save('smile_labels.npy' , labels)