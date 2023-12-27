import numpy as np
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
POINT_INDEXES =  list(mp_face_mesh.FACEMESH_LEFT_IRIS) + list(mp_face_mesh.FACEMESH_RIGHT_IRIS) # + list(mp_face_mesh.FACEMESH_IRISES) + list(mp_face_mesh.FACEMESH_CONTOURS)
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
  print(coords.shape)
  return coords


image = cv2.imread('image.png')

# Run MediaPipe Face Mesh.
with mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
  result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

face_mesh = [landmark.landmark for landmark in result.multi_face_landmarks][0]

coords = getAllLandmarksCoords(face_mesh)

for x, y, z in coords:
    # Scale x and y to image dimensions
    ix, iy = int(x * (image.shape[0])), int(y * (image.shape[1]))

    cv2.circle(image, (ix, iy), 1, (255, 255, 255), -1)  # Drawing a small circle for each point
  
cv2.imwrite('image.png' , image)