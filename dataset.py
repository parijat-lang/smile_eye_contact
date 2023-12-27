import mediapipe as mp
import numpy as np
import youtube_dl
import cv2 as cv
import labelbox
import shutil
import json
import cv2
import os

import os

mp_face_mesh = mp.solutions.face_mesh
POINT_INDEXES =  list(mp_face_mesh.FACEMESH_LIPS)
POINT_INDEXES = {element for tup in POINT_INDEXES for element in tup}

class videoAnnotations:
    def __init__(self):
        # Initialize class properties
        self.video_directory = "one_smile_videos"
        self.metadata_directory = "one_smile_metadata"
        self.export_parameters = {
            "attachments": True,
            "metadata_fields": True,
            "data_row_details": True,
            "project_details": True,
            "label_details": True,
            "performance_details": True,
            "interpolated_frames": True
        }
        self.filter_parameters = {
            "last_activity_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
            "label_created_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
            # Uncomment if needed:
            # "data_row_ids": ["<data_row_id>", "<data_row_id>"],
            # "batch_ids": ["<batch_id>", "<batch_id>"],
        }

    def initialize_project(self, project_id):
        # Initialize project related properties
        self.project_id = project_id
        self.client = labelbox.Client(os.environ['LB_API_KEY'])
        self.project = self.client.get_project(self.project_id)
        self.ontology_uid = self.project.ontology().uid

        # # Ensure required directories exist
        # if os.path.exists(self.video_directory):
        #     shutil.rmtree(self.video_directory)
        #     print('Deleted Existing video directory')
        # if os.path.exists(self.metadata_directory):
        #     shutil.rmtree(self.metadata_directory)
        #     print('Deleted Existing metadata directory')

        # os.makedirs(self.metadata_directory)
        # os.makedirs(self.video_directory)

    def download_video(self, url, path):
        # Configure YouTube downloader options
        ydl_options = {
            'format': 'best',
            'outtmpl': path,
        }

        # Use youtube_dl to download video
        with youtube_dl.YoutubeDL(ydl_options) as ydl:
            ydl.download([url])

    def get_project_details(self):
        # Export project details based on specified parameters and filters
        export_task = self.project.export_v2(params=self.export_parameters, filters=self.filter_parameters)
        export_task.wait_till_done()

        # Check for errors in export task
        if export_task.errors:
            print(export_task.errors)
            return None
        else:
            return export_task.result

    def export_for_annotation(self, email=None, data_row_ids= []):

        # Process results for a given email
        self.results = self.get_project_details()
        for result in self.results:
            if data_row_ids != []:
                if result['data_row']['id'] in data_row_ids:
                    data_row_id = result['data_row']['id']
                    # Save metadata to file
                    with open(os.path.join(self.metadata_directory, f'{data_row_id}.json'), 'w') as file:
                        json.dump(result['projects'][self.project_id], file, indent=4)
                    # Download associated video
                    self.download_video(result['data_row']['row_data'], os.path.join(self.video_directory, f'{data_row_id}.mp4'))

            else:
                if result['projects'][self.project_id]['project_details']['workflow_status'] == 'DONE':
                    if email == result['projects'][self.project_id]['labels'][0]['label_details']['created_by']:

                        data_row_id = result['data_row']['id']
                        # Save metadata to file
                        with open(os.path.join(self.metadata_directory, f'{data_row_id}.json'), 'w') as file:
                            json.dump(result['projects'][self.project_id], file, indent=4)
                        # Download associated video
                        self.download_video(result['data_row']['row_data'], os.path.join(self.video_directory, f'{data_row_id}.mp4'))
        print('Video and Annotations exported Successfully')

    def get_annotated_video(self , video_path , metadata_path):
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

        def get_frame_annotations(frame_data):
            frame_annotations = {}

            # We will use these to keep track of the smallest and largest 'left' values for left and right eyes respectively
            left_eye_left_value = float('inf')
            right_eye_left_value = -1

            for feature_id in frame_data['objects'].keys():
                object_ = frame_data['objects'][feature_id]

                if object_['name'] == 'eyes_bb':
                    # Check the 'left' value to determine if it's the left or right eye
                    if object_['bounding_box']['left'] < left_eye_left_value:
                        # This is the most left so far, so we consider it the left eye
                        left_eye_left_value = object_['bounding_box']['left']
                        frame_annotations['eyes_left_bb'] = object_['bounding_box']
                        if len([classification['radio_answer']['name'] for classification in object_['classifications']]) > 0:
                            frame_annotations['eyes_left'] = [classification['radio_answer']['name'] for classification in object_['classifications']][0]
                    elif object_['bounding_box']['left'] > right_eye_left_value:
                        # This is the most right so far, so we consider it the right eye
                        right_eye_left_value = object_['bounding_box']['left']
                        frame_annotations['eyes_right_bb'] = object_['bounding_box']
                        if len([classification['radio_answer']['name'] for classification in object_['classifications']]) > 0:
                            frame_annotations['eyes_right'] = [classification['radio_answer']['name'] for classification in object_['classifications']][0]
                else:
                    frame_annotations[object_['name']] = object_['bounding_box']
                    for classification in object_['classifications']:
                        frame_annotations[classification['name']] = classification['radio_answer']['name']

            return frame_annotations

        if os.path.exists(metadata_path):
            with open(metadata_path , 'r') as f:
                video_metadata = json.load(f)
        else:
            print("video metadata not found")
            return None

        facemesh_images = []
        labels = []
        # BaseOptions = mp.tasks.BaseOptions
        # FaceLandmarker = mp.tasks.vision.FaceLandmarker
        # FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        # VisionRunningMode = mp.tasks.vision.RunningMode
        # model_path = 'face_landmarker.task'
        # Create a face landmarker instance with the video mode:
        # options = FaceLandmarkerOptions(
        # base_options=BaseOptions(model_asset_path=model_path),
        # running_mode=VisionRunningMode.VIDEO)

        if os.path.exists(video_path):
            try:
                video = cv.VideoCapture(video_path)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_data = video_metadata['labels'][0]['annotations']['frames']
                frame_num = 1
                annotated_video = []
                all_frames = frame_data.keys()
                with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
                    for i in range(total_frames):
                        ret , frame = video.read()
                        if ret and str(frame_num) in all_frames:
                            frame_annotations = get_frame_annotations(frame_data[str(frame_num)])
                            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            if result and result.multi_face_landmarks:
                                all_face_mesh_coords = [landmark.landmark for landmark in result.multi_face_landmarks][0]
                                
                                coords = getAllLandmarksCoords(all_face_mesh_coords)
                                normalized_mesh = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))

                                if 'eyes_left' in frame_annotations.keys():
                                    labels.append([int(frame_annotations['eyes_left'] == 'eye_contact') , int(frame_annotations['mouth'] == 'smile')])
                                    facemesh_images.append(normalized_mesh)
                                elif 'eyes_right' in frame_annotations.keys():
                                    labels.append([int(frame_annotations['eyes_right'] == 'eye_contact') , int(frame_annotations['mouth'] == 'smile')])
                                    facemesh_images.append(normalized_mesh)

                        else:
                            break
                        frame_num += 1
            except Exception as e:
                print(e)
            return facemesh_images , labels
        else:
            print("Video not found")
            return None

    def get_dataset(self):
        facemesh_images = []
        labels = []
        video_files = os.listdir(self.video_directory)
        for video_file in video_files:
            data_row_id = os.path.splitext(video_file)[0]
            facemesh_images_temp , labels_temp = self.get_annotated_video(os.path.join(self.video_directory , video_file) , os.path.join(self.metadata_directory , data_row_id + ".json"))
            facemesh_images += facemesh_images_temp
            labels += labels_temp
            print(f'{data_row_id} Done')

        # shutil.rmtree(self.video_directory)
        # shutil.rmtree(self.metadata_directory)

        # df = pd.DataFrame.from_records(dataset)
        # df.ffill()

        # if os.path.exists(filename):
        #     prev_df = pd.read_pickle(filename)
        #     df = pd.concat([prev_df , df])

        # with open(filename , 'wb') as f:
        #     pickle.dump(df , f)

        print('DONE!!')

        return np.array(facemesh_images) , np.array(labels)

api_keys = ['eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHBqZGZtaTYwaHR1MDcyZzhvMjFkcTc2Iiwib3JnYW5pemF0aW9uSWQiOiJjbHBqZGZtaGowaHR0MDcyZ2Nycjk0eGVsIiwiYXBpS2V5SWQiOiJjbHBqZjNsMW8wMzJ6MDc0MjR0cng3ZzB4Iiwic2VjcmV0IjoiOGZhZWRlNTI2MjJmNTJkNjc5MDhjYTEwYzNiNThkZTAiLCJpYXQiOjE3MDEyNDEyMjMsImV4cCI6MjMzMjM5MzIyM30.Fy0JLnpWG3BVYKcH2ylDhvp7lctql6A76x5Iu9UIT3g']
project_ids = ['clpjf5sqq0l21070par09g8qy']
email_ids = ['deepanshuparijat@gmail.com']
facemesh_images_old , labels_old = np.load('smile_images.npy') , np.load('smile_labels.npy')
for api_key , project_id , email_id in zip(api_keys , project_ids , email_ids):

    os.environ['LB_API_KEY'] = api_key
    annotator = videoAnnotations()
    annotator.initialize_project(project_id)
    #annotator.export_for_annotation(email_id)
    facemesh_images , labels = annotator.get_dataset()
    facemesh_images = np.concatenate((facemesh_images , facemesh_images_old) , axis=0)
    labels = np.concatenate((labels , labels_old) , axis=0)
    print(facemesh_images.shape)
    print(labels.shape)
    np.save('smile_images.npy' , facemesh_images)
    np.save('smile_labels.npy' , labels)