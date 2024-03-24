import os
import cv2
import mediapipe as mp
import numpy as np

class FaceMesh:
    def __init__(self, face_mesh):
        self.face_mesh = face_mesh
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def draw(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        # background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         mp.solutions.drawing_utils.draw_landmarks(
        #             image=background,
        #             landmark_list=face_landmarks,
        #             connections=mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = [(int(landmark.x * image.shape[0]), int(landmark.y * image.shape[1])) for landmark in face_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(background, hull, 0)
        return background


import mediapipe as mp
from facemesh import FaceMesh

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

mesher = FaceMesh(face_mesh)

for root, dirs, files in os.walk("/workspace/comfyui_controlnet_aux/src/HQ_origin"):
        for file in files:
            if file.lower().endswith('.jpg'):
                filepath = os.path.join(root, file)
                
                # image_path = "/workspace/comfyui_controlnet_aux/src/HQ_origin/batch_0/000000.jpg"

                new_path = filepath.replace("HQ_origin", "HQ_facemesh")
                parent_new, tail = os.path.split(new_path)
                try:
                    os.makedirs(parent_new)
                    print(f"Directory {parent_new} created")
                except FileExistsError:
                    print(f"Directory {parent_new} already exists")
                except Exception as e:
                    print(f"Error creating directory {parent_new}: {e}")
                result_path = mesher.draw(filepath, parent_new)
                print(result_path)
