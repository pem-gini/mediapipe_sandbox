import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np

import src.Utils as utils


class GestureRecognizer:
  def __init__(self, num_hands=2):
      self.mp_hands = mp.solutions.hands
      self.mp_drawing = mp.solutions.drawing_utils
      self.mp_drawing_styles = mp.solutions.drawing_styles
      self.stamp = 0
      # STEP 2: Create an GestureRecognizer object.
      VisionRunningMode = mp.tasks.vision.RunningMode
      base_options = python.BaseOptions(model_asset_path='modeltasks/gesture_recognizer.task')
      options = vision.GestureRecognizerOptions(base_options=base_options, 
                                                running_mode=VisionRunningMode.LIVE_STREAM,
                                                num_hands=num_hands,
                                                min_hand_detection_confidence=0.1,
                                                min_hand_presence_confidence=0.8,
                                                min_tracking_confidence=0.1,
                                                result_callback=self.process_frame
      )
      self.recognizer = vision.GestureRecognizer.create_from_options(options)
      self.results = None

  def process_frame(self, result: vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    self.results = result

  def update(self, image):
    if not utils.isLegitImage(image):
      return image
    ### flip image, so that hand sides are correct
    image = cv2.flip(image, 1)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) ### numpy image to mpImage
    # recognize gestures in the input image.
    self.recognizer.recognize_async(mpImage, self.stamp)
    ### calculate time
    self.stamp = self.stamp + 1
    # process the result. In this case, visualize it.
    top_gestures = []
    hands_landmarks = None
    handedness = None
    if self.results:
      # if recognition_result.gestures and recognition_result.hand_landmarks:
      for top in self.results.gestures:
        top_gestures.append(top[0])
      hands_landmarks = self.results.hand_landmarks
      handedness = self.results.handedness
    resultImage = self.draw(mpImage.numpy_view(), top_gestures, hands_landmarks, handedness) ##mpImage back to numpy image
    return resultImage
    
  def draw(self, image, top_gestures, hands_landmarks, handedness):
    annotated_image = image.copy()
    ### amount of hands to display 
    n = len(top_gestures)
    ### Display gestures and hand landmarks.
    if hands_landmarks and top_gestures:
      ### create skeleton and landmark
      for hand_landmarks in hands_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        ### draw landmark
        self.mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          self.mp_hands.HAND_CONNECTIONS,
          self.mp_drawing_styles.get_default_hand_landmarks_style(),
          self.mp_drawing_styles.get_default_hand_connections_style())
        ### draw title
        side_d = {}
        if handedness:
          ### remember which category (left, right is which index in the hand landmar list)
          side_d = {c[0].category_name : i for i, c in enumerate(handedness)}
          font = cv2.FONT_HERSHEY_SIMPLEX
          if "Left" in side_d:
            titleLeft = f"L:{top_gestures[side_d['Left']].category_name} ({top_gestures[side_d['Left']].score:.2f})"
            cv2.putText(annotated_image, titleLeft, (10,30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
          if "Right" in side_d:
            titleRight = f"R:{top_gestures[side_d['Right']].category_name} ({top_gestures[side_d['Right']].score:.2f})"
            cv2.putText(annotated_image, titleRight, (400,30), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return annotated_image

