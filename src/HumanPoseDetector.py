import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

class HumanPoseDetector:
  def __init__(self):
    self.stamp = 0
    ### create an PoseLandmarker object.
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='modeltasks/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_segmentation_masks=True,
        result_callback=self.process_frame
    )
    self.detector = vision.PoseLandmarker.create_from_options(options)
    self.results = None

  def process_frame(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    self.results = result

  def update(self, image):
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) ### numpy image to mpImage
    # detect poses in the input image.
    self.detector.detect_async(mpImage, self.stamp)
    # ### calculate time
    self.stamp = self.stamp + 1
    # process the result. In this case, visualize it.
    pose_landmarks = None
    resultImage = self.draw(mpImage.numpy_view(), self.results) ##mpImage back to numpy image
    maskImage = self.createMask(mpImage.numpy_view(), self.results)
    return resultImage, maskImage
    
  def draw(self, image, detection_result):
    annotated_image = image.copy()
    # Loop through the detected poses to visualize.
    if detection_result:
      pose_landmarks_list = detection_result.pose_landmarks
      for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
  
  def createMask(self, image, detection_result):
    visualized_mask = image.copy()
    ### if detection exists, create propr mask image
    try:
      segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
      visualized_mask = cv2.convertScaleAbs(segmentation_mask, alpha=1.00)
      ### inflate mask
      kernel = np.ones((25,25),np.uint8)
      visualized_mask = cv2.dilate(visualized_mask, kernel, iterations=5)
    ### else, create a white image based on the input image
    except:
      img_h, img_w, cannel = image.shape
      visualized_mask = np.zeros((img_h, img_w), dtype=np.uint8)
      visualized_mask[:] = 255