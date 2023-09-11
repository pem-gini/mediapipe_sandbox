import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

import src.Utils as utils

class HumanPoseDetector:
  def __init__(self, use_human_pose_mask=False):
    self.stamp = 0
    self.use_human_pose_mask = use_human_pose_mask
    ### create an PoseLandmarker object.
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='modeltasks/pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_segmentation_masks=self.use_human_pose_mask,
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
  
  ### create a single mask image from multiple sources
  def createMask(self, image, detection_result, iterations=4, whitelist_regions : list[utils.RegionOfInterest] = []):
    visualized_mask = image.copy()
    ### if detection exists, create proper mask images
    try:
      ### use human pose in mask
      if self.use_human_pose_mask:
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = cv2.convertScaleAbs(segmentation_mask, alpha=1.00)
      else:
        h,w, c = visualized_mask.shape
        visualized_mask = np.zeros((h,w), dtype = np.uint8)
      ### inflate mask
      kernel = np.ones((25,25),np.uint8)
      visualized_mask = cv2.dilate(visualized_mask, kernel, iterations=iterations)
      for roi in whitelist_regions:
        if roi:
          visualized_mask = roi.setColor(visualized_mask, (255,255,255))
    ### else, create a white image based on the input image
    except Exception as e:
      print(e)
      img_h, img_w, cannel = image.shape
      visualized_mask = np.zeros((img_h, img_w), dtype=np.uint8)
      visualized_mask[:] = 255
    return visualized_mask
  
  ### create multiple images provided by whitelist regions
  def createMutipleMasks(self, image, detection_result, whitelist_regions : list[utils.RegionOfInterest], iterations=4):
    maskImageList = []
    for roi in whitelist_regions:
      if roi:
        visualized_mask = image.copy()
        try:
          ### use human pose in mask
          if self.use_human_pose_mask:
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            visualized_mask = cv2.convertScaleAbs(segmentation_mask, alpha=1.00)
          else:
            h,w, c = visualized_mask.shape
            visualized_mask = np.zeros((h,w), dtype = np.uint8)
          ### inflate mask
          kernel = np.ones((25,25),np.uint8)
          visualized_mask = cv2.dilate(visualized_mask, kernel, iterations=iterations)
          visualized_mask = roi.setColor(visualized_mask, (255,255,255))
        ### else, create a white image based on the input image
        except Exception as e:
          print(e)
          img_h, img_w, cannel = image.shape
          visualized_mask = np.zeros((img_h, img_w), dtype=np.uint8)
          visualized_mask[:] = 255
        maskImageList.append(visualized_mask)
    return maskImageList
