import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

import src.Utils as utils


class FaceDetector:
  def __init__(self, ca=0.5):
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='modeltasks/detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options, 
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        min_detection_confidence=ca,
                                        result_callback=self.process_frame
    )
    self.detector = vision.FaceDetector.create_from_options(options)
    self.results = None
    self.stamp = 0

  def process_frame(self, result: vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    self.results = result

  def update(self, image):
    if not utils.isLegitImage(image):
      return image
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) ### numpy image to mpImage
    # recognize gestures in the input image.
    self.detector.detect_async(mpImage, self.stamp)
    ### calculate time
    self.stamp = self.stamp + 1
    # process the result. In this case, visualize it.
    bboxes = []
    if self.results:
        image, bboxes = self.draw(mpImage.numpy_view(), self.results) ##mpImage back to numpy image
    return image, len(bboxes) > 0, bboxes 
    
  def draw(self, image, detection_result):
    annotated_image = image.copy()
    # Loop through the detected poses to visualize.
    bboxes = []
    if detection_result:
        height, width, _ = image.shape
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, (0,255,0), 3)
            bboxes.append(bbox)      
    return annotated_image, bboxes

