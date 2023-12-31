


import cv2
import collections
import numpy as np
from enum import Enum

from src.GestureRecognizer import GestureRecognizer
from src.HumanPoseDetectorWithHands import SpecialHandsOrientedHumanPoseDetector
from src.FaceDetector import FaceDetector
import src.Utils as utils


### zoom in on the hands once detected 
mode = 'complex'
### only use gesture recognizer without anything else
# mode = 'simple'

#############################################################################################

def zoomFuncHands(roi):
  # return utils.clamp(1.0, -1/5 * roi.r + 8, 10.0) # big -1  zoom
  return utils.clamp(1.0, -1/3.5 * roi.getR() + 8, 8.0) # big -2  zoom
  # return utils.clamp(1.0, -1/6 * roi.r + 8, 10.0) # big zoom
def zoomFuncFace(roi):
  # return utils.clamp(1.0, -1/6 * roi.getR() + 10, 10.0) # big zoom
  return utils.clamp(1.0, -1/7.5 * roi.getR() + 10, 10.0) # big zoom


class FaceDetectionFilter:
  class State(Enum):
      FACE_PRESENT = 1
      FACE_MISSING = 0
  def __init__(self, filterTime, fps, threshold_negative_to_positive, treshold_positive_to_negative):
    self.filterTime = filterTime
    self.bufsize = int(fps * self.filterTime)
    self.tresholdNegativeToPositive = threshold_negative_to_positive
    self.tresholdPositiveToNegative  = treshold_positive_to_negative
    self.buf = collections.deque(maxlen=self.bufsize)
    self.state = FaceDetectionFilter.State.FACE_MISSING
  def update(self, faceDetectionSuccess):
    self.buf.append(faceDetectionSuccess)
    visibility = np.mean(self.buf)
    ### when we currently have no face
    if self.state == FaceDetectionFilter.State.FACE_MISSING:
      ### return true, whenever some faces face has been seen depending on the "positive" hysteresis threshold
      if visibility >= self.tresholdNegativeToPositive:
        self.state = FaceDetectionFilter.State.FACE_PRESENT
        self.buf.extend([True for x in range(self.bufsize)])
        return True
      else:
        return False
    elif self.state == FaceDetectionFilter.State.FACE_PRESENT:
      ### return true, whenever some faces face has been seen depending on the "negative" hysteresis threshold
      if visibility <= self.tresholdPositiveToNegative:
        self.state = FaceDetectionFilter.State.FACE_MISSING
        self.buf.extend([False for x in range(self.bufsize)])
        return False
      else: 
        return True
  def isFaceVisible(self):
    return self.state == FaceDetectionFilter.State.FACE_PRESENT

    
def complex_main():
  cap = cv2.VideoCapture(0)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  fps = cap.get(cv2.CAP_PROP_FPS)
  ### instantiate gesture recognizer
  r1 = GestureRecognizer(num_hands=1, ca=0.1, cb=0.1, cc=0.1)
  r2 = GestureRecognizer(num_hands=1, ca=0.1, cb=0.1, cc=0.1)
  ### instantiate normal human ose detector
  # d = HumanPoseDetector()
  ### or instantiate special human pose detector which alos draws rects around the hands
  d = SpecialHandsOrientedHumanPoseDetector(roi_filtered=True, use_human_pose_mask=False)
  ### instantiate face detector
  f = FaceDetector(ca=0.8)
  ### create ringbuffer for filtering faceDetectionSuccesses a little
  faceSuccessFilter = FaceDetectionFilter(
    2.0, ### X seconds
    fps, 
    0.05, ### 5% of successful detections to accept in the last X seconds when face is not visible (before getting visible)
    0.25  ### 50% of successful detections to accept in the last X seconds when face is visible (before getting not visible)
  )
  ### read images from camera and feed into recognizer classs
  while cap.isOpened():
    success, image = cap.read()
    if success:
      ### crop image to nearest side, so that it gets rectangular and the focus is tn the middle of the image
      # cropped = utils.cropToSmallestSide(image)
      # zoomed = zoom_at(cropped, 1.7) #1.3
      # filtered = cv2.GaussianBlur(zoomed,(5,5),0)
      ### do human pose detection
      human, maskImageLeft, maskImageRight, faceRegion, leftHandRegion, rightHandRegion = d.update(image) ### image
      ### zoom & check face region
      ### find face first, only do the rest if a face is visible
      faceImageZoomed = faceRegion.zoomInto(image, f=zoomFuncFace)
      if utils.isLegitImage(faceImageZoomed):
        faceImage, successfullFaceDetection, faceBoxes = f.update(faceImageZoomed)
        faceImage = cv2.resize(faceImage, (400, 300), interpolation = cv2.INTER_AREA)
        utils.showInMovedWindow("FaceVideo",faceImage, 700, 10)
        ### filter the face detection slightly
        faceSuccessFilter.update(successfullFaceDetection)
        # print(faceSuccessFilter.isFaceVisible())
        #if successfullFaceDetection:
        if faceSuccessFilter.isFaceVisible():
          ### use the masked image to mask out everything except for the human body
          maskedHumanLeft = cv2.bitwise_and(image, image, mask=maskImageLeft)
          maskedHumanRight = cv2.bitwise_and(image, image, mask=maskImageRight)
          ## cut out the hands and feed the dedicated hand images into the gesture recognition
          if leftHandRegion != None:
            #leftHandImage = leftHandRegion.zoomInto(cropped)
            leftHandImage = leftHandRegion.zoomInto(maskedHumanLeft, f=zoomFuncHands)
            # leftHandImage = leftHandRegion.cropFrom(leftHandImage)
            resultImage1 = r1.update(leftHandImage)
            ### resze for visualization
            if utils.isLegitImage(resultImage1):
              resultImage1 = cv2.resize(resultImage1, (500, 700), interpolation = cv2.INTER_AREA)
              utils.showInMovedWindow("LeftHandVideo",resultImage1, 50, 10)
          if rightHandRegion != None:
            # rightHandImage = rightHandRegion.cropFrom(cropped)
            rightHandImage = rightHandRegion.zoomInto(maskedHumanRight, f=zoomFuncHands)
            resultImage2 = r2.update(rightHandImage)
            ### resize for visualization
            if utils.isLegitImage(resultImage2):
              resultImage2 = cv2.resize(resultImage2, (500, 700), interpolation = cv2.INTER_AREA)
              utils.showInMovedWindow("RightHandVideo",resultImage2, 1350, 10)
      ### show result
      ### resze for visualization
      human = cv2.resize(human, (800, 600), interpolation = cv2.INTER_AREA)
      ### draw combine annotaed image
      utils.showInMovedWindow("AnnotatedVideo", human, 600, 600)
    ### exit condition is random keypress
    if cv2.waitKey(1) != -1:
      break
  cv2.destroyAllWindows()
  cap.release()

def simple_main():
  cap = cv2.VideoCapture(0)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  cap.set(cv2.CAP_PROP_FPS, 20)
  fps = 20
  ### instantiate gesture recognizer
  r = GestureRecognizer(num_hands=2, ca=0.5, cb=0.5, cc=0.5)
  ### read images from camera and feed into recognizer class
  while cap.isOpened():
    success, image = cap.read()
    if success:
      ### crop image to nearest side, so that it gets rectangular and the focus is tn the middle of the image
      cropped = utils.cropToSmallestSide(image)
      # zoomed = zoom_at(cropped, 1.7) #1.3
      # filtered = cv2.GaussianBlur(zoomed,(5,5),0)
      resultImage = r.update(cropped)
      ### show result
      ### draw combine annotaed image
      utils.showInMovedWindow("AnnotatedVideo", resultImage, 600, 10)
    ### exit condition is random keypress
    if cv2.waitKey(1) != -1:
      break
  cv2.destroyAllWindows()
  cap.release()

if __name__ == "__main__":
  if mode == "complex":
    complex_main()
  else:
    simple_main()
  print("Done")
