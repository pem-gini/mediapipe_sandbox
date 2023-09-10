


import cv2

from src.GestureRecognizer import GestureRecognizer
from src.HumanPoseDetectorWithHands import SpecialHandsOrientedHumanPoseDetector
import src.Utils as utils

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 20)
fps = 20
### instantiate gesture recognizer
r1 = GestureRecognizer(num_hands = 1)
r2 = GestureRecognizer(num_hands = 1)
### instantiate normal human ose detector
# d = HumanPoseDetector()
### or instantiate special human pose detector which alos draws rects around the hands
d = SpecialHandsOrientedHumanPoseDetector()
### read images from camera and feed into recognizer class
while cap.isOpened():
  success, image = cap.read()
  if success:
    ### crop image to nearest side, so that it gets rectangular and the focus is tn the middle of the image
    cropped = utils.cropToSmallestSide(image)
    # zoomed = zoom_at(cropped, 1.7) #1.3
    # filtered = cv2.GaussianBlur(zoomed,(5,5),0)
    human, mask, leftHandRegion, rightHandRegion = d.update(cropped)
    ### use the masked image to mask out everything except for the human body
    # maskedHuman = cv2.bitwise_and(human, human, mask=mask)
    # resultImage = r.update(maskedHuman)
    ### cut out the hands and feed the dedicated hand images into the gesture recognition
    if leftHandRegion != None:
      leftHandImage = leftHandRegion.zoomInto(cropped)
      # leftHandImage = leftHandRegion.cropFrom(leftHandImage)
      resultImage1 = r1.update(leftHandImage)
      ### resze for visualization
      if utils.isLegitImage(resultImage1):
        resultImage1 = cv2.resize(resultImage1, (500, 500), interpolation = cv2.INTER_AREA)
        utils.showInMovedWindow("LeftHandVideo",resultImage1, 50, 10)
    if rightHandRegion != None:
      rightHandImage = rightHandRegion.zoomInto(cropped)
      # rightHandImage = rightHandRegion.cropFrom(cropped)
      resultImage2 = r2.update(rightHandImage)
      ### resze for visualization
      if utils.isLegitImage(resultImage2):
        resultImage2 = cv2.resize(resultImage2, (500, 500), interpolation = cv2.INTER_AREA)
        utils.showInMovedWindow("RightHandVideo",resultImage2, 1350, 10)
    ### show result
    ### draw combine annotaed image
    utils.showInMovedWindow("AnnotatedVideo", human, 600, 10)
  ### exit condition is random keypress
  if cv2.waitKey(1) != -1:
    break
cv2.destroyAllWindows()
cap.release()
print("Done")