


import cv2

from src.GestureRecognizer import GestureRecognizer
from src.HumanPoseDetectorWithHands import SpecialHandsOrientedHumanPoseDetector
import src.Utils as utils


### zoom in on the hands once detected 
mode = 'complex'
### only use gesture recognizer without anything else
# mode = 'simple'

#############################################################################################

def complex_main():
  cap = cv2.VideoCapture(0)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  cap.set(cv2.CAP_PROP_FPS, 20)
  fps = 20
  ### instantiate gesture recognizer
  r1 = GestureRecognizer(num_hands=1, ca=0.1, cb=0.1, cc=0.1)
  r2 = GestureRecognizer(num_hands=1, ca=0.1, cb=0.1, cc=0.1)
  ### instantiate normal human ose detector
  # d = HumanPoseDetector()
  ### or instantiate special human pose detector which alos draws rects around the hands
  d = SpecialHandsOrientedHumanPoseDetector(roi_filtered=True, use_human_pose_mask=False)
  ### read images from camera and feed into recognizer class
  while cap.isOpened():
    success, image = cap.read()
    if success:
      ### crop image to nearest side, so that it gets rectangular and the focus is tn the middle of the image
      # cropped = utils.cropToSmallestSide(image)
      # zoomed = zoom_at(cropped, 1.7) #1.3
      # filtered = cv2.GaussianBlur(zoomed,(5,5),0)
      human, maskImageLeft, maskImageRight, leftHandRegion, rightHandRegion = d.update(image)
      ### use the masked image to mask out everything except for the human body
      maskedHumanLeft = cv2.bitwise_and(image, image, mask=maskImageLeft)
      maskedHumanRight = cv2.bitwise_and(image, image, mask=maskImageRight)
      ## cut out the hands and feed the dedicated hand images into the gesture recognition
      if leftHandRegion != None:
        #leftHandImage = leftHandRegion.zoomInto(cropped)
        leftHandImage = leftHandRegion.zoomInto(maskedHumanLeft)
        # leftHandImage = leftHandRegion.cropFrom(leftHandImage)
        resultImage1 = r1.update(leftHandImage)
        ### resze for visualization
        if utils.isLegitImage(resultImage1):
          resultImage1 = cv2.resize(resultImage1, (500, 500), interpolation = cv2.INTER_AREA)
          utils.showInMovedWindow("LeftHandVideo",resultImage1, 50, 10)
      if rightHandRegion != None:
        # rightHandImage = rightHandRegion.cropFrom(cropped)
        rightHandImage = rightHandRegion.zoomInto(maskedHumanRight)
        resultImage2 = r2.update(rightHandImage)
        ### resze for visualization
        if utils.isLegitImage(resultImage2):
          resultImage2 = cv2.resize(resultImage2, (500, 500), interpolation = cv2.INTER_AREA)
          utils.showInMovedWindow("RightHandVideo",resultImage2, 1350, 10)
      ### show result
      ### resze for visualization
      human = cv2.resize(human, (800, 600), interpolation = cv2.INTER_AREA)
      ### draw combine annotaed image
      utils.showInMovedWindow("AnnotatedVideo", human, 600, 10)
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
