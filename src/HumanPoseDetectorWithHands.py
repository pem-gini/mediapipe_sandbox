import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

import numpy as np
import cv2

import src.HumanPoseDetector as HPD
import src.Utils as utils

class HandRegion:
    def __init__(self, center, r):
        self.center = center
        self.r = r
    def cropFrom(self, image):
        cropped = image.copy()
        x,y = self.center
        # print(x,y,self.r)
        img_h, img_w, cannel = image.shape
        w1 = utils.clamp(0, int(x - self.r*1.75), img_w)
        w2 = utils.clamp(0, int(x + self.r*1.75), img_w)
        h1 = utils.clamp(0, int(y - self.r*1.75), img_h)
        h2 = utils.clamp(0, int(y + self.r*1.75), img_h)
        cropped_image = cropped[h1:h2, w1:w2]
        return cropped_image
    def zoomInto(self, image):
        ### adaptive linear zoom based on r with k = 10
        # k = 20.0
        # zoomfactor = clamp(2.0, (np.sqrt(1.0 / self.r) * k), 2.0) 
        ### adaptive linear zoom based on some math
        # zoom = r/110 + 5/11 + 6
        # zoomfactor = clamp(1.0, (self.r /110 + 5.0/11.0 + 6.0), 6.0) 
        ### adaptive quadratic zoom based on some math (r = pixelradius)
        #zoom = 0.00001r^2 - 0.0158r + 6.4
        zoomfactor = utils.clamp(1.0, (0.00001 * self.r ** 2 - 0.0158 * self.r + 6.4), 6.0) 
        ### do the zoom
        zoomed = utils.zoom_at(image, zoomfactor, self.center)
        return zoomed

class SpecialHandsOrientedHumanPoseDetector(HPD.HumanPoseDetector):
    def __init__(self):
        super().__init__()
    def update(self, image):        
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) ### numpy image to mpImage
        # detect poses in the input image.
        self.detector.detect_async(mpImage, self.stamp)
        # ### calculate time
        self.stamp = self.stamp + 1
        # process the result. In this case, visualize it.
        pose_landmarks = None
        resultImage, leftHand, rightHand = self.draw(mpImage.numpy_view(), self.results) ##mpImage back to numpy image
        maskImage = self.createMask(mpImage.numpy_view(), self.results)
        return resultImage, maskImage, leftHand, rightHand
    def draw(self, image, detection_result):
        ### call base class draw
        annotated_image = super().draw(image, detection_result)
        ### do something more
        hand_annotaded_image = annotated_image.copy()
        leftHandRegion = None
        rightHandRegion = None   
        if detection_result:
            img_h, img_w, cannel = annotated_image.shape
            ### crawl through landmarks, find the hand wrists and draw a rect around them
            #### from: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
            ###        https://camo.githubusercontent.com/7fbec98ddbc1dc4186852d1c29487efd7b1eb820c8b6ef34e113fcde40746be2/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67
            ###
            ### the left hand wrist index is: 15
            ### the right hand wrist index is: 16
            ### the left hand index index is: 19
            ### the right hand index index is: 20
            ### the left hand pinky index is: 17
            ### the right hand pinky index is: 18
            pose_landmarks_list = detection_result.pose_landmarks
            for idx in range(len(pose_landmarks_list)):
                pose_landmarks = pose_landmarks_list[idx]
                ### do rect around all landmarks
                coords = [_normalized_to_pixel_coordinates(l.x, l.y, img_w, img_h) for l in pose_landmarks]
                ### grab hand landmark values before filtering
                handLeftVal = None
                handRightVal = None
                handLeftR = None
                handRightR = None
                ### get point between wrist and pinky
                idxl1, idxl2 = (15, 17)
                idxr1, idxr2 = (16, 18)
                ### Left Hand
                try:
                    ### get coordinate between wrist and pinky
                    handLeftVal = np.mean([coords[idxl1],coords[idxl2]], axis=0).astype(int)
                    ### get dist between wrist and pinky
                    handLeftR = np.linalg.norm(np.array(coords[idxl1])-np.array(coords[idxl2]))
                except Exception as e:
                    ### hopeyfully only fires when coords[x] is None
                    # print(e)
                    pass
                ### Right Hand
                try:
                    handRightR = np.linalg.norm(np.array(coords[idxr1])-np.array(coords[idxr2]))
                    handRightVal = np.mean([coords[idxr1],coords[idxr2]], axis=0).astype(int)
                except Exception as e:
                    ### hopeyfully only fires when coords[x] is None
                    # print(e)
                    pass
                ### filter none values for non existent landmarks
                coords = np.array(list(filter(lambda i: i is not None, coords)))
                try:
                    ### not necessary, so we tr catch this
                    rect = cv2.minAreaRect(np.array(coords))
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(hand_annotaded_image,[box],0,(0,0,255),2)
                except:
                    pass
                ### do rect around both hands
                ### calculate radius based on distance between     
                if handLeftVal is not None and handLeftR is not None:
                    ### radius safety factor = 4.5 
                    handLeftR = int(handLeftR * 4.5)
                    ### radius minimum size = 100 pixel, max size = 500 pixels
                    handLeftR = utils.clamp(100, handLeftR, 500)
                    cv2.circle(hand_annotaded_image, handLeftVal, handLeftR, (255,0,0), 2)
                    leftHandRegion = HandRegion(handLeftVal, handLeftR)
                if handRightVal is not None and handRightR is not None:
                    ### radius safety factor = 4.5 
                    handRightR = int(handRightR * 4.5)
                    ### radius minimum size = 100 pixel, max size = 500 pixels
                    handRightR = utils.clamp(100, handRightR, 500)
                    cv2.circle(hand_annotaded_image, handRightVal, handRightR, (0,255,0), 2)
                    rightHandRegion = HandRegion(handRightVal, handRightR)

        return hand_annotaded_image, leftHandRegion, rightHandRegion
