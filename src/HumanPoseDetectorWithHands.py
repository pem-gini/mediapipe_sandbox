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


class SpecialHandsOrientedHumanPoseDetector(HPD.HumanPoseDetector):
    def __init__(self, roi_filtered=False, use_human_pose_mask=False):
        super().__init__(use_human_pose_mask=use_human_pose_mask)
        self.roiFiltered = roi_filtered
        self.faceRegion = utils.RegionOfInterest((0,0), 0)
        ### define regions of interest for hands
        ### dont allow radius to be too small, as we would later zoom too far in 
        ### set inflation, for some algorithms will need an inflated RoI
        self.leftHandRegion = utils.RegionOfInterest((0,0), 0, minR=5.0, inflation=5.5) 
        self.rightHandRegion = utils.RegionOfInterest((0,0), 0, minR=5.0, inflation=5.5)
    def update(self, image):        
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) ### numpy image to mpImage
        # detect poses in the input image.
        self.detector.detect_async(mpImage, self.stamp)
        # ### calculate time
        self.stamp = self.stamp + 1
        # process the result. In this case, visualize it.
        pose_landmarks = None
        resultImage, faceRegion, leftHandRegion, rightHandRegion = self.draw(mpImage.numpy_view(), self.results) ##mpImage back to numpy image
        ### creat special mask, where the hand regions are whitelisted, and the mask iteratons is smaller then usual
        maskImages = self.createMutipleMasks(mpImage.numpy_view(), self.results, iterations=1, whitelist_regions=[leftHandRegion, rightHandRegion])
        return resultImage, maskImages[0], maskImages[1], faceRegion, leftHandRegion, rightHandRegion
    def draw(self, image, detection_result):
        ### call base class draw
        annotated_image = super().draw(image, detection_result)
        ### do something more
        hand_annotaded_image = annotated_image.copy()
        leftHandRegion = None
        rightHandRegion = None
        faceRegion = None
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
                # idxl1, idxl2 = (15, 17)
                # idxr1, idxr2 = (16, 18)
                ### get point between index and pinky
                idxl1, idxl2 = (19, 17)
                idxr1, idxr2 = (20, 18)
                ### indices for face = 0,1,2,3,4,5,6,7,8,9,10
                faceIndices = [i for i in range(0,11)]
                faceCoords = [coords[i] for i in faceIndices]
                faceDetected = False if None in faceCoords else True
                faceCenter = coords[0] ### nose
                ### only continue the rest when a face was detected
                if faceDetected:
                    faceregioninflation = 2.0
                    faceR = np.linalg.norm(np.array(coords[0]) - np.array(coords[8])) ### face to edge of face
                    newFaceRegion = utils.RegionOfInterest(faceCenter, faceR, inflation=faceregioninflation)
                    self.faceRegion.update(newFaceRegion, kxy=0.6, kr=0.2)
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
                        handRightVal = np.mean([coords[idxr1],coords[idxr2]], axis=0).astype(int)
                        handRightR = np.linalg.norm(np.array(coords[idxr1])-np.array(coords[idxr2]))
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
                    ### filter hand regions
                    kxy = 1.0
                    kr = 0.2
                    if handLeftVal is not None and handLeftR is not None:
                        newLeftHandRegion = utils.RegionOfInterest(handLeftVal, handLeftR)
                        ### filter roi if necessary
                        self.leftHandRegion.update(newLeftHandRegion,  kxy=kxy, kr=kr)
                        ### draw region
                        ### radius minimum size = whatever the min size of the roi is, max size = 500 pixels
                        inflatedR = utils.clamp(self.leftHandRegion.minR,  int(self.leftHandRegion.getInflatedR()), 500)
                        cv2.circle(hand_annotaded_image, self.leftHandRegion.center, inflatedR, (255,0,0), 2)
                    if handRightVal is not None and handRightR is not None:
                        newRightHandRegion = utils.RegionOfInterest(handRightVal, handRightR)
                        ### filter roi if necessary
                        self.rightHandRegion.update(newRightHandRegion, kxy=kxy, kr=kr)
                        ### draw region
                        ### radius minimum size = whatever the min size of the roi is, max size = 500 pixels
                        inflatedR = utils.clamp(self.rightHandRegion.minR,  int(self.rightHandRegion.getInflatedR()), 500)
                        cv2.circle(hand_annotaded_image, self.rightHandRegion.center, inflatedR, (0,255,0), 2)

        return hand_annotaded_image, self.faceRegion, self.leftHandRegion, self.rightHandRegion
