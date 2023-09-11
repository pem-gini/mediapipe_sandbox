import cv2
import numpy as np

def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))

def showInMovedWindow(winname, img, x, y):
    w,h,d = img.shape
    if w > 0 and h > 0:
        cv2.namedWindow(winname)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)
        cv2.imshow(winname,img)

def cropToSmallestSide(image):
  if not isLegitImage(image):
    return image
  img_h, img_w, cannel = image.shape
  frame_size = min(img_w, img_h)
  crop_w = max((img_w - frame_size) // 2, 0)
  # print("Cropping on width :", crop_w)
  crop_h = max((img_h - frame_size) // 2, 0)
  # print("Cropping on height :", crop_h)
  pad_w = max((frame_size - img_w) // 2, 0)
  # print("Padding on width :", pad_w)
  pad_h = max((frame_size - img_h) // 2, 0)
  # print("Padding on height :", pad_h)
  new_img_h = new_img_w = frame_size
  # print(f"New Frame working size: {new_img_w}x{new_img_h}")
  ### the .copy is important, as slicing up he image makes it unviable for mp.Image constructor
  cropped = image[crop_h:img_h-crop_h, crop_w:img_w-crop_w].copy()
  return cropped

def zoom_at(img, zoom, coord=None):
  """
  Simple image zooming without boundary checking.
  Centered at "coord", if given, else the image center.
  img: numpy.ndarray of shape (h,w,:)
  zoom: float
  coord: (float, float)
  """
  if not isLegitImage(img):
    return img.copy()
  h, w, c = img.shape
  # Translate to zoomed coordinates
  hzoom, wzoom, _ = [ zoom * i for i in img.shape ]
  if coord is None: cx, cy = wzoom/2, hzoom/2
  else: cx, cy = [ zoom*c for c in coord ]
  img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
  ### for some reason, the cant be negative or resize fails ... but a too large l and r work? Debug later how this can be approached
  t = max(0, int(round(cy - hzoom/zoom * .5)))
  b = max(0, int(round(cy + hzoom/zoom * .5)))
  l = max(0, int(round(cx - wzoom/zoom * .5)))
  r = max(0, int(round(cx + wzoom/zoom * .5)))
  # print(t,b,l,r)
  zoomedImg = img[t:b, l:r, :].copy()
  return zoomedImg

def isLegitImage(img):
  img_h, img_w, cannel = img.shape
  return img is not None and (img_h > 0 and img_w > 0)