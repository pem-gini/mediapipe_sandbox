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


class RegionOfInterest():
  def __init__(self, center, r, inflation=1.0):
    self.center = center
    self.r = r
    self.inflation = inflation
  def update(self, roi,  kxy=0.5, kr=0.5):
    if roi:
      self.inflation = roi.inflation
      x,y = self.center
      newx, newy = roi.center
      ### smal complementary weight filter
      x = int((1.0 - kxy) * x + (kxy * newx))
      y = int((1.0 - kxy) * y + (kxy * newy))
      self.center = (x,y)
      self.r = int((1.0 - kr) * self.r + (kr * roi.r))
  def cropFrom(self, image):
    cropped = image.copy()
    x,y = self.center
    x,y = self.center
    img_h, img_w = image.shape
    w1 = clamp(0, int(x - self.getInflatedR()), img_w)
    w2 = clamp(0, int(x + self.getInflatedR()), img_w)
    h1 = clamp(0, int(y - self.getInflatedR()), img_h)
    h2 = clamp(0, int(y + self.getInflatedR()), img_h)
    l = min(w1,w2)
    r = max(w1,w2)
    t = min(h1,h2)
    b = max(h1,h2)
    cropped_image = cropped[t:b, l:r]
    return cropped_image
  def zoomInto(self, image, f=None):
    zoomfactor = 1.0
    if f:
      ### more correct math after changing r definition
      zoomfactor = f(self)
    ### do the zoom
    zoomed = zoom_at(image, zoomfactor, self.center)
    return zoomed
  def setColor(self, image, color, inflate=True):
    ### circle
    x,y = self.center
    img_h, img_w = image.shape
    r = self.getInflatedR() if inflate else self.r
    image = cv2.circle(image, self.center, int(r), color, -1)
    return image
  def setRectColor(self, image, color):
    x,y = self.center
    img_h, img_w = image.shape
    w1 = clamp(0, int(x - self.r), img_w)
    w2 = clamp(0, int(x + self.r), img_w)
    h1 = clamp(0, int(y - self.r), img_h)
    h2 = clamp(0, int(y + self.r), img_h)
    l = min(w1,w2)
    r = max(w1,w2)
    t = min(h1,h2)
    b = max(h1,h2)
    image = cv2.rectangle(image, (l,t), (r, b), color, -1)
    return image
  def getInflatedR(self):
    return self.r * self.inflation