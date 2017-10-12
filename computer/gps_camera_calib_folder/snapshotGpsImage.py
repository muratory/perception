import cv2
import urllib
import numpy as np

try:
  stream=urllib.urlopen('http://192.168.0.81:8020/?action=stream')

  bytes=''
  index_file = 1
  while True:
    bytes+=stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a != -1 and b != -1:
      jpg = bytes[a:b+2]
      bytes = bytes[b+2:]

      i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), -1)

      cv2.imshow('image_to_capture', i)

      key = cv2.waitKey(1) & 0xFF

      if key == 27:
        exit(0)
      elif key == ord("s"):
        # Convert to grayscale
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        if ret==True:
          file = 'calibration'+str(index_file)+'.bmp'
          print 'save file'+file
          cv2.imwrite(file, i)
          index_file+=1
        else:
          print 'Corners not found, please move and save again'

except IOError as e:
  print 'connection Error',str(e)

cv2.destroyAllWindows()
