import cv2
import dwpose
import time

import onnxruntime

device=onnxruntime.get_device()
print(device)
if device=="GPU":
    onnxruntime.preload_dlls()
detector = dwpose.DWposeDetector(device=device)

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    cv2.imshow('camara', frame)
    
    t1=time.time()
    skeleton = detector(frame)
    t2=time.time()
    cv2.imshow("dwpose", skeleton)
    print(t2-t1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
