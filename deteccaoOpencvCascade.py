import numpy as np
import cv2

cap = cv2.VideoCapture(0)
classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(40,40), maxSize=(400, 400), scaleFactor=1.2)
    for (x, y, l, a) in facesDetectadas:
      cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

    cv2.imshow('video1', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
