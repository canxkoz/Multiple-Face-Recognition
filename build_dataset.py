import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("[INFO] starting video stream...")
capture = cv2.VideoCapture(0)
total = 0

while True:
    ret, frame = capture.read()
    faces = detector.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("k"):
        cv2.imwrite("dataset/Can_{}.jpg".format(str(total).zfill(1)), frame[y:y+h, x:x+w])
        total += 1
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
capture.release()
cv2.destroyAllWindows()