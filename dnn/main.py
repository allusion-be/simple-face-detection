import cv2
import numpy as np

CONFIDENCE = 0.5

print(f"[INFO] Using OpenCV version {cv2.__version__}.")

print("[INFO] Loading the model.")
net = cv2.dnn.readNetFromCaffe(
    prototxt="deploy.prototxt",
    caffeModel="res10_300x300_ssd_iter_140000.caffemodel",
)

print("[INFO] Setting up window.")
cv2.namedWindow("", flags=cv2.WINDOW_GUI_NORMAL)

print("[INFO] Starting video capture.")
vc   = cv2.VideoCapture(0)
if (not vc.isOpened()):
  print('[ERROR] Could not read video capture.')

while True:
    rval, frame = vc.read()
    if not rval:
      break;

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
	    confidence = detections[0, 0, i, 2]
	    if confidence < CONFIDENCE:
		    continue
	    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
	    (startX, startY, endX, endY) = box.astype("int")
	    cv2.rectangle(
            img=frame,
            pt1=(startX, startY),
            pt2=(endX, endY),
            color=(0, 0, 255),
            thickness=2,
        )
	    cv2.putText(
            img=frame,
            text="{:.2f}%".format(confidence * 100),
            org=(startX, startY-10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.45,
            color=(0, 0, 255),
            thickness=2,
        )
    cv2.imshow("", frame)

    # Exit on ESC
    key = cv2.waitKey(20)
    if key == 27:
        break
    if cv2.getWindowProperty("", cv2.WND_PROP_VISIBLE) < 1:
        break

print("[INFO] Shutting down...")
vc.release()
cv2.destroyWindow("")
