import cv2

print(f"[INFO] Using OpenCV version {cv2.__version__}.")

print("[INFO] Loading the model.")
cascade = cv2.CascadeClassifier("haarcascade_fontalface_default.xml")

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

    frame_gray = cv2.cvtColor(
        src=frame,
        code=cv2.COLOR_BGR2GRAY,
    )
    faces = cascade.detectMultiScale(frame_gray)
    for x, y, width, height in faces:
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(x + width, y + height),
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
