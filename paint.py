from collections import deque
import numpy as np
import argparse
import cv2
import imutils
import time

print("Imported packages!")

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to optional video file")
ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size")
args = vars(ap.parse_args())

print("Controller are all set!")

# color range (red)
redLower = (160, 20, 70)
redUpper = (190, 255, 255)

# tracking variables
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

# camera setup
if not args.get("video", False):
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture(args["video"])

print("All ok video capture started!")

time.sleep(2.0)

# canvas
canvas = cv2.imread("white.jpg")

while True:
    ret, frame = vs.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # mask
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)

    # draw lines
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        if counter >= 10 and i == 1 and pts[-10] is not None:
            dX = -(pts[-10][0] - pts[i][0])
            dY = pts[-10][1] - pts[i][1]
            dirX, dirY = "", ""

            if abs(dX) > 30:
                dirX = "East" if np.sign(dX) == 1 else "West"

            if abs(dY) > 30:
                dirY = "North" if np.sign(dY) == 1 else "South"

            direction = f"{dirY}-{dirX}" if dirX and dirY else dirX or dirY

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.line(canvas, pts[i - 1], pts[i], (255, 0, 0), thickness)

    # text
    cv2.putText(frame, direction, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    cv2.putText(frame, f"dx: {dX}, dy: {dY}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # display
    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    counter += 1

    if key == ord("q"):
        cv2.imwrite("your_creation.jpg", canvas)
        print("Canvas saved!")
        break

# release camera
vs.release()

if not args.get("video", False):
    print("I didn't get any camera, Akshaya :<")
else:
    print("B-Bye Akshaya")

cv2.destroyAllWindows()