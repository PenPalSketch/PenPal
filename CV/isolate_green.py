"""
run with:

python ./isolate_green.py
"""

import cv2


def main():
    camera = cv2.VideoCapture(0)
    window = "Green only - Q or Esc to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # HSV range sampled from Grant's laptop camera, should be updated later.
    lower_green = (45, 60, 40)
    upper_green = (65, 255, 255)

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow(window, green)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
