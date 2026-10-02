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

    # Green blobs smaller than this many pixels are treated as noise.
    minimum_blob_area = 50
    previous_position = None

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green = cv2.bitwise_and(frame, frame, mask=mask)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

        if largest_contour is None or cv2.contourArea(largest_contour) < minimum_blob_area:
            # Forget the last position so the next detection does not report a jump.
            print("lost")
            previous_position = None
        else:
            moments = cv2.moments(largest_contour)
            pen_x = round(moments["m10"] / moments["m00"])
            pen_y = round(moments["m01"] / moments["m00"])

            if previous_position is None:
                delta_x = 0
                delta_y = 0
            else:
                delta_x = pen_x - previous_position[0]
                delta_y = pen_y - previous_position[1]

            print(f"x={pen_x} y={pen_y} dx={delta_x} dy={delta_y}")
            previous_position = (pen_x, pen_y)
            cv2.circle(green, (pen_x, pen_y), 6, (0, 0, 255), -1)

        cv2.imshow(window, green)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
