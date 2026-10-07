"""
run with:

python ./isolate_green.py
"""

import cv2
import numpy as np

from util import draw_tracked_contour, largest_contour


def main():
    camera = cv2.VideoCapture(0)
    window = "Raw | Green only | Canvas - Q or Esc to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Width of the whole three-panel window, small enough to fit on a laptop screen.
    display_width = 1440

    # HSV range covers Grant's laptop camera (hue 45-65) and the Logitech C920 (hue 69-76).
    lower_green = (45, 60, 40)
    upper_green = (85, 255, 255)

    # Green blobs smaller than this many pixels are treated as noise.
    minimum_blob_area = 50
    previous_position = None

    # White drawing surface, created once the first frame tells us its size.
    canvas = None

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        if canvas is None:
            canvas = np.full(frame.shape, 255, dtype=np.uint8)

            # Three frames sit side by side, so the window is three frames wide and one frame tall.
            frame_height, frame_width = frame.shape[:2]
            display_height = round(display_width * frame_height / (3 * frame_width))
            cv2.resizeWindow(window, display_width, display_height)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green = cv2.bitwise_and(frame, frame, mask=mask)

        largest_green_contour = largest_contour(mask)

        if largest_green_contour is None or cv2.contourArea(largest_green_contour) < minimum_blob_area:
            # Forget the last position so the next detection does not report a jump.
            print("lost")
            previous_position = None
        else:
            moments = cv2.moments(largest_green_contour)
            pen_x = round(moments["m10"] / moments["m00"])
            pen_y = round(moments["m01"] / moments["m00"])

            if previous_position is None:
                delta_x = 0
                delta_y = 0
            else:
                delta_x = pen_x - previous_position[0]
                delta_y = pen_y - previous_position[1]

            # The pen counts as down whenever it is visible, so join this point to the last one.
            if previous_position is not None:
                cv2.line(canvas, previous_position, (pen_x, pen_y), (0, 0, 0), 3)

            print(f"x={pen_x} y={pen_y} dx={delta_x} dy={delta_y}")
            previous_position = (pen_x, pen_y)
            draw_tracked_contour(green, largest_green_contour)

        side_by_side = cv2.hconcat([frame, green, canvas])
        cv2.imshow(window, side_by_side)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
