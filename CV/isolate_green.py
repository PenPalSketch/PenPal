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

    display_width = 1440

    lower_green = (45, 60, 40)
    upper_green = (85, 255, 255)

    # Red wraps around the hue axis in OpenCV (0-180), so it needs two ranges.
    # Slightly more sensitive than before: wider hue, lower saturation/value floors.
    lower_red_1 = (0, 101, 186)
    upper_red_1 = (5, 255, 255)
    lower_red_2 = (157, 101, 186)
    upper_red_2 = (179, 255, 255)

    minimum_blob_area = 50
    minimum_red_area = 5  # half the LED blob visible right now (0)

    # Debounce: how many consecutive frames must agree before the plotter state flips.
    red_frames_needed = 2
    red_lost_frames_needed = 3
    red_seen_count = 0
    red_missing_count = 0
    plotter_on = False  # starts OFF until red is seen

    previous_position = None
    canvas = None

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # Mirror the camera horizontally so moving right on screen is moving right in real life.
        frame = cv2.flip(frame, 1)

        # Where the green pen is this frame (None if not found). Used for the hover cursor.
        current_position = None

        if canvas is None:
            canvas = np.full(frame.shape, 255, dtype=np.uint8)
            frame_height, frame_width = frame.shape[:2]
            display_height = round(display_width * frame_height / (3 * frame_width))
            cv2.resizeWindow(window, display_width, display_height)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Green pen detection ---
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green = cv2.bitwise_and(frame, frame, mask=mask)

        # --- Red LED detection ---
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red_1, upper_red_1),
            cv2.inRange(hsv, lower_red_2, upper_red_2),
        )

        largest_red_contour = largest_contour(red_mask)
        red_visible = (
            largest_red_contour is not None
            and cv2.contourArea(largest_red_contour) >= minimum_red_area
        )

        # Debounced state: red seen -> plotter ON, red not seen -> plotter OFF.
        if red_visible:
            red_seen_count += 1
            red_missing_count = 0
            if red_seen_count >= red_frames_needed and not plotter_on:
                plotter_on = True
                print("RED detected: plotter ON")
            draw_tracked_contour(frame, largest_red_contour)
        else:
            red_missing_count += 1
            red_seen_count = 0
            if red_missing_count >= red_lost_frames_needed and plotter_on:
                plotter_on = False
                print("No red: plotter OFF")

        largest_green_contour = largest_contour(mask)

        if largest_green_contour is None or cv2.contourArea(largest_green_contour) < minimum_blob_area:
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

            # Only draw and report motion while the plotter is on.
            if plotter_on:
                if previous_position is not None:
                    cv2.line(canvas, previous_position, (pen_x, pen_y), (0, 0, 0), 3)
                print(f"x={pen_x} y={pen_y} dx={delta_x} dy={delta_y}")

            # Always keep tracking so turning the plotter on does not cause a jump.
            previous_position = (pen_x, pen_y)
            current_position = (pen_x, pen_y)
            draw_tracked_contour(green, largest_green_contour)

        status_text = "PEN DOWN" if plotter_on else "PEN UP"
        status_color = (0, 200, 0) if plotter_on else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Draw the cursor on a copy so it never gets painted into the drawing itself.
        canvas_view = canvas.copy()
        if current_position is not None:
            cx, cy = current_position
            cv2.circle(canvas_view, (cx, cy), 10, status_color, 2)
            cv2.line(canvas_view, (cx - 16, cy), (cx + 16, cy), status_color, 1)
            cv2.line(canvas_view, (cx, cy - 16), (cx, cy + 16), status_color, 1)

        side_by_side = cv2.hconcat([frame, green, canvas_view])
        cv2.imshow(window, side_by_side)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()