"""
Calibrate the red LED thresholds for isolate_green.py.

run with:

python ./calibrate_red.py

Controls:
    left click on the LED   sample the pixels around the click and auto-fit the sliders
                            (click several times, or on several frames, to widen the fit)
    sliders                 fine tune by hand while watching the mask panel
    P                       print paste-ready values for isolate_green.py
    R                       forget all clicked samples
    Q or Esc                quit
"""

import cv2
import numpy as np

# Red sits at hue 0/180, where the range wraps around. Shifting hue by 90 moves red to the
# middle of the axis (about 90), so one slider range can describe it without wrapping.
HUE_SHIFT = 90

WINDOW = "Raw | Red mask - click the LED, P prints, Q quits"

# Half-width of the square patch sampled around each click, in pixels.
PATCH_RADIUS = 6

# Extra room added around the sampled values so the LED survives small lighting changes.
HUE_MARGIN = 3
SATURATION_MARGIN = 20
VALUE_MARGIN = 20


def shift_hue(hsv):
    shifted = hsv.copy()
    shifted[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + HUE_SHIFT) % 180
    return shifted


def to_original_ranges(hue_low, hue_high, sat_min, val_min):
    """Convert a shifted-hue range back into the two ranges isolate_green.py expects."""
    if hue_low >= HUE_SHIFT:
        # Entirely on the 0-89 side of the original hue axis.
        low = hue_low - HUE_SHIFT
        high = hue_high - HUE_SHIFT
        first = ((low, sat_min, val_min), (high, 255, 255))
        return first, first
    if hue_high < HUE_SHIFT:
        # Entirely on the 90-179 side of the original hue axis.
        low = hue_low + HUE_SHIFT
        high = hue_high + HUE_SHIFT
        first = ((low, sat_min, val_min), (high, 255, 255))
        return first, first
    # The range crosses red's wrap-around point, so it splits in two.
    first = ((0, sat_min, val_min), (hue_high - HUE_SHIFT, 255, 255))
    second = ((hue_low + HUE_SHIFT, sat_min, val_min), (179, 255, 255))
    return first, second


def largest_blob_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None
    biggest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(biggest), biggest


def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    # Starting values match the current red range in isolate_green.py, expressed in shifted hue.
    def nothing(_value):
        pass

    cv2.createTrackbar("Hue min", WINDOW, 84, 179, nothing)
    cv2.createTrackbar("Hue max", WINDOW, 96, 179, nothing)
    cv2.createTrackbar("Sat min", WINDOW, 130, 255, nothing)
    cv2.createTrackbar("Val min", WINDOW, 90, 255, nothing)

    state = {"shifted": None, "width": 0, "samples": []}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        shifted = state["shifted"]
        if shifted is None or x >= state["width"]:
            return

        height = shifted.shape[0]
        y0, y1 = max(0, y - PATCH_RADIUS), min(height, y + PATCH_RADIUS + 1)
        x0, x1 = max(0, x - PATCH_RADIUS), min(state["width"], x + PATCH_RADIUS + 1)
        patch = shifted[y0:y1, x0:x1].reshape(-1, 3)

        # The center of a bright LED is often blown out to white, which has low saturation
        # and a meaningless hue. Keep only the colored pixels so the fit is not dragged off.
        colored = patch[patch[:, 1] >= 60]
        if len(colored) == 0:
            print("No colored pixels near that click (LED may be blown out to white).")
            print("Click on the red halo around the LED, or lower the camera exposure.")
            return

        state["samples"].append(colored)
        fit_sliders()

    def fit_sliders():
        pixels = np.vstack(state["samples"])
        hue_low = int(np.percentile(pixels[:, 0], 2)) - HUE_MARGIN
        hue_high = int(np.percentile(pixels[:, 0], 98)) + HUE_MARGIN
        sat_min = int(np.percentile(pixels[:, 1], 2)) - SATURATION_MARGIN
        val_min = int(np.percentile(pixels[:, 2], 2)) - VALUE_MARGIN

        cv2.setTrackbarPos("Hue min", WINDOW, int(np.clip(hue_low, 0, 179)))
        cv2.setTrackbarPos("Hue max", WINDOW, int(np.clip(hue_high, 0, 179)))
        cv2.setTrackbarPos("Sat min", WINDOW, int(np.clip(sat_min, 0, 255)))
        cv2.setTrackbarPos("Val min", WINDOW, int(np.clip(val_min, 0, 255)))
        print(f"Fitted from {len(pixels)} LED pixels across {len(state['samples'])} click(s).")

    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        # Mirror to match isolate_green.py, so clicks land where you see the LED.
        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        shifted = shift_hue(hsv)
        state["shifted"] = shifted
        state["width"] = frame.shape[1]

        hue_low = cv2.getTrackbarPos("Hue min", WINDOW)
        hue_high = cv2.getTrackbarPos("Hue max", WINDOW)
        sat_min = cv2.getTrackbarPos("Sat min", WINDOW)
        val_min = cv2.getTrackbarPos("Val min", WINDOW)

        mask = cv2.inRange(shifted, (hue_low, sat_min, val_min), (hue_high, 255, 255))
        area, contour = largest_blob_area(mask)

        raw_view = frame.copy()
        if contour is not None:
            cv2.drawContours(raw_view, [contour], -1, (0, 255, 0), 2)

        # Live readout: with the LED on you want one blob, with the LED off you want zero.
        pass_pixels = int(cv2.countNonZero(mask))
        cv2.putText(raw_view, f"mask pixels: {pass_pixels}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(raw_view, f"largest blob: {int(area)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow(WINDOW, cv2.hconcat([raw_view, mask_view]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            state["samples"] = []
            print("Cleared all samples.")
        if key in (ord("p"), ord("P")):
            first, second = to_original_ranges(hue_low, hue_high, sat_min, val_min)
            suggested_min_area = max(5, int(area * 0.5))
            print()
            print("# ---- paste into isolate_green.py ----")
            print(f"lower_red_1 = {first[0]}")
            print(f"upper_red_1 = {first[1]}")
            print(f"lower_red_2 = {second[0]}")
            print(f"upper_red_2 = {second[1]}")
            print(f"minimum_red_area = {suggested_min_area}  # half the LED blob visible right now ({int(area)})")
            print("# --------------------------------------")
            print()
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()