"""Map four camera-view board corners onto a proportional drawing canvas."""
import cv2
import numpy as np


class BoardCalibration:
    CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.corners = []
        self.transform = None

    def add_corner(self, x, y):
        if self.transform is not None:
            return
        self.corners.append((x, y))
        if len(self.corners) < 4:
            return

        source = np.array(self.corners, dtype=np.float32)
        # Reject crossed, reversed, duplicate, or nearly collinear corners.
        if not cv2.isContourConvex(source) or cv2.contourArea(source, oriented=True) < 100:
            self.reset()
            return
        target = np.array([
            (0, 0), (self.width, 0),
            (self.width, self.height), (0, self.height),
        ], dtype=np.float32)
        self.transform = cv2.getPerspectiveTransform(source, target)

    def point(self, center):
        if self.transform is None or center is None:
            return None
        source = np.array([[center]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(source, self.transform)[0, 0]
        if not np.isfinite([x, y]).all():
            return None
        if not (-0.001 <= x <= self.width + 0.001 and -0.001 <= y <= self.height + 0.001):
            return None
        return float(np.clip(x, 0, self.width)), float(np.clip(y, 0, self.height))

    def draw(self, frame):
        for index, corner in enumerate(self.corners):
            cv2.circle(frame, corner, 6, (0, 255, 255), -1)
            cv2.putText(frame, str(index + 1), (corner[0] + 10, corner[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if self.transform is None:
            message = f"Click board {self.CORNER_NAMES[len(self.corners)]} corner (C to restart)"
        else:
            cv2.polylines(frame, [np.array(self.corners, dtype=np.int32)], True, (0, 255, 255), 2)
            message = "Board calibrated - C to recalibrate"
        cv2.putText(frame, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
