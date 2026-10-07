import cv2


def largest_contour(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return max(contours, key=cv2.contourArea, default=None)


def draw_tracked_contour(frame, contour):
    x, y, width, height = cv2.boundingRect(contour)
    center = (x + width // 2, y + height // 2)
    cv2.rectangle(
        frame, (x, y), (x + width - 1, y + height - 1),
        (0, 255, 255), 2,
    )
    cv2.circle(frame, center, 3, (0, 0, 255), -1)
