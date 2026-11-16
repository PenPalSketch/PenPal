"""Place generated strokes inside the user's drawing without distorting them."""
import cv2
import numpy as np


def fit_to_drawing(generated, drawing, width, height):
    if not generated or not drawing:
        return generated

    outlines = []
    for path in drawing:
        points = np.asarray(path, dtype=np.float32)
        if len(points) < 4:
            continue
        span = np.linalg.norm(np.ptp(points, axis=0))
        if np.linalg.norm(points[-1] - points[0]) <= max(5, 0.1 * span):
            outlines.append(points)

    # Use the largest closed outline so interior details don't replace the boundary.
    if outlines:
        outline = max(outlines, key=cv2.contourArea)
    else:
        points = np.asarray([point for path in drawing for point in path], dtype=np.float32)
        if not len(points):
            return generated
        left, top = points.min(axis=0)
        right, bottom = points.max(axis=0)
        outline = np.array([(left, top), (right, top), (right, bottom), (left, bottom)], dtype=np.float32)
    if cv2.contourArea(outline) < 1:
        return generated

    # Find an interior centre, including for concave outlines. Padding keeps the
    # canvas edge outside the mask when an outline touches that edge.
    raster_scale = min(1.0, 1024 / max(width, height))
    mask = np.zeros((round(height * raster_scale) + 3,
                     round(width * raster_scale) + 3), dtype=np.uint8)
    contour = np.rint(outline * raster_scale).astype(np.int32) + 1
    cv2.fillPoly(mask, [contour], 255)
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    row, column = np.unravel_index(np.argmax(distances), distances.shape)
    centre = np.array([(column - 1) / raster_scale, (row - 1) / raster_scale])
    radius = min(cv2.pointPolygonTest(outline, tuple(centre), True),
                 centre[0], width - centre[0], centre[1], height - centre[1])
    if radius <= 0:
        return generated

    points = np.asarray([point for path in generated for point in path], dtype=np.float64)
    source_centre = (points.min(axis=0) + points.max(axis=0)) / 2
    source_radius = np.linalg.norm(points - source_centre, axis=1).max()
    # Every segment stays in this interior circle, including across concave gaps.
    # Use one scale for both axes and keep a 10% clearance from the outline.
    scale = min(1.0, 0.9 * radius / max(source_radius, 1e-9))
    return [((np.asarray(path) - source_centre) * scale + centre).tolist()
            for path in generated]
