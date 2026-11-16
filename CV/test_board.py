import unittest

import cv2
import numpy as np

from CV.board import BoardCalibration
from HW.Plotter.gcode import CANVAS_WIDTH, CANVAS_HEIGHT, CanvasToPaper


class BoardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.board = BoardCalibration(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.corners = [(120, 80), (1080, 110), (1150, 920), (80, 950)]
        for corner in self.corners:
            self.board.add_corner(*corner)

    def test_camera_corners_fill_canvas_and_physical_board(self):
        expected_canvas = [(0, 0), (700, 0), (700, 600), (0, 600)]
        # Check the orientation with +X pointing left and +Y pointing down.
        expected_mm = [(275.6, -75), (-80, -75), (-80, 229.8), (275.6, 229.8)]
        paper = CanvasToPaper(CANVAS_WIDTH, CANVAS_HEIGHT, flip_x=True, flip_y=False)
        for corner, canvas, physical in zip(self.corners, expected_canvas, expected_mm):
            point = self.board.point(corner)
            np.testing.assert_allclose(point, canvas, atol=0.001)
            np.testing.assert_allclose(paper.point(*point), physical, atol=0.001)

    def test_perspective_corrects_interior_points(self):
        points = np.array([[(100, 200), (350, 300), (600, 500)]], dtype=np.float32)
        camera_points = cv2.perspectiveTransform(points, np.linalg.inv(self.board.transform))[0]
        for camera, expected in zip(camera_points, points[0]):
            np.testing.assert_allclose(self.board.point(camera), expected, atol=0.001)

    def test_one_inch_has_equal_scale_on_both_axes(self):
        paper = CanvasToPaper(CANVAS_WIDTH, CANVAS_HEIGHT)
        origin = np.array(paper.point(350, 300))
        for point in [(400, 300), (350, 350)]:
            self.assertAlmostEqual(np.linalg.norm(np.array(paper.point(*point)) - origin), 25.4)

    def test_outside_and_missing_pen_are_ignored(self):
        self.assertIsNone(self.board.point((0, 0)))
        self.assertIsNone(self.board.point(None))

    def test_reset_requires_new_corners(self):
        self.board.reset()
        self.assertEqual(self.board.corners, [])
        self.assertIsNone(self.board.point((350, 300)))
        for corner in self.corners[:3]:
            self.board.add_corner(*corner)
        self.assertIsNone(self.board.transform)

    def test_invalid_corner_order_can_be_retried(self):
        for corners in [self.corners[::-1], [self.corners[i] for i in [0, 2, 1, 3]], [(0, 0)] * 4]:
            self.board.reset()
            for corner in corners:
                self.board.add_corner(*corner)
            self.assertEqual(self.board.corners, [])
            self.assertIsNone(self.board.transform)
        for corner in self.corners:
            self.board.add_corner(*corner)
        self.assertIsNotNone(self.board.transform)


if __name__ == "__main__":
    unittest.main()
