import unittest
from unittest.mock import patch

import cv2
import numpy as np

from CV.server import CanvasCalibrator, find_canvas_corners, get_canvas_transform


class BlueDotCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((600, 800, 3), 255, np.uint8)
        self.positions = [(150, 150), (650, 140), (660, 450), (140, 460)]
        for position in self.positions:
            cv2.circle(self.frame, position, 5, (255, 0, 0), -1)

    def test_finds_blue_dots_in_paper_order(self):
        corners = find_canvas_corners(self.frame)

        self.assertIsNotNone(corners)
        expected = np.array(self.positions)
        np.testing.assert_allclose(corners, expected, atol=1)

    def test_requires_all_four_dots(self):
        cv2.circle(self.frame, self.positions[-1], 7, (255, 255, 255), -1)
        self.assertIsNone(find_canvas_corners(self.frame))

    def test_ignores_blue_drawing_and_extra_small_dot(self):
        cv2.line(self.frame, (250, 250), (550, 350), (255, 0, 0), 8)
        cv2.circle(self.frame, (400, 300), 5, (255, 0, 0), -1)

        corners = find_canvas_corners(self.frame)
        expected = np.array(self.positions)
        np.testing.assert_allclose(corners, expected, atol=1)

    def test_detects_dark_and_uneven_corner_dots(self):
        frame = np.full((600, 800, 3), 255, np.uint8)
        colours = [(35, 5, 5), (45, 8, 8), (180, 35, 20), (255, 30, 20)]
        sizes = [(4, 5), (5, 4), (6, 8), (7, 5)]
        for position, colour, size in zip(self.positions, colours, sizes):
            cv2.ellipse(frame, position, size, 0, 0, 360, colour, -1)

        corners = find_canvas_corners(frame)
        np.testing.assert_allclose(corners, self.positions, atol=1)

    def test_transform_uses_physical_board_dimensions(self):
        corners = find_canvas_corners(self.frame)
        transform, size = get_canvas_transform(corners)

        self.assertEqual(size, (740, 580))
        mapped = cv2.perspectiveTransform(corners.reshape(1, 4, 2), transform)[0]
        expected = np.array([[0, 0], [739, 0], [739, 579], [0, 579]])
        np.testing.assert_allclose(mapped, expected, atol=0.01)

    def test_calibration_stays_frozen_after_locking(self):
        original = np.array(self.positions, np.float32)
        moved = original + 50
        calibrator = CanvasCalibrator(duration_seconds=0, minimum_samples=1)
        with patch("CV.server.find_canvas_corners", side_effect=[original, moved]) as finder:
            first = calibrator.update(self.frame)
            second = calibrator.update(self.frame)

        np.testing.assert_array_equal(first, original)
        np.testing.assert_array_equal(second, original)
        finder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
