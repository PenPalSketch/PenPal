import asyncio
import json
import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from starlette.requests import Request

from HW.Plotter.gcode import CanvasToPaper, strokes_to_gcode
from HW.Plotter.simulate import simulate
from HW.Plotter.send import FakeGrbl, send_gcode
from ML.sketch_support import lines_to_strokes, simplify_lines
from Web.inference import complete, complete_drawing
from Web.placement import fit_to_drawing


def circle():
    return [[300 + 140 * math.cos(angle), 300 + 140 * math.sin(angle)]
            for angle in np.linspace(0, 2 * math.pi, 65)]


class RecordedModel:
    simplify_lines = staticmethod(simplify_lines)
    lines_to_stroke = staticmethod(lines_to_strokes)

    def __init__(self, anchor):
        self.samples = iter([
            [520 - anchor[0], 400 - anchor[1], 1, 0, 0],
            [100, 0, 1, 0, 0], [0, 60, 0, 1, 0],
            [80, 30, 1, 0, 0], [40, 40, 0, 1, 0], [0, 0, 0, 0, 1],
        ])

    def zero_input(self): return None
    def zeroState(self): return None
    def update(self, sample, state): return None
    def updateStrokes(self, seed, state, length): return None
    def getPDF(self, state, temperature): return None
    def sample(self, pdf): return next(self.samples)


class PlacementTests(unittest.TestCase):
    def assert_inside(self, paths, outline):
        contour = np.asarray(outline, dtype=np.float32)
        for path in paths:
            for start, end in zip(path, path[1:]):
                for point in np.linspace(start, end, 21):
                    self.assertGreaterEqual(cv2.pointPolygonTest(contour, tuple(point), True), 0)

    def test_displaced_continuation_and_gcode_fit_circle(self):
        outline = circle()
        result = complete_drawing(RecordedModel(outline[-1]), [outline], 700, 600)
        self.assertTrue(result['finished'])
        self.assertEqual(result['warnings'], [])
        self.assertEqual(len(result['strokes']), 2)
        self.assert_inside(result['strokes'], outline)
        self.assert_gcode_matches(result)

    def test_generated_coordinates_have_fixed_offsets_and_can_be_sent(self):
        # Put the drawing near the canvas origin to exercise negative G-code coordinates.
        outline = [[x - 150, y - 150] for x, y in circle()]
        result = complete_drawing(RecordedModel(outline[-1]), [outline], 700, 600)
        original_mapping = CanvasToPaper(700, 600, bounds=(0, 0, 355.6, 304.8))
        expected = [[*original_mapping.point(*start), *original_mapping.point(*end)]
                    for path in result['strokes'] for start, end in zip(path, path[1:])]
        expected = np.asarray(expected) - [80, 75, 80, 75]
        lines = result['gcode'].splitlines()
        simulation = simulate(lines)
        actual = [[step.x0, step.y0, step.x1, step.y1]
                  for step in simulation['steps'] if step.pen_down and step.moves]
        np.testing.assert_allclose(actual, expected, atol=0.0051)
        self.assertEqual(simulation['warnings'], [])
        self.assertTrue(np.any(np.asarray(actual) < 0))
        self.assertIn('G92 X0 Y0', lines)
        self.assertTrue(lines[-1].startswith('G1 X0.00 Y0.00 '))
        board = FakeGrbl()
        send_gcode(lines, ser=board, log=lambda message: None)
        self.assertEqual(board.sent[0], 'G21')

    def test_offsets_are_applied_once_to_points_and_not_accumulated_in_deltas(self):
        mapping = CanvasToPaper(700, 600)
        anchor = mapping.point(200, 200)
        moves = mapping.strokes([[0, 0, 1, 0, 0], [50, 50, 1, 0, 0], [50, 0, 0, 1, 0]])
        lines = strokes_to_gcode(moves, anchor)
        simulation = simulate(lines)
        drawn = [step for step in simulation['steps'] if step.pen_down and step.moves]
        np.testing.assert_allclose([drawn[0].x0, drawn[0].y0], [21.6, 26.6], atol=0.0051)
        np.testing.assert_allclose([drawn[-1].x1, drawn[-1].y1], [72.4, 52.0], atol=0.0051)
        self.assertEqual(simulation['warnings'], [])

    def assert_gcode_matches(self, result):
        paper = CanvasToPaper(700, 600)
        expected = [[*paper.point(*start), *paper.point(*end)]
                    for path in result['strokes'] for start, end in zip(path, path[1:])]
        simulation = simulate(result['gcode'].splitlines())
        actual = [[step.x0, step.y0, step.x1, step.y1]
                  for step in simulation['steps'] if step.pen_down and step.moves]
        np.testing.assert_allclose(actual, expected, atol=0.0051)
        self.assertEqual(simulation['stats']['pen_lowers'], len(result['strokes']))

    def test_concave_outline_contains_whole_segments(self):
        outline = [[100, 100], [500, 100], [500, 200], [200, 200],
                   [200, 500], [100, 500], [100, 100]]
        raw = [[[600, 500], [900, 500], [900, 800]], [[700, 600], [800, 700]]]
        fitted = fit_to_drawing(raw, [outline], 700, 600)
        self.assert_inside(fitted, outline)
        # All segments retain one common scale, including between separate paths.
        original = np.concatenate(raw)
        placed = np.concatenate(fitted)
        ratios = np.linalg.norm(np.diff(placed, axis=0), axis=1) / np.linalg.norm(np.diff(original, axis=0), axis=1)
        np.testing.assert_allclose(ratios, ratios[0])

    def test_open_drawing_uses_its_bounds(self):
        raw = [[[600, 500], [900, 800]]]
        fitted = fit_to_drawing(raw, [[[100, 100], [300, 300]]], 700, 600)
        self.assert_inside(fitted, [[100, 100], [300, 100], [300, 300], [100, 300]])

    def test_degenerate_drawing_preserves_existing_behavior(self):
        raw = [[[600, 500], [650, 550]]]
        self.assertEqual(fit_to_drawing(raw, [[[100, 100], [300, 100]]], 700, 600), raw)
        self.assertEqual(fit_to_drawing([], [circle()], 700, 600), [])

    def test_request_uses_human_outline_despite_generated_history(self):
        outline = circle()
        previous_model_stroke = [[10, 10], [690, 10], [690, 590], [10, 590], [10, 10]]
        history = [outline, previous_model_stroke]
        model = RecordedModel(previous_model_stroke[-1])
        body = {'strokes': history, 'human_strokes': [outline], 'width': 700, 'height': 600}

        async def receive():
            return {'type': 'http.request', 'body': json.dumps(body).encode(), 'more_body': False}

        app = SimpleNamespace(state=SimpleNamespace(model=model))
        request = Request({'type': 'http', 'method': 'POST', 'path': '/complete',
                           'headers': [], 'app': app}, receive)
        response = asyncio.run(complete(request))
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.body)
        self.assert_inside(result['strokes'], outline)
        self.assert_gcode_matches(result)


if __name__ == '__main__':
    unittest.main()
