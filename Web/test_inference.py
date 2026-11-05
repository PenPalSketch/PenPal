import unittest

from starlette.testclient import TestClient

from HW.Plotter.gcode import FLIP_X, FLIP_Y, PAPER_BOUNDS, PEN_DELAY, PEN_UP_CMD
from HW.Plotter.simulate import simulate
from ML.sketch_support import lines_to_strokes
from Web.inference import app, complete_drawing


class SampleModel:
    """Run the real completion/export loop with a known sequence of samples."""

    def __init__(self, samples):
        self.samples = iter(samples)

    def simplify_lines(self, strokes, tolerance):
        return strokes

    def lines_to_stroke(self, paths):
        return lines_to_strokes(paths)

    def zero_input(self):
        return [0, 0, 1, 0, 0]

    def zeroState(self):
        return None

    def update(self, sample, state):
        return state

    def updateStrokes(self, seed, state, count):
        return state

    def getPDF(self, state, temperature):
        return None

    def sample(self, pdf):
        return next(self.samples)


HUMAN = [[[90, 100], [100, 100]]]
SAMPLES = [
    [10, 0, 1, 0, 0],  # travel, then lower pen
    [0, 10, 0, 1, 0],  # draw, then lift pen
    [20, 0, 1, 0, 0],  # travel between paths
    [0, 10, 0, 1, 0],  # draw, then lift pen
    [999, 999, 0, 0, 1],  # end-token displacement must not be exported
]


class GcodeCompletionTests(unittest.TestCase):
    def test_export_matches_visible_paths_at_different_canvas_sizes(self):
        for width, height in ((297, 210), (640, 480), (1280, 720)):
            with self.subTest(size=(width, height)):
                result = complete_drawing(SampleModel(SAMPLES), HUMAN, width, height)
                expected = [[[110, 100], [110, 110]], [[130, 110], [130, 120]]]
                self.assertEqual(result["strokes"], expected)
                self.assertTrue(result["finished"])
                self.assertEqual(result["warnings"], [])
                self.assertTrue(result["gcode"].startswith(f"G21\nG90\nG92 X0 Y0\n{PEN_UP_CMD}\n"))
                self.assertTrue(result["gcode"].endswith(f"{PEN_UP_CMD}\nG4 P{PEN_DELAY}\n"))

                sim = simulate(result["gcode"].splitlines())
                drawn = [s for s in sim["steps"] if s.pen_down and s.moves]
                self.assertEqual(len(drawn), 2)
                xmin, ymin, xmax, ymax = PAPER_BOUNDS
                scale = min((xmax - xmin) / width, (ymax - ymin) / height)
                left = xmin + (xmax - xmin - width * scale) / 2
                top = ymin + (ymax - ymin - height * scale) / 2
                for step, (start, end) in zip(drawn, expected):
                    actual = (step.x0, step.y0, step.x1, step.y1)
                    mapped = tuple(value for x, y in (start, end) for value in (
                        left + (width - x if FLIP_X else x) * scale,
                        top + (height - y if FLIP_Y else y) * scale,
                    ))
                    for a, b in zip(actual, mapped):
                        self.assertAlmostEqual(a, b, delta=0.0051)

    def test_empty_or_only_pen_up_completion_has_no_download(self):
        for samples in ([SAMPLES[-1]], [[10, 0, 0, 1, 0], SAMPLES[-1]]):
            result = complete_drawing(SampleModel(samples), HUMAN, 640, 480)
            self.assertEqual(result, {"strokes": [], "finished": True, "gcode": "", "warnings": []})

    def test_step_limit_exports_all_500_moves(self):
        result = complete_drawing(SampleModel([[0.1, 0, 1, 0, 0]] * 500), HUMAN, 297, 210)
        self.assertFalse(result["finished"])
        self.assertEqual(len(result["strokes"][0]), 500)
        sim = simulate(result["gcode"].splitlines())
        self.assertEqual(sum(step.pen_down and step.moves for step in sim["steps"]), 499)
        self.assertEqual(result["warnings"], [])

    def test_endpoint_returns_serializable_gcode(self):
        app.state.model = SampleModel(SAMPLES)
        response = TestClient(app).post("/complete", json={
            "strokes": HUMAN, "width": 640, "height": 480,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("G1 ", response.json()["gcode"])
        self.assertEqual(response.json()["warnings"], [])

    def test_endpoint_rejects_invalid_dimensions(self):
        app.state.model = SampleModel([])
        client = TestClient(app)
        for width in (None, 0, -1, 1.5, True, "640"):
            response = client.post("/complete", json={"strokes": HUMAN, "width": width, "height": 480})
            self.assertEqual(response.status_code, 400)
            self.assertIn("canvas width and height", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
