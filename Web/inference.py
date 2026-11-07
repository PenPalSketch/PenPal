import sys
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from HW.Plotter.gcode import CanvasToPaper, strokes_to_gcode
from HW.Plotter.simulate import simulate
from HW.Plotter.send import PlotterError, send_gcode


plot_lock = Lock()

# this has to be offloaded into a separate thread because sending it
# must hold an entire thread until drawing is complete.
def plot_gcode(lines):
    if not plot_lock.acquire(blocking=False):
        return JSONResponse({"error": "The plotter is busy."}, status_code=409)
    try:
        send_gcode(lines)
        return JSONResponse({"status": "complete"})
    except (PlotterError, OSError) as error:
        return JSONResponse({"error": str(error)}, status_code=503)
    finally:
        plot_lock.release()


def complete_drawing(model, strokes, width, height, temperature=0.25, max_steps=500):
    if any(type(size) is not int or size <= 0 for size in (width, height)):
        raise ValueError("Send positive integer canvas width and height values.")
    paths = model.simplify_lines(strokes, 1.5)
    paths = [path for path in paths if len(path) >= 2 and any(p != path[0] for p in path)]
    if not paths:
        raise ValueError("Draw a stroke with at least two different points first.")

    # Remove the end-of-drawing token: this drawing is still being completed.
    seed = model.lines_to_stroke(paths)[:-1]
    state = model.update(model.zero_input(), model.zeroState())
    state = model.updateStrokes(seed, state, len(seed))
    x, y = paths[-1][-1]
    anchor = (x, y)
    pen_down = bool(seed[-1][2])
    generated = []
    moves = []
    path = None
    finished = False

    for _ in range(max_steps):
        sample = model.sample(model.getPDF(state, temperature))
        if sample[4] == 1:
            finished = True
            break

        moves.append([float(sample[0]), float(sample[1]), *map(int, sample[2:])])
        next_x, next_y = x + float(sample[0]), y + float(sample[1])
        # The previous pen state determines whether this movement draws a line.
        if pen_down:
            if path is None:
                path = [[x, y]]
                generated.append(path)
            path.append([next_x, next_y])
        else:
            path = None

        x, y = next_x, next_y
        pen_down = bool(sample[2])
        state = model.update(sample, state)

    gcode = ""
    warnings = []
    if generated:
        to_paper = CanvasToPaper(width, height)
        lines = strokes_to_gcode(
            to_paper.strokes(moves), to_paper.point(*anchor), max_steps=len(moves),
        )
        warnings = simulate(lines)["warnings"]
        gcode = "\n".join(lines) + "\n"
    return {"strokes": generated, "finished": finished, "gcode": gcode, "warnings": warnings}


@asynccontextmanager
async def lifespan(app):
    from ML.model import SketchRNN

    model = SketchRNN("cat")
    model.initialize()
    model.setPixelFactor(5.0)
    app.state.model = model
    try:
        yield
    finally:
        model.dispose()


async def homepage(request):
    return FileResponse(Path(__file__).with_name("index.html"))


async def complete(request):
    try:
        body = await request.json()
        if not isinstance(body, dict) or "strokes" not in body:
            raise ValueError("Send a JSON object with a strokes field.")
        result = await run_in_threadpool(
            complete_drawing, request.app.state.model,
            body["strokes"], body.get("width"), body.get("height"),
            body.get("temperature", 0.25),
        )
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    return JSONResponse(result)


async def plot(request):
    body = await request.json()
    return await run_in_threadpool(plot_gcode, body["gcode"].splitlines())


app = Starlette(
    routes=[
        Route("/", homepage),
        Route("/complete", complete, methods=["POST"]),
        Route("/plot", plot, methods=["POST"]),
    ],
    lifespan=lifespan,
    middleware=[Middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000", "http://127.0.0.1:3000",
            "http://localhost:8000", "http://127.0.0.1:8000",
        ],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )],
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
