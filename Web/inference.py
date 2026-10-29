import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route


def complete_drawing(model, strokes, temperature=0.25, max_steps=500):
    paths = model.simplify_lines(strokes, 1.5)
    paths = [path for path in paths if len(path) >= 2 and any(p != path[0] for p in path)]
    if not paths:
        raise ValueError("Draw a stroke with at least two different points first.")

    # Remove the end-of-drawing token: this drawing is still being completed.
    seed = model.lines_to_stroke(paths)[:-1]
    state = model.update(model.zero_input(), model.zeroState())
    state = model.updateStrokes(seed, state, len(seed))
    x, y = paths[-1][-1]
    pen_down = bool(seed[-1][2])
    generated = []
    path = None
    finished = False

    for _ in range(max_steps):
        sample = model.sample(model.getPDF(state, temperature))
        if sample[4] == 1:
            finished = True
            break

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

    return {"strokes": generated, "finished": finished}


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
            body["strokes"], body.get("temperature", 0.25),
        )
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    return JSONResponse(result)


app = Starlette(
    routes=[Route("/", homepage), Route("/complete", complete, methods=["POST"])],
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
