<<<<<<< HEAD
import base64
import numpy as np

def stringToArray(b64encoded: str) -> np.ndarray:
    u = base64.b64decode(b64encoded)
    result = np.frombuffer(u, dtype=np.int16)
    return result
=======
"""
Support modules used by SketchRNN (Python port).

Original TypeScript:
Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import base64
import math
import random
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Base64 decoding
# ---------------------------------------------------------------------------

def string_to_uint8array(b64encoded: str) -> np.ndarray:
    """Decode b64 data into a uint8 array."""
    return np.frombuffer(base64.b64decode(b64encoded), dtype=np.uint8)


def string_to_array(b64encoded: str) -> np.ndarray:
    """Decode b64 data into an int16 array (little-endian, like JS on most machines)."""
    raw = string_to_uint8array(b64encoded).tobytes()
    return np.frombuffer(raw, dtype="<i2")


# ---------------------------------------------------------------------------
# Sampling normals / bi-normals
# (random number util from https://github.com/karpathy/recurrentjs)
# ---------------------------------------------------------------------------

_return_v = False
_v_val = 0.0


def gauss_random() -> float:
    global _return_v, _v_val
    if _return_v:
        _return_v = False
        return _v_val
    while True:
        u = 2 * random.random() - 1
        v = 2 * random.random() - 1
        r = u * u + v * v
        if r != 0 and r <= 1:
            break
    c = math.sqrt(-2 * math.log(r) / r)
    _v_val = v * c  # cache this
    _return_v = True
    return u * c


def randf(a: float, b: float) -> float:
    return random.random() * (b - a) + a


def randi(a: float, b: float) -> int:
    return math.floor(random.random() * (b - a) + a)


def randn(mu: float, std: float) -> float:
    return mu + gauss_random() * std


def birandn(mu1: float, mu2: float, std1: float, std2: float, rho: float) -> List[float]:
    """From http://www.math.grin.edu/~mooret/courses/math336/bivariate-normal.html"""
    z1 = randn(0, 1)
    z2 = randn(0, 1)
    x = math.sqrt(1 - rho * rho) * std1 * z1 + rho * std1 * z2 + mu1
    y = std2 * z2 + mu2
    return [x, y]


def sample_softmax(z_sample: Sequence[float]) -> int:
    """Sample from a categorical distribution."""
    x = randf(0, 1)
    accumulate = 0.0
    for i, p in enumerate(z_sample):
        accumulate += p
        if accumulate >= x:
            return i
    print("error sampling pi index")
    return -1


# ---------------------------------------------------------------------------
# Ramer-Douglas-Peucker line simplification
#
# from https://gist.github.com/adammiller/826148
# Copyright 2002, softSurfer (www.softsurfer.com)
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.
# ---------------------------------------------------------------------------

def simplify_line(V: List[List[float]], tolerance: float = 2.0) -> List[List[float]]:
    """V is a polyline [[x1, y1], [x2, y2], ...]; tolerance is the approximation tolerance."""
    tol = tolerance

    def diff(u, v):
        return [u[0] - v[0], u[1] - v[1]]

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1]

    def norm2(v):
        return v[0] * v[0] + v[1] * v[1]

    def d2(u, v):
        return norm2(diff(u, v))

    def simplify_dp(tol, v, j, k, mk):
        """Douglas-Peucker recursive routine. Marks vertices of v[j..k] to keep in mk."""
        if k <= j + 1:  # nothing to simplify
            return
        maxi = j  # index of vertex farthest from segment S
        maxd2 = 0.0  # squared distance of farthest vertex
        tol2 = tol * tol
        S = [v[j], v[k]]  # segment from v[j] to v[k]
        u = diff(S[1], S[0])  # segment direction vector
        cu = norm2(u)  # segment length squared

        for i in range(j + 1, k):
            w = diff(v[i], S[0])
            cw = dot(w, u)
            if cw <= 0:
                dv2 = d2(v[i], S[0])
            elif cu <= cw:
                dv2 = d2(v[i], S[1])
            else:
                b = cw / cu
                pb = [S[0][0] + b * u[0], S[0][1] + b * u[1]]
                dv2 = d2(v[i], pb)
            if dv2 <= maxd2:
                continue
            maxi = i
            maxd2 = dv2

        if maxd2 > tol2:  # error is worse than the tolerance
            mk[maxi] = 1  # keep the farthest vertex
            simplify_dp(tol, v, j, maxi, mk)
            simplify_dp(tol, v, maxi, k, mk)
        # else the approximation is OK, ignore intermediate vertices

    n = len(V)
    if n == 0:
        return []
    tol2 = tol * tol

    # STAGE 1. Vertex reduction within tolerance of prior vertex cluster
    vt = [V[0]]
    pv = 0
    for i in range(1, n):
        if d2(V[i], V[pv]) < tol2:
            continue
        vt.append(V[i])
        pv = i
    if pv < n - 1:
        vt.append(V[n - 1])  # finish at the end

    # STAGE 2. Douglas-Peucker polyline simplification
    k = len(vt)
    mk = [0] * k
    mk[0] = mk[k - 1] = 1  # mark first and last vertices
    simplify_dp(tol, vt, 0, k - 1, mk)

    # copy marked vertices to the output
    return [vt[i] for i in range(k) if mk[i]]


def simplify_lines(lines: List[List[List[float]]],
                   tolerance: Optional[float] = None) -> List[List[List[float]]]:
    """Clean wrapper to use the RDP function on many lines."""
    tol = tolerance if tolerance else 2.0
    return [simplify_line(line, tol) for line in lines]


# ---------------------------------------------------------------------------
# Stroke-5 conversion
# ---------------------------------------------------------------------------

def lines_to_strokes(raw_data: List[List[List[float]]]) -> List[List[float]]:
    """Convert from polylines to the stroke-5 format that sketch-rnn uses."""
    px, py = 0, 0
    stroke: List[List[float]] = []
    for line in raw_data:
        length = len(line)
        if length > 1:
            for j, p in enumerate(line):
                x, y = p[0], p[1]
                if j == length - 1:
                    poff, pon = 1, 0
                else:
                    poff, pon = 0, 1
                dx = x - px
                dy = y - py
                px, py = x, y
                stroke.append([dx, dy, pon, poff, 0])
    stroke.append([0, 0, 0, 0, 1])
    return stroke[1:]


def line_to_stroke(line: List[List[float]], last_point: List[float]) -> List[List[float]]:
    """Convert a single polyline to stroke-5 format, starting from last_point."""
    stroke: List[List[float]] = []
    px, py = last_point[0], last_point[1]
    length = len(line)
    if length > 1:
        for j, p in enumerate(line):
            x, y = p[0], p[1]
            if j == length - 1:
                poff, pon = 1, 0
            else:
                poff, pon = 0, 1
            dx = x - px
            dy = y - py
            px, py = x, y
            stroke.append([dx, dy, pon, poff, 0])
    return stroke

def test():
    print('hello world')
    return
>>>>>>> origin/main
