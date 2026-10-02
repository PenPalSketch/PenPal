#*
#* Core implementation for RNN-based Magenta sketch models such as SketchRNN.
#*
#* @license
#* Copyright 2018 Google Inc. All Rights Reserved.
#* Licensed under the Apache License, Version 2.0 (the "License");
#* you may not use this file except in compliance with the License.
#* You may obtain a copy of the License at
#*
#* http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing, software
#* distributed under the License is distributed on an "AS IS" BASIS,
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#* See the License for the specific language governing permissions and
#* limitations under the License.


#*
#* Imports

import tensorflow as tf

#* Interface for JSON specification of a `MusicVAE` model.
#*
#* @property max_seq_len: Model trained on dataset w/ this max sequence length.
#* @property mode: Pre-trained models have this parameter for legacy reasons.
#* 0 for VAE, 1 for Decoder only. This model is Decoder only (not used).
#* @property name: QuickDraw name, like cat, dog, elephant, etc
#* @property scale_factor: the factor to convert from neural-network space to
#* pixel space. Most pre-trained models have this number between 80-120
#* @property version: Pre-trained models have a version between 1-6, for
#* the purpose of experimental research log.

from typing import TypedDict, Optional
import numpy as np

import sketch_support as support

class SketchRNNInfo(TypedDict):
    max_seq_len: int
    mode: int
    name: str
    scale_factor: float 
    version: int

#* Interface for specification of the Probability Distribution Function
#* of a pen stroke.
#* 
#* Please refer to "A Neural Representation of Sketch Drawings"
#* https://arxiv.org/abs/1704.03477
#* 
#* In Eq.3 is an explanation of all of these parameters.
#* 
#* Below is a brief description:
#* 
#* @property pi: categorial distribution for mixture of Gaussian
#* @property muX: mean for x-axis
#* @property muY: mean for y-axis
#* @property sigmaX: standard deviation of x-axis
#* @property sigmaY: standard deviation of y-axis
#* @property corr: correlation parameter between x and y
#* @property pen: categorical distribution for the 3 pen states

class StrokePDF(TypedDict):
    pi: np.ndarray
    muX: np.ndarray
    muY: np.ndarray
    sigmaX: np.ndarray
    sigmaY: np.ndarray
    corr: np.ndarray
    pen: np.ndarray

#* States of the LSTM Cell
#* 
#* Long-Short Term Memory: ftp://ftp.idsia.ch/pub/juergen/lstm.pdf
#* 
#* @property c: memory "cell" of the LSTM.
#* @property h: hidden state (also the output) of the LSTM.

class LSTMState(TypedDict):
    c: np.ndarray
    h: np.ndarray

#* Main SketchRNN model class.
#*
#* Implementation of decoder model in https://arxiv.org/abs/1704.03477
#* 
#* TODO(hardmaru): make a "batch" continueSequence-like method
#* that runs fully on GPU.

class SketchRNN:
    checkpoint_url: str
    initialized: bool
    forget_bias: tf.Tensor

    info: SketchRNNInfo
    numUnits: int

    pixelFactor: int
    scaleFactor: int

    # raw weights and dimensions directly from JSON
    weights: list[np.ndarray]
    weight_dims: list[list[int]]

    output_kernel: tf.Tensor
    output_bias: tf.Tensor
    lstm_kernel: tf.Tensor
    lstm_bias: tf.Tensor

    raw_vars: list[tf.Tensor]
    
    NMIXTURE = 20


    #* `SketchRNN` constructor.
    #*
    #* @param checkpointURL Path to the checkpoint directory.






    
    #* Samples the next point of the sketch given pdf parameters
    #*
    #* @param pdf result from get_pdf() call (a StrokePDF)
    #*
    #* @returns [dx, dy, penDown, penUp, penEnd]
 
    def sample(self, pdf: StrokePDF) -> list[float]:
        # pdf is a StrokePDF
        # returns [dx, dy, penDown, penUp, penEnd]
        idx = support.sample_softmax(pdf["pi"])
        mu1 = pdf["muX"][idx]
        mu2 = pdf["muY"][idx]
        sigma1 = pdf["sigmaX"][idx]
        sigma2 = pdf["sigmaY"][idx]
        corr = pdf["corr"][idx]
        pen_idx = support.sample_softmax(pdf["pen"])
        penstate = [0, 0, 0]
        if pen_idx >= 0:  # sample_softmax returns -1 if sampling failed
            penstate[pen_idx] = 1
        delta = support.birandn(mu1, mu2, sigma1, sigma2, corr)
        stroke = [
            delta[0] * self.scaleFactor,
            delta[1] * self.scaleFactor,
            penstate[0],
            penstate[1],
            penstate[2]
        ]
        return stroke
 
    #* Simplifies line using RDP algorithm
    #*
    #* @param line list of points [[x0, y0], [x1, y1], ...]
    #* @param tolerance (Optional) default 2.0
    #*
    #* @returns simplified line [[x0', y0'], [x1', y1'], ...]
 
    def simplify_line(self, line: list[list[float]],
                      tolerance: Optional[float] = None) -> list[list[float]]:
        if tolerance is None:
            tolerance = 2.0
        return support.simplify_line(line, tolerance)
 
    #* Simplifies lines using RDP algorithm
    #*
    #* @param lines list of lines (each element is [[x0, y0], [x1, y1], ...])
    #* @param tolerance (Optional) default 2.0
    #*
    #* @returns simplified lines (each elem is [[x0', y0'], [x1', y1'], ...])
 
    def simplify_lines(self, lines: list[list[list[float]]],
                       tolerance: Optional[float] = None) -> list[list[list[float]]]:
        return support.simplify_lines(lines, tolerance)
 
    #* Convert from polylines to stroke-5 format that sketch-rnn uses
    #*
    #* @param lines list of lines, each elem is ([[x0, y0], [x1, y1], ...])
    #*
    #* @returns stroke-5 format of the lines, list of [dx, dy, p0, p1, p2]
 
    def lines_to_stroke(self, lines: list[list[list[float]]]) -> list[list[float]]:
        return support.lines_to_strokes(lines)
 
    #* Convert from a line format to stroke-5
    #*
    #* @param line list of points [[x0, y0], [x1, y1], ...]
    #* @param last_point the absolute position of the last point
    #*
    #* @returns stroke-5 format of the line, list of [dx, dy, p0, p1, p2]
 
    def line_to_stroke(self, line: list[list[float]],
                       last_point: list[float]) -> list[list[float]]:
        return support.line_to_stroke(line, last_point)
