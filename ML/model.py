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

from typing import TypedDict
import numpy as np

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

    def __init__(self, checkpoint_url: str):
        self.checkpoint_url = checkpoint_url
        self.initialized = False

    # * Match the legacy TensorFlow.js basicLSTMCell
    def basic_lstm_cell(self, forget_bias, lstm_kernel, lstm_bias, x, c, h):
        combined = tf.concat([x, h], axis=1)  
        gates = tf.matmul(combined, lstm_kernel) + lstm_bias 
        i, j, f, o = tf.split(gates, 4, axis=1)

        new_c = tf.sigmoid(f + forget_bias) * c + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.sigmoid(o) * tf.tanh(new_c)

        return new_c, new_h

    # * Updates the RNN, returns the next state.
    # *
    # * @param stroke [dx, dy, penDown, penUp, penEnd].
    # * @param state previous LSTMState.
    # *
    # * @returns next LSTMState.
    
    def update(self, stroke, state):
        numUnits = self.numUnits
        s = self.scaleFactor

        normStroke = [
            stroke[0]/s, 
            stroke[1]/s, 
            stroke[2], 
            stroke[3], 
            stroke[4]
        ]

        x = tf.convert_to_tensor([normStroke], dtype=tf.float32) # current stroke tensor
        c = tf.convert_to_tensor([state["c"]], dtype=tf.float32) # cell memory context tensor
        h = tf.convert_to_tensor([state["h"]], dtype=tf.float32) # hidden state context tensor

        # apply lstm cell math to update c and h for next iteration
        new_c, new_h = self.basic_lstm_cell(
            self.forget_bias,
            self.lstm_kernel,
            self.lstm_bias,
            x,
            c,
            h
        )

        # return updated c and h
        return {
            "c": new_c.numpy()[0],
            "h": new_h.numpy()[0],
        }

    #* Updates the RNN on a series of Strokes, returns the next state.
    #*
    #* @param strokes list of [dx, dy, penDown, penUp, penEnd].
    #* @param state previous LSTMState.
    #* @param steps (Optional) number of steps of the stroke to update
    #* (default is length of strokes list)
    #* 
    #*
    #* @returns the final LSTMState.

    def updateStrokes(self, strokes, state, steps):
        numUnits = self.numUnits
        s = self.scaleFactor

        x = None
        c = None
        h = None
        newState = None
        numSteps = len(strokes)

        # if the number of steps is specified, use the parameter instead of the array length
        if steps is not None:
            numSteps = steps

        c = tf.convert_to_tensor([state["c"]], dtype=tf.float32) # cell memory context tensor
        h = tf.convert_to_tensor([state["h"]], dtype=tf.float32) # hidden state context tensor

        # iterate through the sequence of steps
        for stroke in strokes[:numSteps]:
            normStroke = [
                stroke[0] / s,
                stroke[1] / s,
                stroke[2],
                stroke[3],
                stroke[4],
            ]

            x = tf.convert_to_tensor([normStroke], dtype=tf.float32)

            new_c, new_h = self.basic_lstm_cell(
                self.forget_bias,
                self.lstm_kernel,
                self.lstm_bias,
                x,
                c,
                h
            )

            c = new_c
            h = new_h

        return {
            "c": new_c.numpy()[0],
            "h": new_h.numpy()[0],
        }