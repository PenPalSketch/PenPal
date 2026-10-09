#* Imports
try:
    from . import sketch_support as support
except ImportError:
    import sketch_support as support

from numpy._core.numeric import dtype
import tensorflow as tf
from typing import TypedDict, Optional
from pathlib import Path
import numpy as np
import math
import json

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

    def is_initialized(self):
        return self.initialized
    
    def setPixelFactor(self, scale: int):
        # for best effect, set to 1.0 for d3 or paper.js, 2.0 for p5.js
        self.pixelFactor = scale
        self.scaleFactor = self.info["scale_factor"] / self.pixelFactor

    def dispose(self):
        self.raw_vars = None
        self.forget_bias = None

        self.output_kernel = None
        self.output_bias = None
        self.lstm_kernel = None
        self.lstm_bias = None

        self.initialized = False

    """
    examples:
    SketchRNNInfo: {"mode":2,"version":6,"max_seq_len":130,"name":"cat","scale_factor":82.2}
    weight_dims: [[512,123],[123],[5,2048],[512,2048],[2048]
    weightStrings: "wRCC6sALHvGcC27lXeCdCaz27QCTEOrwuvttDIr3y/fvBZ....
    """
    def instantiate_from_json(self, info: SketchRNNInfo, weight_dims: list[list[int]], weightStrings: list[str]):
        self.forget_bias = tf.convert_to_tensor(1.0, dtype=tf.float32)
        self.info = info
        self.setPixelFactor(2.0)
        self.weight_dims = weight_dims
        self.numUnits = self.weight_dims[0][0]; # size of LSTM
        
        MAXWEIGHT = 10.0
        self.weights = []
        for weightString in weightStrings:
            rawWeights =  np.array(support.string_to_array(weightString))
            N = len(rawWeights)
            rawWeights = MAXWEIGHT* rawWeights / 32767
            self.weights.append(rawWeights)
        
        self.output_kernel = tf.reshape(
            tf.convert_to_tensor(self.weights[0], dtype=float),
            [self.weight_dims[0][0], self.weight_dims[0][1]]
        )

        self.output_bias = tf.convert_to_tensor(
            self.weights[1],
            dtype=tf.float32
        )

        lstmKernelXH = tf.reshape(
            tf.convert_to_tensor(self.weights[2], dtype=tf.float32),
            [self.weight_dims[2][0], self.weight_dims[2][1]]
        )

        lstmKernelHH = tf.reshape(
            tf.convert_to_tensor(self.weights[3], dtype=tf.float32),
            [self.weight_dims[3][0], self.weight_dims[3][1]]
        )
        axis = 0
        self.lstm_kernel = tf.concat(
            [lstmKernelXH, lstmKernelHH],
            axis=axis,
        )

        self.lstm_bias = tf.convert_to_tensor(
            self.weights[4],
            dtype=tf.float32
        )

        self.raw_vars = [
            self.output_kernel,
            self.output_bias,
            self.lstm_kernel,
            self.lstm_bias
        ]

    def initialize(self):
        self.dispose()

        model_dir = Path(__file__).resolve().parent
        with open(model_dir / "cat.txt", "r") as f:
            vars = json.load(f)

        self.instantiate_from_json(
            vars[0],
            vars[1],
            vars[2]
        )

        self.initialized = True
    
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
    #* Given the RNN state, returns the probability distribution function (pdf)
    #* of the next stroke. Optionally adjust the temperature of the pdf here.
    #*
    #* @param state previous LSTMState.
    #* @param temperature (Optional) for dx and dy (default 0.65)
    #* @param softmaxTemperature (Optional) for Pi and Pen discrete states
    #* (default is temperature * 0.5 + 0.5, which is a nice heuristic.)
    #*
    #* @returns StrokePDF (pi, muX, muY, sigmaX, sigmaY, corr, pen)
    def getPDF(self, state: LSTMState, temperature: float = 0.65, softmaxTemperature: float | None = None):

        temp = temperature
        discreteTemp = 0.5 + 0.5 * temperature
        if softmaxTemperature:
            discreteTemp = softmaxTemperature

        NOUT = self.NMIXTURE

        h = tf.reshape(tf.convert_to_tensor(state['h'], dtype=tf.float32),
                       (1, self.numUnits))

        sqrttemp = math.sqrt(temp)

        z = tf.squeeze(tf.matmul(h, self.output_kernel) + self.output_bias)

        rawPen, rst = tf.split(z, [3, NOUT*6])
        rawPi, mu1, mu2, rawSigma1, rawSigma2, rawCorr = tf.split(rst, 6)
        
        pen = tf.nn.softmax(rawPen / discreteTemp)
        pi = tf.nn.softmax(rawPi / discreteTemp)
        sigma1 = tf.exp(rawSigma1) * sqrttemp
        sigma2 = tf.exp(rawSigma2) * sqrttemp
        corr = tf.tanh(rawCorr)

        pdf = StrokePDF(
            pi = pi.numpy(), # convert to a python list of numbers
            muX = mu1.numpy(),
            muY = mu2.numpy(),
            sigmaX = sigma1.numpy(),
            sigmaY = sigma2.numpy(),
            corr = corr.numpy(),
            pen = pen.numpy(),
        )

        return pdf


    #* Returns the zero/initial state of the model
    #*
    #* @returns zero state of the lstm: [c, h], where c and h are zero vectors.
    def zeroState(self):
        result = LSTMState(
            c=np.zeros(self.numUnits, dtype=np.float32),
            h=np.zeros(self.numUnits, dtype=np.float32)
        )
        return result


    #* Returns a new copy of the rnn state
    #*
    #* @param rnnState original LSTMState
    #*
    #* @returns copy of LSTMState
    def copyState(self, rnnState: LSTMState):
        result = LSTMState(
            c=np.array(rnnState['c'], dtype=np.float32),
            h=np.array(rnnState['h'], dtype=np.float32),
        )
        return result

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

sketchy = SketchRNN("cat")
sketchy.initialize()
print("hello")
