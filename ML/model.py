#*
#* Core implementation for RNN-based Magenta sketch models such as SketchRNN.
#*
#* @license
#* Copyright 2018 Google Inc. All Rights Reserved.
#* Licensed under the Apache License, Version 2.0 (the "License");
#* you may not use self file except in compliance with the License.
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
import sketch_support as support
import tensorflow as tf

#* Interface for JSON specification of a `MusicVAE` model.
#*
#* @property max_seq_len: Model trained on dataset w/ self max sequence length.
#* @property mode: Pre-trained models have self parameter for legacy reasons.
#* 0 for VAE, 1 for Decoder only. self model is Decoder only (not used).
#* @property name: QuickDraw name, like cat, dog, elephant, etc
#* @property scale_factor: the factor to convert from neural-network space to
#* pixel space. Most pre-trained models have self number between 80-120
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

    def __init__(self, checkpoint_url: str):
        self.checkpoint_url = checkpoint_url
        self.initialized = False

    def setPixelFactor(self, scale: int):
        # for best effect, set to 1.0 for d3 or paper.js, 2.0 for p5.js
        self.pixelFactor = scale
        self.scaleFactor = self.info["scale_factor"] / self.pixelFactor

    def dispose(self):
        if (self.rawVars):
            for rawVar in self.rawVars:
                rawVar.dispose()
            self.rawVars = None
        
        if (self.forgetBias):
            self.forgetBias.dispose()
            self.forgetBias = None
        
        self.initialized = False

    """
    examples:
    SketchRNNInfo: {"mode":2,"version":6,"max_seq_len":130,"name":"cat","scale_factor":82.2}
    weightDims: [[512,123],[123],[5,2048],[512,2048],[2048]
    weightStrings: "wRCC6sALHvGcC27lXeCdCaz27QCTEOrwuvttDIr3y/fvBZ....
    """
    def instantiateFromJSON(self, info: SketchRNNInfo, weightDims: list[list[int]], weightStrings: list[str]):
        self.forgetBias = tf.convert_to_tensor(1.0, dtype=tf.float32)
        self.info = info
        self.setPixelFactor(2.0)
        self.weightDims = weightDims
        self.numUnits = self.weightDims[0][0]; # size of LSTM
        
        MAXWEIGHT = 10.0
        self.weights = []
        for weightString in weightStrings:
            rawWeights =  np.array(support.stringToArray(weightString), dtype=np.float32)
            N = len(rawWeights)
            rawWeights = MAXWEIGHT* rawWeights / 32767
            self.weights.append(rawWeights)
        
        self.outputKernel = tf.reshape(
            tf.convert_to_tensor(self.weights[0]),
            [self.weightDims[0][0], self.weightDims[0][1]]
        )

        self.outputBias = tf.convert_to_tensor(
            self.weights[1],
            dtype=tf.float32
        )

        lstmKernelXH = tf.reshape(
            tf.convert_to_tensor(self.weights[2]),
            [self.weightDims[2][0], self.weightDims[2][1]]
        )

        lstmKernelHH = tf.reshape(
            tf.convert_to_tensor(self.weights[3]),
            [self.weightDims[3][0], self.weightDims[3][1]]
        )
        axis = 0
        self.lstmKernel = tf.concat(
            [lstmKernelXH, lstmKernelHH],
            axis=axis
        )

        self.lstmBias = tf.convert_to_tensor(
            self.weights[4],
            dtype=tf.float32
        )

        self.rawVars = [
            self.outputKernel,
            self.outputBias,
            self.lstmKernel,
            self.lstmBias
        ]
        print("hello")

sketchy = SketchRNN("url")
info : SketchRNNInfo = {
    "max_seq_len":130,
    "mode":2,
    "name":"cat",
    "scale_factor":82.2,
    "version":6,
}

weights = [[512,123],[123],[5,2048],[512,2048],[2048]]
with open("weight.txt", "r") as f:
    weightString = f.read().strip()

sketchy.instantiateFromJSON(info, weights, [weightString])

