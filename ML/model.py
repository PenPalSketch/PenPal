import numpy as np
import tensorflow as tf
import math

NMIXTURE = 20 # M from the paper (amount of datapoints)


class LSTMState:
    def __init__(self, c, h):
        self.c = c
        self.h = h
    
class SketchRNN:


    '''
    Given the RNN state, returns the probability distribution function (pdf)
    of the next stroke. Optionally adjust the temperature of the pdf here.
   
    @param state previous LSTMState.
    @param temperature (Optional) for dx and dy (default 0.65)
    @param softmaxTemperature (Optional) for Pi and Pen discrete states
    (default is temperature * 0.5 + 0.5, which is a nice heuristic.)
   
    @returns StrokePDF (pi, muX, muY, sigmaX, sigmaY, corr, pen)
    '''
    def getPDF(self, state: LSTMState, temperature: int, softmaxTemperature: int):

        temp = temperature
        discreteTemp = 0.5 + 0.5 * temperature
        if softmaxTemperature:
            discreteTemp = softmaxTemperature

        NOUT = NMIXTURE

        h = tf.reshape(tf.convert_to_tensor(state.h, dtype=tf.float32),
                       (1, self.numUnits))

        sqrttemp = math.sqrt(temp)

        z = tf.squeeze(tf.matmul(h, self.outputKernel) + self.outputBias)

        rawPen, rst = tf.split(z, [3, NOUT*6])
        rawPi, mu1, mu2, rawSigma1, rawSigma2, rawCorr = tf.split(rst, 6)
        
        pen = tf.nn.softmax(rawPen / discreteTemp)
        pi = tf.nn.softmax(rawPi / discreteTemp)
        sigma1 = tf.exp(rawSigma1) * sqrttemp
        sigma2 = tf.exp(rawSigma2) * sqrttemp
        corr = tf.tanh(rawCorr)

        pdf = {
            'pi': pi.numpy(), # convert to a python list of numbers
            'muX': mu1.numpy(),
            'muY': mu2.numpy(),
            'sigmaX': sigma1.numpy(),
            'sigmaY': sigma2.numpy(),
            'corr': corr.numpy(),
            'pen': pen.numpy(),
        }

        return pdf

    '''
    Returns the zero/initial state of the model
   
    @returns zero state of the lstm: [c, h], where c and h are zero vectors.
    '''
    def zeroState(self):
        result = LSTMState(
            np.zeros(self.numUnits, dtype=np.float32),
            np.zeros(self.numUnits, dtype=np.float32)
        )
        return result

    '''
    Returns a new copy of the rnn state

    @param rnnState original LSTMState

    @returns copy of LSTMState
    '''
    def copyState(self, rnnState: LSTMState):
        result = LSTMState(
            np.array(rnnState.c, dtype=np.float32),
            np.array(rnnState.h, dtype=np.float32),
        )
        return result

