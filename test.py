import numpy as np
import pdb   as pdb
from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label, idx=0):
        return (pred[idx] - label) ** 2

    """
    Computes derivative of loss function
    """
    @classmethod
    def loss_derivative(self, pred, label, idx=0):
        diff = np.zeros_like(pred)
        diff[idx] = 2 * (pred[idx] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(3)

    # Number of iterations or epochs
    nEpochs = 100;

    # Internal cell widths
    cellWidth = 100
    
    # Number of random input numbers for each output
    xSize = 50

    ## Initialise parameters
    # Containg weights and derivatives of loss function wrt weights)
    PARAMS = LstmParam(cellWidth, xSize)
    
    ## Prepare target outputs
    outData  = [0.5, 0.2, 0.1, 0.5]
    ySize   = len(outData) # 4
    nCells  = ySize # number of unfolded cells

    # Initialise LSTM 
    LSTM = LstmNetwork(PARAMS, nCells)

    # Input data
    inData = np.random.random([ySize, xSize]) # [4, 50]
    
    # Train and sample at the same time
    for epoch in range(nEpochs):


        # Input data
        for ind in range(ySize):
            # Input 50 random numbers to LSTM
            x = inData[ind]
            LSTM.forward(x)


        # Evaluate loss function and back propagate through time
        loss = LSTM.bptt(outData, ToyLossLayer)

        # Apply weight update
        PARAMS.apply_diff(lr=0.1)

        # Sample from model
        testLSTM = LstmNetwork(PARAMS, nCells)
        state = testLSTM.sample()

        # Clear inputs to start afresh for next epoch
        LSTM.gotoFirstCell()

        #
        # Debug logging 
        #
        for ind in range(nCells):
            print "  Input %d rand.  Target = %1.3f. Output = %1.3f. Delta = %1.3f" % (xSize, outData[ind], state[ind], outData[ind]-state[ind])
        print "Epoch: %3d. loss: %5.10f\n" % (epoch, loss)


if __name__ == "__main__":
    example_0()

#   Input 50 rand.  Target = 0.500. Output = 0.501. Delta = -0.001
#   Input 50 rand.  Target = 0.200. Output = 0.200. Delta = 0.000
#   Input 50 rand.  Target = 0.100. Output = 0.101. Delta = -0.001
#   Input 50 rand.  Target = 0.500. Output = 0.499. Delta = 0.001
# Epoch:  99. loss: 0.0000022563


    #pdb.set_trace()

