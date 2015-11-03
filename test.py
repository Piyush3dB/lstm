import numpy as np

from lstm import LstmParam, LstmNetwork

import pdb as pdb

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

    ## parameters for input data dimension and lstm cell count 
    #Number of iterations or epochs
    nEpochs = 100;

    # Internal cell widths
    cellWidth = 100
    
    # Number of random input numbers for each output
    xSize = 50
    
    #concat_len = xSize + cellWidth

    ## Initialise parameters
    # Containg weights and derivatives of loss function wrt weights)
    PARAMS = LstmParam(cellWidth, xSize)
    
    
    ## Prepare target outputs
    y_list  = [0.5, 0.2, 0.1, 0.5]
    #y_list = [0.12345]
    ySize   = len(y_list) # 4
    nCells  = ySize

    # Initialise LSTM 
    LSTM = LstmNetwork(PARAMS, ySize)

    # Input data
    inData = np.random.random([ySize, xSize]) # [4, 50]

    #pdb.set_trace()
    
    # Train and sample at the same time
    for epoch in range(nEpochs):


        # Input data
        for ind in range(ySize):
            # Input 50 random numbers to LSTM
            x = inData[ind]
            LSTM.forwardPass(x)


        # Sample from model
        state = LSTM.sample()
        for ind in range(nCells):
            print "  Input %d rand.  Target = %1.3f. Output = %1.3f. Delta = %1.3f" % (xSize, y_list[ind], state[ind], y_list[ind]-state[ind])


        # Evaluate loss function and back propagate through time
        loss = LSTM.bptt(y_list, ToyLossLayer)
        print "Epoch: %3d. loss: %5.10f\n" % (epoch, loss)


        # Apply weight update
        PARAMS.apply_diff(lr=0.1)


        # Clear inputs to start afresh for next epoch
        LSTM.gotoFirstCell()


if __name__ == "__main__":
    example_0()

#   Input 50 rand.  Target = 0.500. Output = 0.501. Delta = -0.001
#   Input 50 rand.  Target = 0.200. Output = 0.200. Delta = 0.000
#   Input 50 rand.  Target = 0.100. Output = 0.101. Delta = -0.001
#   Input 50 rand.  Target = 0.500. Output = 0.499. Delta = 0.001
# Epoch:  99. loss: 0.0000022563




