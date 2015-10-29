import numpy as np

from lstm import LstmParam, LstmNetwork

import pdb as pdb

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label, idx):
        return (pred[idx] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label, idx):
        diff = np.zeros_like(pred)
        diff[idx] = 2 * (pred[idx] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(3)

    ## parameters for input data dimension and lstm cell count 
    #Number of iterations or epochs
    nEpochs = 100;

    nCells = 100
    
    # Number of random input numbers for each output
    xSize = 50
    
    concat_len = xSize + nCells

    # Minimise cell index number
    LossIdx = 0;

    ## Initialise parameters
    # Containg weights and derivatives of loss function wrt weights)
    PARAMS = LstmParam(nCells, xSize)
    
    
    ## Prepare target outputs
    y_list = [0.5, 0.2, 0.1, 0.5]
    #y_list = [0.12345]
    nOut   = len(y_list)

    # Initialise LSTM 
    LSTM = LstmNetwork(PARAMS, nOut)

    # Input data
    input_val_arr = np.random.random([nOut, xSize])

    #pdb.set_trace()
    
    # Train and sample at the same time
    for epoch in range(nEpochs):

        for ind in range(nOut):

            # Input 50 random numbers to LSTM
            x = input_val_arr[ind]
            LSTM.forwardPass(x)
            #print x

            # Get state which is the prediction
            state = LSTM.CELLS[ind].state.h[0]
            #print LSTM.CELLS[ind].state.h
            #pdb.set_trace()

            #print "  y_pred[%d] : %3.3f" % (ind, state)
            print "  Input %d rand.  Target = %1.3f. Output = %1.3f. Delta = %1.3f" % (xSize, y_list[ind], state, y_list[ind]-state)

            #pdb.set_trace()

        # Evaluate loss function and sample
        loss = LSTM.bptt(y_list, ToyLossLayer, LossIdx)
        print "Epoch: %3d. loss: %5.10f\n" % (epoch, loss)

        # Apply weight update
        PARAMS.apply_diff(lr=0.1)

        # Clear inputs to start afresh for next epoch
        LSTM.gotoStartCell()

if __name__ == "__main__":
    example_0()

#   Input 50 rand.  Target = 0.500. Output = 0.501. Delta = -0.001
#   Input 50 rand.  Target = 0.200. Output = 0.200. Delta = 0.000
#   Input 50 rand.  Target = 0.100. Output = 0.101. Delta = -0.001
#   Input 50 rand.  Target = 0.500. Output = 0.499. Delta = 0.001
# Epoch:  99. loss: 0.0000022563




