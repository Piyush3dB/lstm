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
    
    # number of unfolded cells
    nCells   = len(outData)

    # Initialise LSTM 
    trainLSTM = LstmNetwork(PARAMS, nCells)

    # Input data
    inData = np.random.random([nCells, xSize]) # [4, 50]
    
    # Train and sample at the same time
    for epoch in range(nEpochs):

        #
        # Train model
        #

        # Input data
        trainLSTM.fwdProp(inData)

        # Evaluate loss function and back propagate through time
        loss = trainLSTM.bptt(outData, ToyLossLayer)

        # Clear inputs to start afresh for next epoch
        trainLSTM.gotoFirstCell()

        # Apply weight update
        PARAMS.weightUpdate(lr=0.1)


        #
        # Test model and print logging information
        #

        # Sample from new model configured with the trained weights
        testLSTM = LstmNetwork(PARAMS, nCells)
        testLSTM.fwdProp(inData)
        state = testLSTM.sample()

        #pdb.set_trace()

        # Print logging information
        for ind in range(nCells):
            print "  Input %d rand.  Target = %1.3f. Output = %1.3f. Delta = %1.3f" % (xSize, float(outData[ind]), float(state[ind]), outData[ind]-state[ind])
        print "Epoch: %3d. loss: %5.10f\n" % (epoch, loss)


def gradientCheck():
    print "TODO"


if __name__ == "__main__":
    example_0()
    #gradientCheck()


## Expected final output if still working correctly
##
#   Input 50 rand.  Target = 0.500. Output = 0.501. Delta = -0.001
#   Input 50 rand.  Target = 0.200. Output = 0.200. Delta = 0.000
#   Input 50 rand.  Target = 0.100. Output = 0.101. Delta = -0.001
#   Input 50 rand.  Target = 0.500. Output = 0.499. Delta = 0.001
# Epoch:  99. loss: 0.0000022563

#pdb.set_trace()