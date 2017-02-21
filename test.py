import numpy as np
import pdb   as pdb
pds = pdb.set_trace
#import lstm
from lstm import LstmWeights, LstmNetwork


"""
H = 100 (Hidden state size == cell output size)
V = 50  (Input vocab size)
t = 4   (Sequence Length aka bucket size)

- Input 4 randomised sequences.
- Output hidden vector, of which the first element is relevant


"""


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred - label) ** 2

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
    weights = LstmWeights(cellWidth, xSize)
    
    ## Prepare target outputs
    outData  = [0.1, 0.2, 0.3, 0.4]
    
    # number of unfolded cells
    nCells   = len(outData)

    # Initialise LSTM 
    trainLSTM = LstmNetwork(weights, nCells, cellWidth, xSize)

    # Input data
    inData = np.random.random([nCells, xSize]) # [4, 50]
    #inData = np.ones([nCells, xSize]) # [4, 50]
    
    # Train and sample at the same time
    for epoch in range(nEpochs):

        #
        # Train model
        #

        # Input data and propagate forwards through time for 4 time steps
        trainLSTM.fwdProp(inData, weights)

        #pdb.set_trace()

        # Evaluate loss function and back propagate through time for 4 time steps
        loss, grads = trainLSTM.bptt(outData, ToyLossLayer, weights)

        # Clear inputs to start afresh for next epoch
        trainLSTM.gotoStartCell()

        # Collect the gradients


        # Apply weight update
        weights.update(grads, lr=0.1)
        #weights.weightUpdate(lr=0.1)


        #
        # Test model and print logging information
        #

        # Sample from new model configured with the trained weights
        testLSTM = LstmNetwork(weights, nCells, cellWidth, xSize)
        testLSTM.fwdProp(inData, weights)
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
#  Input 50 rand.  Target = 0.100. Output = 0.106. Delta = -0.006
#  Input 50 rand.  Target = 0.200. Output = 0.196. Delta = 0.004
#  Input 50 rand.  Target = 0.300. Output = 0.300. Delta = -0.000
#  Input 50 rand.  Target = 0.400. Output = 0.400. Delta = -0.000
#Epoch:  99. loss: 0.0000608719


#pdb.set_trace()

