import numpy as np

from lstm import LstmParam, LstmNetwork

import pdb as pdb

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(3)

    ## parameters for input data dimension and lstm cell count 
    #Number of iterations or epochs
    nEpochs = 100;
    
    mem_cell_ct = 100
    
    # Number of random input numbers for each output
    x_dim = 50
    
    concat_len = x_dim + mem_cell_ct

    ## Initialise parameters
    # Containg weights and derivatives of loss function wrt weights)
    PARAMS = LstmParam(mem_cell_ct, x_dim) 
    
    # Initialise LSTM 
    LSTM = LstmNetwork(PARAMS)
    
    ## Prepare target outputs
    y_list = [0.5, 0.2, 0.1, 0.5]
    nOut   = len(y_list)

    # Input data
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    #np.shape(input_val_arr) = (4,50)

    #pdb.set_trace()
    
    # Train and sample at the same time
    for epoch in range(nEpochs):

        print "Epoch: %3d" % (epoch)
        
        for ind in range(nOut):

            # Input 50 random numbers to LSTM
            xRandIn = input_val_arr[ind]
            LSTM.x_list_add(xRandIn)
            # Get state which is the prediction
            state = LSTM.lstm_node_list[ind].state.h[0]

            #print "  y_pred[%d] : %3.3f" % (ind, state)
            print "  Input %d rand.  Target = %1.3f. Output = %1.3f" % (x_dim, y_list[ind], state)

            #pdb.set_trace()

        # Evaluate loss function
        loss = LSTM.y_list_is(y_list, ToyLossLayer)
        print "loss: %5.10f\n" % (loss)

        # Apply weight update
        PARAMS.apply_diff(lr=0.1)

        # Clear inouts to start afresh for next epoch
        LSTM.x_list_clear()

if __name__ == "__main__":
    example_0()

