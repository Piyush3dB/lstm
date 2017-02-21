import random
import numpy as np
import math
import pdb as pdb
pds = pdb.set_trace

# See arXiv:1506.00019 for notation
# Check out http://kbullaughey.github.io/lstm-play/lstm/
# https://apaszke.github.io/lstm-explained.html


def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

# createst uniform random array w/ values in [a,b) and shape args
def randArr(a, b, *args): 
    np.random.seed(3)
    return np.random.rand(*args) * (b - a) + a





class networkGradients:
    """
    Weights and update function
    """


    def __init__(self, cellWidth, xSize):

        concat_len  = xSize + cellWidth

        W  = randArr(-0.1, 0.1, cellWidth, concat_len)
        B  = randArr(-0.1, 0.1, cellWidth)


        # diffs (derivative of loss function w.r.t. all parameters)
        self.dWg  = np.zeros_like( W )
        self.dWi  = np.zeros_like( W )
        self.dWf  = np.zeros_like( W )
        self.dWo  = np.zeros_like( W )

        # [100, 1]
        self.dBg  = np.zeros_like( B )
        self.dBi  = np.zeros_like( B )
        self.dBf  = np.zeros_like( B )
        self.dBo  = np.zeros_like( B )




    def clip(self):
        for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(dparam, -5, 5, out=dparam)






class LstmWeights:
    """
    All LSTM network parameters
    """

    def __init__(self, cellWidth, xSize):

        #print "__init__ LstmParam"

        self.xSize     = xSize
        self.cellWidth = cellWidth
        concat_len     = xSize + cellWidth

        ##
        # Weight matrices describe the linear fransformation from 
        # input space to output space.
        self.Wg  = randArr(-0.1, 0.1, cellWidth, concat_len)
        self.Wi  = randArr(-0.1, 0.1, cellWidth, concat_len)
        self.Wf  = randArr(-0.1, 0.1, cellWidth, concat_len)
        self.Wo  = randArr(-0.1, 0.1, cellWidth, concat_len)

        ## bias terms
        self.Bg  = randArr(-0.1, 0.1, cellWidth)
        self.Bi  = randArr(-0.1, 0.1, cellWidth)
        self.Bf  = randArr(-0.1, 0.1, cellWidth)
        self.Bo  = randArr(-0.1, 0.1, cellWidth)


        # diffs (derivative of loss function w.r.t. all parameters)
        self.dWg  = np.zeros_like(self.Wg )
        self.dWi  = np.zeros_like(self.Wi )
        self.dWf  = np.zeros_like(self.Wf )
        self.dWo  = np.zeros_like(self.Wo )
        
        # [100, 1]
        self.dBg  = np.zeros_like(self.Bg)
        self.dBi  = np.zeros_like(self.Bi)
        self.dBf  = np.zeros_like(self.Bf)
        self.dBo  = np.zeros_like(self.Bo)


    def update(self, grads, lr=1):
        """
        Weight update
        """
        # [150, 100]
        self.Wg  -= lr * grads.dWg
        self.Wi  -= lr * grads.dWi
        self.Wf  -= lr * grads.dWf
        self.Wo  -= lr * grads.dWo

        # [100 , 1]
        self.Bg  -= lr * grads.dBg
        self.Bi  -= lr * grads.dBi
        self.Bf  -= lr * grads.dBf
        self.Bo  -= lr * grads.dBo
        




class CellState:
    """
    State associated with an LSTM node
    """

    def __init__(self, cellWidth, xSize):
        #print "__init__ CellState"

        # N dimensional vectors
        self.g = np.zeros(cellWidth) # g - cell input
        self.i = np.zeros(cellWidth) # i - input gate
        self.f = np.zeros(cellWidth) # f - forget gate
        self.o = np.zeros(cellWidth) # o - output gate
        self.s = np.zeros(cellWidth) # s - cell state
        self.h = np.zeros(cellWidth) # h - cell output

        # Persistent state for derivatives
        self.dh = np.zeros_like(self.h)
        self.ds = np.zeros_like(self.s)
        self.dx = np.zeros(xSize)




class LstmCell:
    """
    A single LSTM cell composed of State and Weight parameters.
    Function methods define the forward and backward passes
    """

    def __init__(self, cellWidth, xSize):

        #print "__init__ LstmCell"

        # store reference to parameters and to activations
        self.state = CellState(cellWidth, xSize)
        #self.param = PARAMS

        # non-recurrent input to node
        #self.x  = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None

        # States

    def forwardPass(self, x, weights, s_prev_cell = None, h_prev_cell = None):
        """
        Present data to the bottom of the Cell and compute the values as we
          forwardPass 'upwards'.
        Old name : bottom_data_is
        """
        # save data for use in backprop
        # [100, 1]
        self.s_prev_cell = s_prev_cell
        self.h_prev = h_prev_cell

        # concatenate x(t) and h(t-1)
        # [150 , 1]
        xc      = np.hstack((x,  h_prev_cell))
        self.xc = xc
        
        # Apply cell equations to new weights and inputs
        # [100, 1] here

        DP = np.dot

        Wg = weights.Wg
        Wi = weights.Wi
        Wf = weights.Wf
        Wo = weights.Wo

        Bg  = weights.Bg
        Bi  = weights.Bi
        Bf  = weights.Bf
        Bo  = weights.Bo

        #pdb.set_trace()
        self.state.g = np.tanh( DP(Wg,xc) + Bg )  # cell input
        self.state.i = sigmoid( DP(Wi,xc) + Bi )  #    input gate
        self.state.f = sigmoid( DP(Wf,xc) + Bf )  #    forget gate
        self.state.o = sigmoid( DP(Wo,xc) + Bo )  #    output gate
        
        self.state.s = self.state.g * self.state.i + s_prev_cell * self.state.f # cell state
        self.state.h = self.state.s * self.state.o                         # cell output

        #pds()


    def sample(self):
        """
        Sample from network cell
        """
        #print  self.state.h[0]
        return self.state.h[0]

    
    def backwardPass(self, diff_h, diff_s, weights):
        # notice that diff_s is carried along the constant error carousel

        # All [nMemCells ,1] == [100,1] here
        ds = self.state.o * diff_h + diff_s
        do = self.state.s * diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev_cell  * ds

        # diffs w.r.t. vector inside sigma / tanh function

        # [100,1] here
        self.di_input = (1. - self.state.i) * self.state.i * di 
        self.df_input = (1. - self.state.f) * self.state.f * df 
        self.do_input = (1. - self.state.o) * self.state.o * do 
        self.dg_input = (1. - self.state.g ** 2) * dg # Tanh backprop here

        # diffs w.r.t. inputs
        # [100,150] here
        self.dWi = np.outer(self.di_input, self.xc)
        self.dWf = np.outer(self.df_input, self.xc)
        self.dWo = np.outer(self.do_input, self.xc)
        self.dWg = np.outer(self.dg_input, self.xc)


        # compute bottom diff
        # [150, 1]
        dxc  = np.zeros_like(self.xc)
        dxc += np.dot(weights.Wi.T, self.di_input)
        dxc += np.dot(weights.Wf.T, self.df_input)
        dxc += np.dot(weights.Wo.T, self.do_input)
        dxc += np.dot(weights.Wg.T, self.dg_input)

        # save bottom diffs
        # [100, 1]
        self.state.ds = ds * self.state.f

        # [50 , 1]
        #self.state.dx = dxc[  : weights.xSize ]

        # [100  1]
        self.state.dh = dxc[ weights.xSize :  ]


class LstmNetwork():
    def __init__(self, PARAMS, nCells, cellWidth, xSize):
        """
        Initialise LSTM network 
        """
        #print "__init__ LstmNetwork"

        # Total number of unfolded cells in network
        self.nCells = nCells

        self.cellWidth = cellWidth
        self.xSize     = xSize

        # Create network of cells
        self.CELLS = []
        for _ in range(self.nCells):
            newCell  = LstmCell(cellWidth, xSize)
            self.CELLS.append(newCell)

        # Current number of used cells in network
        self.nUsedCells = 0

    def sample(self):
        """
        Sample from network cell
        """
        state = randArr(-0.1, 0.1, self.nCells)
        for ind in range(self.nCells):
            state[ind] = self.CELLS[ind].sample()

        return state


    def gotoStartCell(self):
        """
        Reset counter to go to first cell in unfolded network
        """
        self.nUsedCells = 0

    def fwdProp(self, x, weights):
        """
        Propagate inputs through unfolded network
        """

        # Initialise previous states for first cell
        s_prev = np.zeros(self.cellWidth)
        h_prev = np.zeros(self.cellWidth)

        # Forward propagate in time
        for idx in range(self.nCells):

            self.CELLS[idx].forwardPass(x[idx], weights, s_prev, h_prev)

            s_prev = self.CELLS[idx].state.s
            h_prev = self.CELLS[idx].state.h

            self.nUsedCells += 1




    def bptt(self, targetData, LOSS_LAYER, weights):
        """
        Back propagation through time through unfolded network.
        Updates derivatives by setting target sequence 
        with corresponding loss layer. 
        """

        assert len(targetData) == self.nUsedCells

        # Initialise derivative arrays
        diff_s = np.zeros(self.cellWidth)
        diff_h = np.zeros(self.cellWidth)

        # Local variables
        totalLoss = 0
        dh_prev   = 0
        ds_prev   = 0

        grads = networkGradients(self.cellWidth, self.xSize)

        # Back propagate gradients towards oldest cell
        for idx in reversed(range(self.nCells)):

            # Get target and prediction
            pred  = self.CELLS[idx].state.h
            label = targetData[idx]

            #pds()

            # Compute loss function and accumulate
            cellLoss   = LOSS_LAYER.loss( pred[0], label )
            totalLoss += cellLoss

            # Derivative of loss function and accumulate with previous derivative
            dh     = LOSS_LAYER.loss_derivative( pred, label )
            diff_h = dh + dh_prev

            # Propagate error along constant error carousel
            diff_s = ds_prev

            # Backprop for this cell
            self.CELLS[idx].backwardPass(diff_h, diff_s, weights)

            # Gradients Accumulation
            grads.dWi += self.CELLS[idx].dWi
            grads.dWf += self.CELLS[idx].dWf
            grads.dWo += self.CELLS[idx].dWo
            grads.dWg += self.CELLS[idx].dWg

            grads.dBi += self.CELLS[idx].di_input
            grads.dBf += self.CELLS[idx].df_input       
            grads.dBo += self.CELLS[idx].do_input
            grads.dBg += self.CELLS[idx].dg_input   


            # Save current derivatives for next cell
            dh_prev = self.CELLS[idx].state.dh
            ds_prev = self.CELLS[idx].state.ds

        return totalLoss, grads
