import random
import numpy as np
import math
import pdb as pdb

# See arXiv:1506.00019 for notation
# Check out http://kbullaughey.github.io/lstm-play/lstm/
# https://apaszke.github.io/lstm-explained.html


def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

# createst uniform random array w/ values in [a,b) and shape args
def randArr(a, b, *args): 
    np.random.seed(3)
    return np.random.rand(*args) * (b - a) + a




class LstmParam:
    """
    All LSTM network parameters
    """

    def __init__(self, nCells, xSize):

        print "__init__ LstmParam"

        self.xSize  = xSize
        self.nCells = nCells
        concat_len  = xSize + nCells

        ##
        # Weight matrices describe the linear fransformation from 
        # input space to output space.

        # Input weights
        self.Wgx = randArr(-0.1, 0.1, nCells, xSize )
        self.Wgh = randArr(-0.1, 0.1, nCells, nCells)

        # Input gate weights
        self.Wix = randArr(-0.1, 0.1, nCells, xSize )
        self.Wih = randArr(-0.1, 0.1, nCells, nCells)

        # Forget gate weights
        self.Wfx = randArr(-0.1, 0.1, nCells, xSize )
        self.Wfh = randArr(-0.1, 0.1, nCells, nCells)

        # Output gate weights
        self.Wox = randArr(-0.1, 0.1, nCells, xSize )
        self.Woh = randArr(-0.1, 0.1, nCells, nCells)



        self.Wg  = randArr(-0.1, 0.1, nCells, concat_len)
        self.Wi  = randArr(-0.1, 0.1, nCells, concat_len)
        self.Wf  = randArr(-0.1, 0.1, nCells, concat_len)
        self.Wo  = randArr(-0.1, 0.1, nCells, concat_len)

        ## bias terms
        self.Bg  = randArr(-0.1, 0.1, nCells)
        self.Bi  = randArr(-0.1, 0.1, nCells)
        self.Bf  = randArr(-0.1, 0.1, nCells)
        self.Bo  = randArr(-0.1, 0.1, nCells)


        # diffs (derivative of loss function w.r.t. all parameters)
        self.dWg  = np.zeros_like(self.Wg )
        self.dWgx = np.zeros_like(self.Wgx)
        self.dWgh = np.zeros_like(self.Wgh)
        
        self.dWi  = np.zeros_like(self.Wi )
        self.dWix = np.zeros_like(self.Wix)
        self.dWih = np.zeros_like(self.Wih)
        
        self.dWf  = np.zeros_like(self.Wf )
        self.dWfx = np.zeros_like(self.Wfx)
        self.dWfh = np.zeros_like(self.Wfh)
        
        self.dWo  = np.zeros_like(self.Wo )
        self.dWox = np.zeros_like(self.Wox)
        self.dWoh = np.zeros_like(self.Woh)

        # [100, 1]
        self.dBg  = np.zeros_like(self.Bg)
        self.dBi  = np.zeros_like(self.Bi)
        self.dBf  = np.zeros_like(self.Bf)
        self.dBo  = np.zeros_like(self.Bo)


    def apply_diff(self, lr = 1):
        """
        Weight update
        """
        # [150, 100]
        self.Wg  -= lr * self.dWg
        self.Wgx -= lr * self.dWgx
        self.Wgh -= lr * self.dWgh
        
        self.Wi  -= lr * self.dWi
        self.Wix -= lr * self.dWix
        self.Wih -= lr * self.dWih
        
        self.Wf  -= lr * self.dWf
        self.Wfx -= lr * self.dWfx
        self.Wfh -= lr * self.dWfh
        
        self.Wo  -= lr * self.dWo
        self.Wox -= lr * self.dWox
        self.Woh -= lr * self.dWoh

        # [100 , 1]
        self.Bg  -= lr * self.dBg
        self.Bi  -= lr * self.dBi
        self.Bf  -= lr * self.dBf
        self.Bo  -= lr * self.dBo
        
        # reset derivatives to zero

        # [150, 100]
        self.dWg  = np.zeros_like(self.Wg)
        self.dWgx = np.zeros_like(self.Wgx)
        self.dWgh = np.zeros_like(self.Wgh)
        
        self.dWi  = np.zeros_like(self.Wi)
        self.dWix = np.zeros_like(self.Wix)
        self.dWih = np.zeros_like(self.Wih)
        
        self.dWf  = np.zeros_like(self.Wf)
        self.dWfx = np.zeros_like(self.Wfx)
        self.dWfh = np.zeros_like(self.Wfh)
        
        self.dWo  = np.zeros_like(self.Wo)
        self.dWox = np.zeros_like(self.Wox)
        self.dWoh = np.zeros_like(self.Woh)

        # [100, 1]
        self.dBg = np.zeros_like(self.Bg)
        self.dBi = np.zeros_like(self.Bi)
        self.dBf = np.zeros_like(self.Bf)
        self.dBo = np.zeros_like(self.Bo)



class CellState:
    """
    State associated with an LSTM node
    """

    def __init__(self, nCells, xSize):
        print "__init__ CellState"

        # N dimensional vectors
        self.g = np.zeros(nCells) # g - cell input
        self.i = np.zeros(nCells) # i - input gate
        self.f = np.zeros(nCells) # f - forget gate
        self.o = np.zeros(nCells) # o - output gate
        self.s = np.zeros(nCells) # s - cell state
        self.h = np.zeros(nCells) # h - cell output

        # Persistent state for derivatives
        self.dh = np.zeros_like(self.h)
        self.ds = np.zeros_like(self.s)
        self.dx = np.zeros(xSize)




class LstmCell:
    """
    A single LSTM cell composed of State and Weight parameters.
    Function methods define the forward and backward passes
    """

    def __init__(self, PARAMS):

        print "__init__ LstmCell"

        # store reference to parameters and to activations
        self.state = CellState(PARAMS.nCells, PARAMS.xSize)
        self.param = PARAMS

        # non-recurrent input to node
        self.x  = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None

        # States

    def forwardPass(self, x, s_prev = None, h_1 = None):
        """
        Present data to the bottom of the Cell and compute the values as we
          forwardPass 'upwards'.
        Old name : bottom_data_is
        """
        # save data for use in backprop
        # [100, 1]
        self.s_prev = s_prev
        self.h_prev = h_1

        # concatenate x(t) and h(t-1)
        # [150 , 1]
        xc      = np.hstack((x,  h_1))
        self.xc = xc
        
        # Apply cell equations to new weights and inputs
        # [100, 1] here

        DP = np.dot

        Wg = self.param.Wg
        Wi = self.param.Wi
        Wf = self.param.Wf
        Wo = self.param.Wo

        Wgx = self.param.Wgx
        Wgh = self.param.Wgh
        Wix = self.param.Wix
        Wih = self.param.Wih
        Wfx = self.param.Wfx
        Wfh = self.param.Wfh
        Wox = self.param.Wox
        Woh = self.param.Woh

        Bg  = self.param.Bg
        Bi  = self.param.Bi
        Bf  = self.param.Bf
        Bo  = self.param.Bo

        #pdb.set_trace()
        self.state.g = np.tanh( DP(Wg,xc) + Bg )  # cell input
        self.state.i = sigmoid( DP(Wi,xc) + Bi )  #    input gate
        self.state.f = sigmoid( DP(Wf,xc) + Bf )  #    forget gate
        self.state.o = sigmoid( DP(Wo,xc) + Bo )  #    output gate
        
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f # cell state
        self.state.h = self.state.s * self.state.o                         # cell output

        

    
    def backwardPass(self, diff_h, diff_s):
        # notice that diff_s is carried along the constant error carousel

        # All [nMemCells ,1] == [100,1] here
        ds = self.state.o * diff_h + diff_s
        do = self.state.s * diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev  * ds

        # diffs w.r.t. vector inside sigma / tanh function

        # [100,1] here
        di_input = (1. - self.state.i) * self.state.i * di 
        df_input = (1. - self.state.f) * self.state.f * df 
        do_input = (1. - self.state.o) * self.state.o * do 
        dg_input = (1. - self.state.g ** 2) * dg # Tanh backprop here

        # diffs w.r.t. inputs
        # [100,150] here
        self.param.dWi += np.outer(di_input, self.xc)
        self.param.dWf += np.outer(df_input, self.xc)
        self.param.dWo += np.outer(do_input, self.xc)
        self.param.dWg += np.outer(dg_input, self.xc)

        # All [nMemCells ,1] == [100,1] here
        self.param.dBi += di_input
        self.param.dBf += df_input       
        self.param.dBo += do_input
        self.param.dBg += dg_input       

        # compute bottom diff
        # [150, 1]
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.Wi.T, di_input)
        dxc += np.dot(self.param.Wf.T, df_input)
        dxc += np.dot(self.param.Wo.T, do_input)
        dxc += np.dot(self.param.Wg.T, dg_input)

        # save bottom diffs
        # [100, 1]
        self.state.ds = ds * self.state.f

        # [50 , 1]
        self.state.dx = dxc[:self.param.xSize]

        # [100  1]
        self.state.dh = dxc[self.param.xSize:]


class LstmNetwork():
    def __init__(self, PARAMS, nOut):
        """
        Initialise LSTM network 
        """
        print "__init__ LstmNetwork"

        # Total number of cells in network
        self.nCells = nOut

        # Init parameters structure
        self.PARAMS = PARAMS

        # Create network of cells
        self.CELLS = []
        for _ in range(self.nCells):
            newCell  = LstmCell(self.PARAMS)
            self.CELLS.append(newCell)

        # Current number of used cells in network
        self.nUsedCells = 0


    def bptt(self, y_list, LOSS_LAYER, lossIdx):
        """
        Updates diffs by setting target sequence 
        with corresponding loss layer. 
        Will *NOT* update parameters.  To update parameters,
        call self.PARAMS.apply_diff() 
        """

        assert len(y_list) == self.nUsedCells
        idx = self.nUsedCells - 1

        # first node only gets diffs from label ...
        pred    = self.CELLS[idx].state.h
        label   = y_list[idx],

        loss    = LOSS_LAYER.loss(        pred, label, lossIdx )

        diff_h  = LOSS_LAYER.bottom_diff( pred, label, lossIdx ) # derivative of loss function

        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.PARAMS.nCells)

        self.CELLS[idx].backwardPass(diff_h, diff_s)

        idx -= 1
        lidx = idx

        while lidx >= 0: # loop through every cell

            pred    = self.CELLS[lidx].state.h
            label   = y_list[lidx],

            loss   += LOSS_LAYER.loss(        pred, label, lossIdx )

            lidx -= 1 



        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0: # loop through every cell

            pred    = self.CELLS[idx].state.h
            label   = y_list[idx],

            #loss   += LOSS_LAYER.loss(        pred, label, lossIdx )

            diff_h  = LOSS_LAYER.bottom_diff( pred, label, lossIdx )
            diff_h += self.CELLS[idx + 1].state.dh

            diff_s  = self.CELLS[idx + 1].state.ds

            self.CELLS[idx].backwardPass(diff_h, diff_s)
            idx -= 1 

        return loss

    def gotoStartCell(self):
        """
        Reset counter to go to first cell in network
        """
        self.nUsedCells = 0

    def forwardPass(self, x):
        """
        Apply input to network
        """

        # get index of current cell
        idx = self.nUsedCells

        if self.nUsedCells == 0: 
            # no recurrent inputs yet
            s_prev = np.zeros_like(self.CELLS[idx].state.s)
            h_prev = np.zeros_like(self.CELLS[idx].state.h)
        else: 
            # use recurrent inputs
            s_prev = self.CELLS[idx - 1].state.s
            h_prev = self.CELLS[idx - 1].state.h

        # Apply data to the current LSTM cell moving from bottom to top

        #pdb.set_trace()
        self.CELLS[idx].forwardPass(x, s_prev, h_prev)

        # Increment number of active cells in network
        self.nUsedCells += 1

        # Debug
        #print ("h[0] = %d") % (self.CELLS[idx-1].state.h)
        #print (self.CELLS[idx].state.h[0])
