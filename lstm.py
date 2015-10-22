import random
import numpy as np
import math
import pdb as pdb

# See arXiv:1506.00019 for notation

def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args): 
    return np.random.rand(*args) * (b - a) + a




class LstmParam:
    """
    All LSTM network parameters
    """

    def __init__(self, mem_cell_ct, x_dim):

        print "__init__ LstmParam"

        self.x_dim       = x_dim
        self.mem_cell_ct = mem_cell_ct
        concat_len       = x_dim + mem_cell_ct

        # weight matrices describe the linear fransformation from 
        # input space to output space.
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = 1):
        """
        Weight update
        """
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 




class CellState:
    """
    State associated with an LSTM node
    """

    def __init__(self, mem_cell_ct, x_dim):
        print "__init__ CellState"

        # N dimensional vectors
        self.g = np.zeros(mem_cell_ct) # cell input
        self.i = np.zeros(mem_cell_ct) # input gate
        self.f = np.zeros(mem_cell_ct) # forget gate
        self.o = np.zeros(mem_cell_ct) # output gate
        self.s = np.zeros(mem_cell_ct) # cell state
        self.h = np.zeros(mem_cell_ct) # cell output
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        self.bottom_diff_x = np.zeros(x_dim)




class LstmCell:
    """
    A single LSTM cell composed of State and Weight parameters
    """

    def __init__(self, PARAMS):

        print "__init__ LstmCell"

        # store reference to parameters and to activations
        self.state = CellState(PARAMS.mem_cell_ct, PARAMS.x_dim)
        self.param = PARAMS

        # non-recurrent input to node
        self.x = None
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def step(self, x, s_prev = None, h_prev = None):
        """
        Present data to the bottom of the Cell and compute the values as we
          step 'upwards'.
        Old name : bottom_data_is
        """
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc      = np.hstack((x,  h_prev))
        self.xc = xc
        
        # Apply cell equations to new weights and inputs
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)  # cell input
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)  #    input gate
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)  #    forget gate
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)  #    output gate
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f #    cell state
        self.state.h = self.state.s * self.state.o                         # cell output

        #pdb.set_trace()

    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = (1. - self.state.i) * self.state.i * di 
        df_input = (1. - self.state.f) * self.state.f * df 
        do_input = (1. - self.state.o) * self.state.o * do 
        dg_input = (1. - self.state.g ** 2) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_x = dxc[:self.param.x_dim]
        self.state.bottom_diff_h = dxc[self.param.x_dim:]




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


    def y_list_is(self, y_list, LOSS_LAYER, lossIdx):
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
        diff_h  = LOSS_LAYER.bottom_diff( pred, label, lossIdx )

        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.PARAMS.mem_cell_ct)

        self.CELLS[idx].top_diff_is(diff_h, diff_s)

        idx -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while idx >= 0:

            pred    = self.CELLS[idx].state.h
            label   = y_list[idx],


            loss    += LOSS_LAYER.loss(        pred, label, lossIdx )

            diff_h   = LOSS_LAYER.bottom_diff( pred, label, lossIdx )
            diff_h += self.CELLS[idx + 1].state.bottom_diff_h

            diff_s  = self.CELLS[idx + 1].state.bottom_diff_s

            self.CELLS[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 

        return loss

    def gotoStartCell(self):
        """
        Reset counter to go to first cell in network
        """
        self.nUsedCells = 0

    def input(self, x):
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
        self.CELLS[idx].step(x, s_prev, h_prev)

        # Increment number of active cells in network
        self.nUsedCells += 1

        # Debug
        #print ("h[0] = %d") % (self.CELLS[idx-1].state.h)
        #print (self.CELLS[idx].state.h[0])
