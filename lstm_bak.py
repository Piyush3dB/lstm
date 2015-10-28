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
        self.Wgx = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim      )
        self.Wgh = rand_arr(-0.1, 0.1, mem_cell_ct, mem_cell_ct)

        self.Wix = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim      )
        self.Wih = rand_arr(-0.1, 0.1, mem_cell_ct, mem_cell_ct)

        self.Wfx = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim      )
        self.Wfh = rand_arr(-0.1, 0.1, mem_cell_ct, mem_cell_ct)

        self.Wox = rand_arr(-0.1, 0.1, mem_cell_ct, x_dim      )
        self.Woh = rand_arr(-0.1, 0.1, mem_cell_ct, mem_cell_ct)


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
        self.dWgx = np.zeros((mem_cell_ct, x_dim      )) 
        self.dwgh = np.zeros((mem_cell_ct, mem_cell_ct)) 

        self.dWix = np.zeros((mem_cell_ct, x_dim      )) 
        self.dwih = np.zeros((mem_cell_ct, mem_cell_ct)) 
        
        self.dWfx = np.zeros((mem_cell_ct, x_dim      )) 
        self.dwfh = np.zeros((mem_cell_ct, mem_cell_ct)) 
        
        self.dWox = np.zeros((mem_cell_ct, x_dim      )) 
        self.dwoh = np.zeros((mem_cell_ct, mem_cell_ct)) 
        

        self.dWg = np.zeros((mem_cell_ct, concat_len)) 
        self.dWi = np.zeros((mem_cell_ct, concat_len)) 
        self.dWf = np.zeros((mem_cell_ct, concat_len)) 
        self.dWo = np.zeros((mem_cell_ct, concat_len)) 


        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 

    def apply_diff(self, lr = 1):
        """
        Weight update
        """
        # [150, 100]
        self.wg -= lr * self.dWg
        self.wi -= lr * self.dWi
        self.wf -= lr * self.dWf
        self.wo -= lr * self.dWo

        # [100 , 1]
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        

        # reset diffs to zero

        # [150, 100]
        self.dWg = np.zeros_like(self.wg)
        self.dWi = np.zeros_like(self.wi) 
        self.dWf = np.zeros_like(self.wf) 
        self.dWo = np.zeros_like(self.wo) 

        # [100, 1]
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo) 

        #pdb.set_trace()




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

    def fwdPass(self, x, s_prev = None, h_prev = None):
        """
        Present data to the bottom of the Cell and compute the values as we
          fwdPass 'upwards'.
        Old name : bottom_data_is
        """
        # save data for use in backprop
        # [100, 1]
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        # [150 , 1]
        xc      = np.hstack((x,  h_prev))
        self.xc = xc
        
        # Apply cell equations to new weights and inputs
        # [100, 1] here
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)  # cell input
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)  #    input gate
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)  #    forget gate
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)  #    output gate
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f #    cell state
        self.state.h = self.state.s * self.state.o                         # cell output

        

    
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel

        # All [nMemCells ,1] == [100,1] here
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function

        # [100,1] here
        di_input = (1. - self.state.i) * self.state.i * di 
        df_input = (1. - self.state.f) * self.state.f * df 
        do_input = (1. - self.state.o) * self.state.o * do 
        dg_input = (1. - self.state.g ** 2) * dg

        # diffs w.r.t. inputs
        # [100,150] here
        self.param.dWi += np.outer(di_input, self.xc)
        self.param.dWf += np.outer(df_input, self.xc)
        self.param.dWo += np.outer(do_input, self.xc)
        self.param.dWg += np.outer(dg_input, self.xc)

        # All [nMemCells ,1] == [100,1] here
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        # compute bottom diff
        # [150, 1]
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        # [100, 1]
        self.state.bottom_diff_s = ds * self.state.f

        # [50 , 1]
        self.state.bottom_diff_x = dxc[:self.param.x_dim]

        # [100  1]
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
        while idx >= 0: # loop through every cell

            pred    = self.CELLS[idx].state.h
            label   = y_list[idx],


            loss   += LOSS_LAYER.loss(        pred, label, lossIdx )

            diff_h  = LOSS_LAYER.bottom_diff( pred, label, lossIdx )
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
        self.CELLS[idx].fwdPass(x, s_prev, h_prev)

        # Increment number of active cells in network
        self.nUsedCells += 1

        # Debug
        #print ("h[0] = %d") % (self.CELLS[idx-1].state.h)
        #print (self.CELLS[idx].state.h[0])
