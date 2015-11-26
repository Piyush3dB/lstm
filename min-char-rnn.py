"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

import numpy as np
from random import uniform
import pdb

# vocab_size == input_size



class networkGradients:
    """
    Weights and update function
    """


    def __init__(self, hidden_size, input_size):


        self.dWxh = np.zeros((hidden_size , input_size ))
        self.dWhh = np.zeros((hidden_size , hidden_size))
        self.dWhy = np.zeros((input_size  , hidden_size))
        self.dbh  = np.zeros((hidden_size , 1))
        self.dby  = np.zeros((input_size  , 1))


    def clip(self):
        for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(dparam, -5, 5, out=dparam)








class networkWeights:
    """
    Weights and update function
    """

    def __init__(self, hidden_size, input_size):

        ##
        # Weight matrices describe the linear fransformation from 
        # input space to output space.
        np.random.seed(3)
        self.Wxh = np.random.randn(hidden_size , input_size )*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size , hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(input_size  , hidden_size)*0.01 # hidden to output
        self.bh  = np.zeros((hidden_size , 1)) # hidden bias
        self.by  = np.zeros((input_size  , 1)) # output bias


        # Memory variables for AdaGrad
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad



    def adagradUpdate(self, grads, learning_rate = 1e-1):
        """
        Weight update using Adagrad 
        """

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.Wxh  , self.Whh  , self.Why  , self.bh  , self.by  ], 
                                      [grads.dWxh, grads.dWhh, grads.dWhy, grads.dbh, grads.dby], 
                                      [self.mWxh , self.mWhh , self.mWhy , self.mbh , self.mby ]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        




class CellState:
    """
    State associated with an RNN node
    """
    def __init__(self, cellWidth, xSize):
        name = 'state'


class RnnCell:
    """
    A single LSTM cell composed of State and Weight parameters.
    Function methods define the forward and backward passes
    """

    def __init__(self, hidden_size, input_size):

        # store reference to parameters and to activations
        #self.state = CellState(hidden_size, input_size)

        self.hidden_size = hidden_size
        self.input_size  = input_size 

        self.hs   = 0
        self.hs_1 = 0

    def forwardPass(self, inputs, hprev, weights):
        """
        Present data to the bottom of the Cell and compute the values as we
          forwardPass 'upwards'.
        Old name : bottom_data_is
        """
        
        DP = np.dot

        self.hs_1 = hprev

        Wxh = weights.Wxh
        Whh = weights.Whh
        Why = weights.Why

        bh = weights.bh
        by = weights.by

        # encode in 1-of-k representation
        xs = np.zeros((self.input_size,1))
        xs[inputs] = 1

        # hidden state
        self.hs = np.tanh(DP(Wxh, xs) + DP(Whh, hprev) + bh)
        
        # unnormalized log probabilities for next chars
        ys = DP(Why, self.hs) + by

        # probabilities for next chars
        self.ps = np.exp(ys) / np.sum(np.exp(ys))

        self.xs = xs


    
    def backwardPass(self, dy, dh_1, hs_1, weights):
        """
        Propagate error from output to inpits
        """

        DP = np.dot

        Wxh = weights.Wxh
        Whh = weights.Whh
        Why = weights.Why

        dWhy  = np.dot(dy, self.hs.T)

        dh    = np.dot(Why.T, dy) + dh_1
        
        dhraw = (1 - self.hs * self.hs) * dh

        dWxh  = DP(dhraw, self.xs.T)
        dWhh  = DP(dhraw, hs_1.T)

        dh_1  = DP(Whh.T, dhraw)

        dby = dy

        self.dWxh = dWxh
        self.dWhh = dWhh
        self.dWhy = dWhy
        self.dby  = dby 
        self.dbh  = dhraw

        self.dh_1 = dh_1




class Rnn:
    """
    A single LSTM cell composed of State and Weight parameters.
    Function methods define the forward and backward passes
    """

    def __init__(self, rnn_depth, hidden_size, input_size):

        self.name   = 'rnn'

        # Depth of RNN
        self.rnn_depth   = rnn_depth

        # Other params
        self.hidden_size = hidden_size
        self.input_size  = input_size

        # Create RNN network of cells
        self.CELLS = [];
        for _ in xrange(rnn_depth):
            newCell = RnnCell(hidden_size, input_size)
            self.CELLS.append(newCell)

        self.loss = 0



    def FPTT(self, inputs, hprev, weights):
        """
        Propagate inputs forwards with given weights
        """

        # forward pass
        for t in xrange(self.rnn_depth):
            self.CELLS[t].forwardPass(inputs[t], hprev, weights)
            hprev = self.CELLS[t].hs


    def BPTT(self, targets, weights):
        """
        ackpropagation through time
        """

        self.loss = 0
        grads     = networkGradients(self.hidden_size, self.input_size)
        dh_1      = np.zeros_like(self.CELLS[0].hs)
        
        # BPTT
        for t in reversed(xrange(self.rnn_depth)):

            # softmax (cross-entropy loss) and derivative
            self.loss += -np.log(self.CELLS[t].ps[ targets[t],0 ]) 
            dy     = np.copy(self.CELLS[t].ps)
            dy[targets[t]] -= 1 # backprop into y

            # Previous hidden state
            hs_1 = self.CELLS[t].hs_1

            # Back prop for this cell
            self.CELLS[t].backwardPass(dy, dh_1, hs_1, weights)

            # Hidden delta for this cell
            dh_1 = self.CELLS[t].dh_1

            # Gradient Accumulations
            grads.dWxh  += self.CELLS[t].dWxh
            grads.dWhh  += self.CELLS[t].dWhh
            grads.dWhy  += self.CELLS[t].dWhy
            grads.dby   += self.CELLS[t].dby
            grads.dbh   += self.CELLS[t].dbh


        # clip to mitigate exploding gradients
        grads.clip()

        self.hprev = self.CELLS[self.input_size].hs

        return self.hprev, grads



def sample(hprev, seed_ix, n, hidden_size, input_size, weights):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    """
    
    # Outputs
    ixes = []

    # Create instance of RNN cell
    samplerCell = RnnCell(hidden_size, input_size)

    # Sample 'n' number of characters from model
    for t in xrange(n):
        
        # Forward propagate input
        samplerCell.forwardPass(seed_ix, hprev, weights)
        hprev = samplerCell.hs
        
        # Cell output distribution
        p    = samplerCell.ps

        # Sample an index from this distribution
        ix = np.random.choice(range(input_size), p=p.ravel())
        
        # Set newly sampled index as input for next iteration
        seed_ix = ix
        
        # Save for caller
        ixes.append(ix)
    
    return ixes




# gradient checking
def gradCheck(inputs, target, hprev):
    global Wxh, Whh, Why, bh, by
    num_checks, delta = 10, 1e-5
    _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
    for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
        s0 = dparam.shape
        s1 = param.shape
        assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
        print name
        for i in xrange(num_checks):
            ri = int(uniform(0,param.size))
            # evaluate cost at [x + delta] and [x - delta]
            old_val = param.flat[ri]
            param.flat[ri] = old_val + delta
            cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
            param.flat[ri] = old_val - delta
            cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
            param.flat[ri] = old_val # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = dparam.flat[ri]
            grad_numerical = (cg0 - cg1) / ( 2 * delta )
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
            # rel_error should be on order of 1e-7 or less



## Entry point
def example_0():

    # data I/O
    data  = open('input2.txt', 'r').read() # should be simple plain text file
    chars = list(set(data))

    data_size = len(data)
    input_size = len(chars)

    print 'data has %d characters, %d unique.' % (data_size, input_size)
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    # Hyperparameters
    hidden_size   = 100   # size of hidden layer of neurons
    seq_length    = 25    # number of steps to unroll the RNN for
    learning_rate = 1e-1

    # RNN model parameters
    weights = networkWeights(hidden_size, input_size)


    n, p = 0, 0
    
    # Initialise smoothed loss
    smooth_loss = -np.log(1.0/input_size)*seq_length # loss at iteration 0

    keepGoing = True

    # Create instance of RNN
    rnnObj = Rnn(seq_length, hidden_size, input_size)

    # Main loop
    while keepGoing:

        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0: 
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            p = 0 # go from start of data

        # Prepare inputs and targets
        inputs  = [char_to_ix[ch] for ch in data[ p  :p+seq_length   ]]
        targets = [char_to_ix[ch] for ch in data[ p+1:p+seq_length+1 ]]

        # sample from the model now and then
        if n % 100 == 0:
            seedIn = inputs[0]
            sample_ix = sample(hprev, inputs[0], 200, hidden_size, input_size, weights)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print '----\n %s \n----' % (txt, )

        # forward seq_length characters through the net and fetch gradient
        rnnObj.FPTT(inputs, hprev, weights)
        hprev, grads = rnnObj.BPTT(targets, weights)
        loss = rnnObj.loss


        # Smooth and log message print
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
        
        # Adagrad weight update
        weights.adagradUpdate(grads)


        # move data pointer
        p += seq_length

        # iteration counter 
        n += 1

        # Terminate
        keepGoing = False if (n == 300) else True


if __name__ == "__main__":
    example_0()

    


#

# ----
# iter 200, loss: 77.081298
