"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

import numpy as np
from random import uniform
import pdb

# vocab_size == input_size

class networkWeights:
    """
    Weights and update function
    """


    def __init__(self, hidden_size, input_size):

        #print "__init__ LstmParam"

        self.hidden_size = hidden_size
        self.input_size  = input_size


        ##
        # Weight matrices describe the linear fransformation from 
        # input space to output space.
        np.random.seed(3)

        self.weights = self._weights
        self.grads   = self._grads
        self.mem     = self._mem

        

    def _mem(self):
        # Memory variables for AdaGrad
        mWxh = np.zeros_like(self.weights.Wxh)
        mWhh = np.zeros_like(self.weights.Whh)
        mWhy = np.zeros_like(self.weights.Why)
        mbh  = np.zeros_like(self.weights.bh)
        mby  = np.zeros_like(self.weights.by)

    def _grads(self):
        dWxh = np.zeros_like(self.weights.Wxh)
        dWhh = np.zeros_like(self.weights.Whh)
        dWhy = np.zeros_like(self.weights.Why)
        dbh  = np.zeros_like(self.weights.bh)
        dby  = np.zeros_like(self.weights.by)

    def _weights(self):
        Wxh = np.random.randn(hidden_size , input_size )*0.01 # input to hidden
        Whh = np.random.randn(hidden_size , hidden_size)*0.01 # hidden to hidden
        Why = np.random.randn(input_size  , hidden_size)*0.01 # hidden to output
        bh  = np.zeros((hidden_size , 1)) # hidden bias
        by  = np.zeros((input_size  , 1)) # output bias



    def weightUpdate(self, learning_rate = 1e-1):
        """
        Weight update using Adagrad 
        """

        # perform parameter update with Adagrad
        for weights, grads, mem in zip([self.weights.Wxh, self.weights.Whh, self.weights.Why, self.weights.bh, self.weights.by], 
                                      [ self.grads.dWxh, self.grads.dWhh, self.grads.dWhy, self.grads.dbh, self.grads.dby], 
                                      [ self.mem.mWxh  , self.mem.mWhh  , self.mem.mWhy  , self.mem.mbh  , self.mem.mby  ]):
            mem += grads * grads
            weights += -learning_rate * grads / np.sqrt(mem + 1e-8) # adagrad update
        
        # reset derivatives to zero
        Z = np.zeros_like
        # diffs (derivative of loss function w.r.t. all parameters)
        self.grads.dWxh = np.zeros_like(self.weights.Wxh)
        self.grads.dWhh = np.zeros_like(self.weights.Whh)
        self.grads.dWhy = np.zeros_like(self.weights.Why)
        self.grads.dbh  = np.zeros_like(self.weights.bh)
        self.grads.dby  = np.zeros_like(self.weights.by)




class RnnParam:
    """
    All LSTM network parameters
    """

    def __init__(self, hidden_size, input_size):

        #print "__init__ LstmParam"

        #self.hidden_size = hidden_size
        #self.input_size  = input_size

        ##
        # Weight matrices describe the linear fransformation from 
        # input space to output space.
        np.random.seed(3)
        self.Wxh = np.random.randn(hidden_size , input_size )*0.01 # input to hidden
        self.Whh = np.random.randn(hidden_size , hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(input_size  , hidden_size)*0.01 # hidden to output
        self.bh  = np.zeros((hidden_size , 1)) # hidden bias
        self.by  = np.zeros((input_size  , 1)) # output bias

        # diffs (derivative of loss function w.r.t. all parameters)
        self.dWxh, self.dWhh, self.dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.dbh, self.dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        # Memory variables for AdaGrad
        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad



    def weightUpdate(self, learning_rate = 1e-1):
        """
        Weight update using Adagrad 
        """

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                      [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby], 
                                      [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        
        # reset derivatives to zero
        Z = np.zeros_like
        self.dWxh, self.dWhh, self.dWhy = Z(self.Wxh), Z(self.Whh), Z(self.Why)
        self.dbh, self.dby = Z(self.bh), Z(self.by)


class CellState:
    """
    State associated with an RNN node
    """
    def __init__(self, cellWidth, xSize):
        name = 'state'
        #print "__init__ CellState"
        # N dimensional vectors
        #self.h = np.zeros(cellWidth) # h - cell output
        # Persistent state for derivatives
        #self.dh = np.zeros_like(self.h)


class RnnCell:
    """
    A single LSTM cell composed of State and Weight parameters.
    Function methods define the forward and backward passes
    """

    def __init__(self, hidden_size, input_size):
        #print "__init__ LstmCell"

        # store reference to parameters and to activations
        self.state = CellState(hidden_size, input_size)

    def forwardPass(self, inputs, hprev, weights):
        """
        Present data to the bottom of the Cell and compute the values as we
          forwardPass 'upwards'.
        Old name : bottom_data_is
        """
        
        DP = np.dot

        Wxh = weights.Wxh
        Whh = weights.Whh
        Why = weights.Why

        bh = weights.bh
        by = weights.by

        # encode in 1-of-k representation
        xs = np.zeros((input_size,1))
        xs[inputs] = 1

        # hidden state
        self.hs = np.tanh(DP(Wxh, xs) + DP(Whh, hprev) + bh)
        
        # unnormalized log probabilities for next chars
        ys = DP(Why, self.hs) + by

        # probabilities for next chars
        self.ps = np.exp(ys) / np.sum(np.exp(ys))

        self.xs = xs


    
    def backwardPass(self, dy, dh_1, hs_1, weights):
        # notice that diff_s is carried along the constant error carousel

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

    def __init__(self, PARAMS, rnn_depth, hidden_size, input_size):

        self.name   = 'rnn'
        self.PARAMS = PARAMS

        # Depth of RNN
        self.rnn_depth = rnn_depth

        # Create RNN network of cells
        self.CELLS = [];
        for _ in xrange(rnn_depth):
            newCell = RnnCell(hidden_size, input_size)
            self.CELLS.append(newCell)

        self.loss = 0



    def lossFunModif(self, inputs, targets, hprev, weights):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        hs = {}
        hs[-1] = np.copy(hprev)

        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby         = np.zeros_like(bh), np.zeros_like(by)
        
        ###
        ###



        
        ####
        # forward pass
        for t in xrange(self.rnn_depth):

          self.CELLS[t].forwardPass(inputs[t], hprev, weights)

          hs[t] = self.CELLS[t].hs
          hprev = self.CELLS[t].hs
          

        ####
        # network loss and derivative computation
        loss = 0
        for t in xrange(self.rnn_depth):
            # softmax (cross-entropy loss)
            thisLoss = -np.log(self.CELLS[t].ps[ targets[t],0 ]) 
            loss += thisLoss

        self.loss = loss
        

        ####
        # backward pass: compute gradients going backwards
        dh_1 = np.zeros_like(self.CELLS[0].hs)
        
        for t in reversed(xrange(self.rnn_depth)):
            # Derivative calculation
            dy     = np.copy(self.CELLS[t].ps)
            dy[targets[t]] -= 1 # backprop into y

            # Previous hidden state
            hs_1 = hs[t-1]

            # Back prop for this cell
            self.CELLS[t].backwardPass(dy, dh_1, hs_1, weights)

            # Hidden delta for this cell
            dh_1 = self.CELLS[t].dh_1

            # Accumulators
            dWxh  += self.CELLS[t].dWxh
            dWhh  += self.CELLS[t].dWhh
            dWhy  += self.CELLS[t].dWhy
            dby   += self.CELLS[t].dby
            dbh   += self.CELLS[t].dbh


        # clip to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        self.dWxh = dWxh
        self.dWhh = dWhh
        self.dWhy = dWhy
        self.dby  = dby 
        self.dbh  = dbh 


        #pdb.set_trace()
        
        #return dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
        return hs[len(inputs)-1]



def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  ####
  # forward pass
  for t in xrange(len(inputs)):
    
    xs[t] = np.zeros((input_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1

    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  ####
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  
  for t in reversed(xrange(len(inputs))):
    dy     = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy  += np.dot(dy, hs[t].T)
    dby   += dy
    dh     = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw  = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh   += dhraw
    dWxh  += np.dot(dhraw, xs[t].T)
    dWhh  += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((input_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(input_size), p=p.ravel())
    x = np.zeros((input_size, 1))
    x[ix] = 1
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



###### Main function here
###### Main function here
###### Main function here


np.random.seed(3)

# data I/O
data  = open('input2.txt', 'r').read() # should be simple plain text file
chars = list(set(data))

data_size, input_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, input_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Weights = networkWeights(hidden_size, input_size)

PARAM = RnnParam(hidden_size, input_size)
Wxh = PARAM.Wxh
Whh = PARAM.Whh
Why = PARAM.Why
bh  = PARAM.bh 
by  = PARAM.by 






n, p = 0, 0
smooth_loss = -np.log(1.0/input_size)*seq_length # loss at iteration 0

keepGoing = True


rnnObj = Rnn(PARAM, seq_length, hidden_size, input_size)

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
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )

    # forward seq_length characters through the net and fetch gradient
    #pdb.set_trace()
    hprev = rnnObj.lossFunModif(inputs, targets, hprev, PARAM)
    loss = rnnObj.loss
    PARAM.dWxh = rnnObj.dWxh
    PARAM.dWhh = rnnObj.dWhh
    PARAM.dWhy = rnnObj.dWhy
    PARAM.dbh  = rnnObj.dbh
    PARAM.dby  = rnnObj.dby

    #rnnObj.resetGrads()

    # Smooth and log message print
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
    
    # Adagrad weight update
    PARAM.weightUpdate()


    # move data pointer
    p += seq_length

    # iteration counter 
    n += 1

    # Terminate
    if (n == 300):
        keepGoing = False




#

# ----
# iter 200, loss: 77.081298
