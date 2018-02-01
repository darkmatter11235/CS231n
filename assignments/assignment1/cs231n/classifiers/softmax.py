import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W) #(N,C)
  for i in xrange(N):
      scores[i] -= np.max(scores[i])
      denom = 0.0
      num = np.exp(scores[i,y[i]])
      for j in xrange(C):
        denom += np.exp(scores[i,j])
      softmax_loss = num/denom
      c_norm = -np.log(softmax_loss)
      #print 'num %f c_norm %f' %(num,c_norm,)
      loss += c_norm
      for j in xrange(C):
          dW[:,j] += (np.exp(scores[i,j])/denom- (j == y[i])) * X[i].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= N
  loss += 0.5* reg * np.sum(W * W)
  dW /= N

  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  scores = X.dot(W) #(N, C)
  scores -= np.max(scores)
  p = np.exp(scores)
  p_y = p[np.arange(N),y].reshape(-1,1)
  #print p_y.shape
  p_sum = np.sum(p, axis=1).reshape(-1,1)
  soft_loss = np.log(p_y / p_sum)*-1
  #print soft_loss[1:2]
  loss = np.sum(soft_loss)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW_mask = p/p_sum
  dW_mask[np.arange(N), y] -= 1
  dW = X.T.dot(dW_mask)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

