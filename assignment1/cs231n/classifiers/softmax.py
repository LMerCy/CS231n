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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #print('X:{}\n'.format(X.shape))
  scores = X.dot(W)
  #print('scores :{}\n'.format(scores.shape))
  num_train,dim = X.shape
  num_class = W.shape[1]
  score_max = np.reshape(np.max(scores,axis = 1),(num_train,1))#avoid numerical instability
  #print('score :{}\nscore_max:{}'.format(scores.shape,score_max.shape))
  scores = scores - score_max
  y_correct = np.zeros((num_train,num_class))
  y_correct[np.arange(num_train),y] = 1
  for i in range(num_train):
      prob = np.exp(scores[i,:])/np.sum(np.exp(scores[i,:]))
      loss +=(-np.log(prob)).dot(y_correct[i,:])
      dW += (prob - y_correct[i,:])*(X[i,:].reshape(dim,1))#see :https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  loss /= num_train
  dW /= num_train
  dW += reg*W
  loss += 0.5*reg*np.sum(np.square(W))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

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

  scores = X.dot(W)
  num_train,dim = X.shape
  score_max = np.reshape(np.max(scores,axis = 1),(num_train,1))
  scores = scores - score_max
  prob = np.exp(scores)/np.sum(np.exp(scores),axis =1,keepdims = True)
  crossEn = -np.log(prob)  
  y_correct = np.zeros_like(prob)
  y_correct[np.arange(num_train),y] = 1.0
  loss = np.sum(crossEn*y_correct)/num_train+0.5*reg*np.sum(np.square(W))
  dW += np.dot(X.T,prob - y_correct)/num_train + reg * W#attention prob is the output of softmax
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

