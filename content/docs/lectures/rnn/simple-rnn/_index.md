---
title: Simple RNNs and their Backpropagation
draft: false
weight: 132
---

# Simple RNN 
![rnn-hidden-recurrence](images/rnn-hidden-recurrence.png#center)

*Simple RNN with recurrences between hidden units. This architecture can compute any computable function and therefore is a [Universal Turing Machine](http://alvyray.com/CreativeCommons/BizCardUniversalTuringMachine_v2.3.pdf).* 

Notice how the path from input $\bm x_{t-1}$ affects the label $\bm y_{t}$ and also the conditional independence between $\bm y$ given $\bm x$. Please note that this is not a computational graph rather one way to represent the hidden state transfer between recurrences.

## Forward Propagation 

This network maps the input sequence to a sequence of the same length and implements the following forward pass:

$$\bm a_t = \bm W \bm h _{t-1} + \bm U \bm x_t + \bm b$$

$$\bm h_t = \tanh(\bm a_t)$$

$$\bm o_t = \bm V \bm h_t + \bm c$$

$$\hat \bm y_t = \mathtt{softmax}(\bm o_t)$$

$$L(\bm x_1, \dots , \bm x_{\tau}, \bm y_1, \dots , \bm y_{\tau}) = D_{KL}[\hat p_{data}(\bm y | \bm x) || p_{model}(\bm y | \bm x; \bm w)]$$

$$= - \mathbb E_{\bm y | \bm x \sim \hat{p}_{data}} \log p_{model}(\bm y | \bm x ; \bm w)  = - \sum_t \log p_{model}(y_t | \bm x_1, \dots, \bm x_t ; \bm w)$$ 

Notice that RNNs can model very generic distributions  $\log p_{model}(\bm x, \bm y ; \bm w)$. The simple RNN architecture above, effectively models the posterior distribution $\log p_{model}(\bm y | \bm x ; \bm w)$  and based on a conditional independence assumption it factorizes into $\sum_t \log p_{model}(y_t | \bm x_1, \dots, \bm x_t ; \bm w)$. 

Note that by connecting the $\bm y_{t-1}$ to $\bm h_t$ via a matrix e.g. $\bm R$ we can avoid this simplifying assumption and be able to model an arbitrary distribution $\log p_{model}(\bm y | \bm x ; \bm w)$. In other words just like in the other DNN architectures, connectivity directly affects the representational capacity of the [hypothesis set]({{<ref "../../learning-problem">}}). 

In many instances we have problems where it only matters the label $y_\tau$ at the end of the sequence. Lets say that you are classifying speech or video inside the cabin of a car to detect the psychological state of the driver. The same architecture shown above can also represent such problems - the only difference is the only the $\bm o_\tau$, $L_\tau$ and $y_\tau$ will be considered. 

Lets see an example to understand better the forward propagation equations.

![example-sentence](images/example-sentence.png#center)
*Example sentence as input to the RNN*

In the figure above you have a hypothetical document (a sentence) that is broken into what in natural language processing called _tokens_. Lets say that a token is a word in this case. In the simpler case where we need a classification of the whole document, given that $\tau=6$, we are going to receive at t=1, the first token $\bm x_1$ and with an input hidden state  $\bm h_0 = 0$ we will calculate the forward equations for $\bm h_1$, ignoring the output $\bm o_1$ and repeat the unrolling when the next input $\bm x_2$ comes in until we reach the end of sentence token $\bm x_6$ which in this case will calculate the output $y_6$ and loss 

$$- \log p_{model} (y_6|\bm x_1, \dots , \bm x_6; \bm  w)$$ 

where $\bm w = \\{ \bm W, \bm U, \bm V, \bm b, \bm c \\}$. 

Where it gets interesting though is when we connect RNNs and the [language models]({{<ref "../../nlp">}}). 

![rnn-language-model](images/rnn-language-model.png#center)
*RNN language model example - training [ref](https://www.youtube.com/watch?v=6niqTuYFZLQ&t=521s)*

In the figure above we see a toy example where the vocabulary is ['h','e','l','o']. where the tokens are single letters represented in the input with a one-hot encoded vector. Let us assume that the network is being trained with the sequence "hello". The letters will come in one at a time, each letter going through the forward pass that produces at the output the $\mathbf y_t$ that indicates which letter is expected to arrive next.  You can see, since we are just started training,  that this network is not predicting correctly - this will improve over time as the model is trained with more sequence permutations form our limited vocabulary. During inference we will use the language model to 

![rnn-language-model-inference](images/rnn-language-model-inference.png#center)
*RNN language model example - generate the next token [ref](https://www.youtube.com/watch?v=6niqTuYFZLQ&t=521s)*

## Back-Propagation Through Time (BPTT)

Lets now see how the backward propagation would work. 

![rnn-BPTT](images/rnn-BPTT.png#center)
*Understanding RNN memory through BPTT procedure*

Backpropagation is similar to that of feed-forward (FF) networks simply because the unrolled architecture resembles a FF one. But there is an important difference and we explain this using the above computational graph for the unrolled recurrences $t$ and $t-1$. During computation of the variable $\bm h_t$ we use the value of the variable $\bm h_{t-1}$ calculated in the previous recurrence. So when we apply the chain rule in the backward phase of BP, for all nodes that involve the such variables with recurrent dependencies, the end result is that _non local_ gradients from previous backpropagation steps ($t$ in the figure) appear. This is effectively why we say that simple RNNs feature _memory_. This is in contrast to the FF network case where during BP only local to each gate gradients where involved as we have seen in the the [DNN chapter]({{<ref "../../dnn/backprop-dnn">}}). 

The key point to notice in the backpropagation in recurrence $t-1$ is the junction between $\tanh$ and $\bm V \bm h_{t-1}$. This junction brings in the gradient $\nabla_{\bm h_{t-1}}L_t$ from the backpropagation of the $\bm W h_{t-1}$ node in recurrence $t$ and just because its a junction, it is added to the backpropagated gradient from above in the current recurrence $t-1$ i.e.

$$\nabla_{\bm h_{t-1}}L_{t-1} \leftarrow \nabla_{\bm h_{t-1}}L_{t-1} + \nabla_{\bm h_{t-1}}L_t $$ 

Ian Goodfellow's book section 10.2.2 provides the exact equations - please note that you need to know only the intuition behind computational graphs for RNNs. In practice BPTT is truncated to avoid having to do one full forward pass and one full reverse pass through the training dataset of an e.g. language model that is usually very large, to do a single gradient update. 

{{<expand "Minimal RNN language model code from Stanford CS231n" >}}

```python
# see here for notation http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture10.pdf
"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  ```
  {{</expand>}}

## Vanishing or exploding gradients

In the figure below we have drafted a conceptual version of what is happening with recurrences over time. Its called an infinite impulse response filter for reasons that will be apparent shortly. 

![rnn-IIR](images/rnn-IIR.png#center)
*Infinite Impulse Response (IIR) filter with weight $w$*

With $D$ denoting a unit delay, the recurrence formula for this system is:

$$h_t = w h_{t-1} + x_t$$

where $w$is a weight (a scalar). Lets consider what happens when an impulse, $x_t = \delta_t$ is fed at the input of this system with $w=-0.9$. 

$$h_0 = -0.9 h_{-1} + \delta_0 = 1$$
$$h_1 = -0.9 h_{0} + \delta_1 = -0.9$$
$$h_2 = -0.9 h_{1} + \delta_2 = 0.81$$
$$h_3 = -0.9 h_{2} + \delta_3 = -0.729$$

With $w=-0.9$, the h_t (called impulse response) follows a decaying exponential envelope while obviously with $w > 1.0$ it would follow an exponentially increasing envelope. Such recurrences if continue will result in vanishing or exploding responses long after the impulse showed up in the input $t=0$. 

In a similar fashion, the RNN hidden state recurrence, in the backwards pass of backpropagation that extends from the $t=\tau$ to $t=1$ can make the gradient, when $\tau$ is large, either _vanish_ or _explode_. Instead of a scalar $w$ we have matrices $\bm W$ involved instead of $h$ we have gradients $\nabla_{\bm h_{t}}L_{t}$. This is discussed in [this](http://proceedings.mlr.press/v28/pascanu13.pdf) paper.

Using this primitive IIR filter as an example, we can see that the weight plays a crucial role in the impulse response. This is further discussed in the [LSTM]({{<ref "../lstm">}}) section. 
 