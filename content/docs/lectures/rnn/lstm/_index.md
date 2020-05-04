---
title: The Long Short-Term Memory (LSTM) Cell Architecture
draft: false
weight: 135
---

# The Long Short-Term Memory (LSTM) Cell Architecture

In [the simple RNN]({{<ref "../simple-rnn/">}}) we have seen the problem of exploding or vanishing gradients when the span of back-propagation is large (large $\tau$). Using the conceptual IIR filter, that ultimately _integrates_ the input signal, we have seen that in order to avoid an exploding or vanishing impulse response, we need to control $w$. This is exactly what is being done in an evolutionary RNN architectures that we will treat in this section called _gated RNNs_.

The way we control $w$ is to have another system produce it for the task at hand. In the best known gated RNN architecture, the IIR recurrence and everything that controls it, is contained in the LSTM _cell_. The LSTM cell adjusts $w$ depending on the input sequence _context_ and this means that (a) there is an internal memory to the cell, we call this the _cell state_ and (b) $w$ will fluctuate depending on $x$. We employ another hidden unit to learn the context and, based on that, set the right $w$. This unit is called the _forget gate_: since by making $w$ equal to zero, $h_t$ stops propagating aka it forgets the previous hidden state. We employ a couple of other gates as well: the _input gate_ and the _output gate_ as shown in the diagram below. 

![lstm-cell](images/rnn-LSTM.png#center)
*LSTM Cell: The cell is divided into three areas: input (green), cell state (blue) and output (red). The $i$ index (see description below) has been supressed for clarity*

The cell is divided into three areas: input (green), cell state (blue) and output (red)

In each area there is a corresponding gate (filled node) - these are the input gate, forget gate, output gate for the input, cell state and output regions respectively. The gates controls the flow of information that goes through these areas via element-wise multipliers. The two inputs to the cells are the concatenation of $\bm x_t$ and $\bm h_{t-1}$ and the cell state from the previous recurrence $\bm s_{t-1}$. 

## The Cell State

Starting at the heart of the LSTM cell, to describe the update we will use two indices: one for the unfolding sequence index $t$ and the other for the cell index $i$. We use the additional index to allow the current cell at step $t$ to use or forget inputs and hidden states from other cells. 

$$s_t(i) = f_t(i) s_{t-1}(i) + g_t(i) \sigma \Big( \bm W^T(i) \bm h_{t-1}(i) + \bm U^T(i) \bm x_t(i) + \bm b(i) \Big)$$

The parameters $\theta_{in} = \\{  \bm W, \bm U, \bm b \\}$  are the recurrent weights, input weights and bias at the input of the LSTM cell. 

The forget gate calculates the forgetting factor,

$$f_t(i) =\sigma \Big( \bm W_f^T(i) \bm h_{t-1}(i) + \bm U_f^T(i) \bm x_t(i) + \bm b_f(i) \Big) $$

## Input

The input gate _protects the cell state_ contents from perturbations by irrelevant to the context inputs. Quantitatively,  input gate calculates the factor,

$$g_t(i) =\sigma \Big( \bm W_g^T(i) \bm h_{t-1}(i) + \bm U_g^T(i) \bm x_t(i) + \bm b_g(i) \Big) $$

The gate with its sigmoid function adjusts the value of each element produced by the input neural network.

## Output

The output gate _protects the subsequent cells_ from perturbations by irrelevant to their context cell state. Quantitatively,

$$h_t(i) = q_t(i) \tanh(s_t(i))$$ 

where $q_t(i)$ is the output factor

$$q_t(i) =\sigma \Big( \bm W_o^T(i) \bm h_{t-1}(i) + \bm U_o^T(i) \bm x_t(i) + \bm b_o(i) \Big) $$

Notice that if you omit the gates we are back to the simple RNN architecture. You can expect backpropagation to work similarly in LSTM albeit with more complicated expressions. Some more diagraming and annimation dont hurt to understand LSTMs. See [1](https://colah.github.io/posts/2015-08-Understanding-LSTMs) and [2](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21). 