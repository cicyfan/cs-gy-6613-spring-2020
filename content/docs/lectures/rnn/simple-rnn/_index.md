---
title: Simple RNNs and their Backpropagation
draft: false
weight: 71
---

# Simple RNN 
![rnn-hidden-recurrence](images/rnn-hidden-recurrence.png#center)

*Simple RNN with recurrences between hidden units. This architecture can compute any computable function and therefore is a [Universal Turing Machine](http://alvyray.com/CreativeCommons/BizCardUniversalTuringMachine_v2.3.pdf). Your laptops and smartphones are descendants of UTM.* 

Notice how the path from input $\bm x_{t-1}$ affects the label $\bm y_{t}$ and also the conditional independence between $\bm y$ given $\bm x$. Please note that this is not a computational graph rather one way to represent the hidden state transfer between recurrences.

## Forward Propagation 

This network maps the input sequence to a sequence of the same length and implements the following forward pass:

$$\bm a_t = \bm W \bm h _{t-1} + \bm U \bm x_t + \bm b$$

$$\bm h_t = \tanh(\bm a_t)$$

$$\bm o_t = \bm V \bm h_t + \bm c$$

$$\hat \bm y_t = \mathtt{softmax}(\bm o_t)$$

$$L(\bm x_1, \dots , \bm x_{\tau}, \bm y_1, \dots , \bm y_{\tau}) = D_{KL}[\hat p_{data}(\bm y | \bm x) || p_{model}(\bm y | \bm x; \bm w)]$$

$$= - E_{\bm y | \bm x ≋ \hat{p}_{data}} \log p_{model}(\bm y | \bm x ; \bm w)  = - \sum_t \log p_{model}(y_t | \bm x_1, \dots, \bm x_t ; \bm w)$$ 

Notice that RNNs can model very generic distributions  $\log p_{model}(\bm x, \bm y ; \bm w)$. The simple RNN architecture above, effectively models the posterior distribution $\log p_{model}(\bm y | \bm x ; \bm w)$  and based on a conditional independence assumption it factorizes into $\sum_t \log p_{model}(y_t | \bm x_1, \dots, \bm x_t ; \bm w)$. 

Note that by connecting the $\bm y_{t-1}$ to $\bm h_t$ via a matrix e.g. $\bm R$ we can avoid this simplifying assumption and be able to model an arbitrary distribution $\log p_{model}(\bm y | \bm x ; \bm w)$. In other words just like in the other DNN architectures, connectivity directly affects the representational capacity of the hypothesis set. 

In many instances we have problems where it only matters the label $y_\tau$ at the end of the sequence. Lets say that you are classifying speech or video inside the cabin of a car to detect the psychological state of the driver. The same architecture shown above can also represent such problems - the only difference is the only the $\bm o_\tau$, $L_\tau$ and $y_\tau$ will be considered. 

Lets see an example to understand better the forward propagation equations.

![example-sentence](images/example-sentence.png#center)
*Example sentence as input to the RNN*

In the figure above you have a hypothetical document (a sentence) that is broken into what in natural language processing called _tokens_. Lets say that a token is a word in this case. In the simpler case where we need a classification of the whole document, given that $\tau=6$, we are going to receive at t=1, the first token $\bm x_1$ and with an input hidden state  $\bm h_0 = 0$ we will calculate the forward equations for $\bm h_1$, ignoring the output $\bm o_1$ and repeat the unrolling when the next input $\bm x_2$ comes in until we reach the end of sentence token $\bm x_6$ which in this case will calculate the output and loss 

$$- \log p_{model} (y_6|\bm x_1, \dots , \bm x_6; \bm  w)$$ 

where $\bm w = \\{ \bm W, \bm U, \bm V, \bm b, \bm c \\}$. 


## Back-Propagation Through Time (BPTT)

Lets now see how the backward propagation would work. 

![rnn-BPTT](images/rnn-BPTT.png#center)
*Understanding RNN memory through BPTT procedure*

Backpropagation is similar to that of feed-forward (FF) networks simply because the unrolled architecture resembles a FF one. But there is an important difference and we explain this using the above computational graph for the unrolled recurrences $t$ and $t-1$. During computation of the variable $\bm h_t$ we use the value of the variable $\bm h_{t-1}$ calculated in the previous recurrence. So when we apply the chain rule in the backward phase of BP, for all nodes that involve the such variables with recurrent dependencies, the end result is that _non local_ gradients from previous backpropagation steps ($t$ in the figure) appear. This is effectively why we say that simple RNNs feature _memory_. This is in contrast to the FF network case where during BP only local to each gate gradients where involved as we have seen in the the [DNN chapter]({{<ref "../../dnn/backprop-dnn">}}). 

The key point to notice in the backpropagation in recurrence $t-1$ is the junction between $\tanh$ and $\bm V \bm h_{t-1}$. This junction brings in the gradient $\nabla_{\bm h_{t-1}}L_t$ from the backpropagation of the $\bm W h_{t-1}$ node in recurrence $t$ and just because its a junction, it is added to the backpropagated gradient from above in the current recurrence $t-1$ i.e.

$$\nabla_{\bm h_{t-1}}L_{t-1} += \nabla_{\bm h_{t-1}}L_t $$ 

Ian Goodfellow's book section 10.2.2 provides the exact equations - please note that you need to know the intuition behind computational graphs for RNNs. 


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

In a similar fashion, the RNN hidden state recurrence, in the backwards pass of backpropagation that extends from the $t=\tau$ to $t=1$ can make the gradient, when $\tau$ is large, either _vanish_ or _explode_. Instead of a scalar $\w$ we have matrices $\bm W$ involved instead of $h$ we have gradients $\nabla_{\bm h_{t}}L_{t}$. This is discussed in [this](http://proceedings.mlr.press/v28/pascanu13.pdf) paper.

Using this primitive IIR filter as an example, we can see that the weight plays a crucial role in the impulse response. This is further discussed in the [LSTM]({{<ref "../lstm">}}) section. 
 