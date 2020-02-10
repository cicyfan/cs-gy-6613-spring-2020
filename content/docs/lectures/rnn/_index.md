---
title: Week 4b - Sequences and Recurrent Neural Networks (RNN)
draft: true
weight: 70
---

# Sequences and Recurrent Neural Networks (RNN)

## Sequences

Data streams are everywhere in various applications. For example, weather station sensor data arrive in streams indexed by time, as is financial trading data - one can think of many others. Time series are a special case of sequenced data where the index is time. Here we will be dealing with sequenced data - an example of non time index sequences are sentences and language in general. We are interested to fit sequenced data with a model and in order to do so we need a hypothesis set that we can draw from that is rich enough for the task at hand. 

Irrespectively of the nature of the index though, linear dynamical systems are very rich models.  In the following we use $t$ as the index variable and the notation $x_{1:t}$ means the sequence from 1 to $t$. 

For an input $\mathbf x_t$, a dynamical system and its _recurrent_ state evolution can be represented as

$$\mathbf{s}\_t = f(\mathbf{s}\_{t-1}, \mathbf{x}_t ; \theta)$$



## RNN Architecture


CNNs are for images what RNNs are for sequences usually time-series data and this includes video (sequence of images).


One of the strongest advantages of CNNs, is their ability to share parametes across the image, by reusing the kernel weights over and over again. 

Hello
