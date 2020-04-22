---
title: Value-based Deep RL 
weight: 106
draft: true
---

# Value-based Deep RL


This chapter, is divided into two parts. In the first part, similar to the [policy-based DRL]({{<ref "../drl-policy">}}) that we presume the reader has gone through, we continue to investigate approaches for the _planning_ problem with a _known MDP_. 

In the second part, we find optimal policy solutions when the MDP is _unknown_ and we need to _learn_ its underlying functions / models - also known as the  _model free_ prediction problem.  

we develop the so called _model-free prediction_ approach to RL. In the first part we will derive optimal value solutions to control problems with _known MDP_. In the second part, we find optimal policy solutions when the MDP is _unknown_ and we need to _learn_ its underlying functions. 

## Dynamic Programming and Value Iteration

The basic principle behind value-iteration is the principle that underlines dynamic programming and is called the _principle of optimality_ as applied to policies. According to this principle an _optimal_ policy can be divided into two components.

1. An optimal first action $A_*$.
2. An optimal policy from the successor states $S^\prime$.

More formally, 

A policy $\pi(a|s)$ achieves the optimal value from state $s$, $v_\pi(s) = v_*(s)$ iff for any state $s^\prime$ reachable from $s$, $v_\pi(s^\prime)=v_*(s)$. 

As an example if I want to move optimally towards a location in the room, I can make a optimal first step and at that point I can follow the optimal policy towards the desired final location.  Effectively this is the decomposition

We can do that by applying the Bellman expectation equation backup once and apply to the resultant value function to the 

## The SARSA RL Algorithm 
