---
title: Value-based Deep RL 
weight: 106
draft: false
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

As an example if I want to move optimally towards a location in the room, I can make a optimal first step and at that point I can follow the optimal policy that I was magically given towards the desired final location. Effectively this principle allows us to decompose the problem into two sub-problems with one of them bring straightforward to determine and use the Bellman **optimality equation** that provides the one step backup induction at each iteration.  

$$v_*(s) = \max_a \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_*(s^\prime)$$

So we start at the end of the problem where we know the final rewards and work backwards to all the states that correct to it in our look ahead tree. Note that algorithm can function though without consideration as to what state results in a successor that is the goal. 

![value-iteration-look-ahead-tree](images/value-iteration-look-ahead-tree.png#center)
*One step look ahead tree representation of value iteration algorithm*

In value iteration for synchronous backups, at each iteration $k+1$ for all states $s \in \mathcal{S}$ we update the $v_{k+1}(s)$ from $v_k(s)$. As the iterations progress, the value function will converge to $v_*$. 

The equation of value iteration is taken straight out of the Bellman optimality equation. 

$$v_{k+1}(s) = \max_a \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_k(s^\prime) \right) $$

which can be written in matrix form as,

$$\mathbf v_{k+1} = \max_a \left( \mathcal R^a + \gamma \mathcal P^a \mathbf v_k \right) $$

Notice that we are not building an explicit policy at every iteration and also perhaps importantly, the intermediate value functions may not correspond to a feasible policy. 

As a trivial example, that shortest path problems we have seen in the [planning chapter]({{<ref "../../planning">}}), can be solved with dynamic programming via the value iteration. This is shown next for a simple grid world. 

![value-iteration-simple-grid-world](images/../../drl-policy/images/value-iteration-simple-grid-world.png#center)
*Simple grid world where each action results in a reward of -1 and we are asked to define the shortest path towards the goal state $g$. Notice that in the synchronous backup case in each iteration we update all states.*


## The SARSA RL Algorithm 
TBC
