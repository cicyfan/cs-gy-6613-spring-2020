---
title: Lecture 8 - Markov Decision Processes 
weight: 101
---

# Markov Decision Processes

We started looking at different agent behavior architectures starting from the [planning agents]({{<ref "../planning">}}) where the _model_ of the environment is known and with _no interaction_ with it the agent improves its policy, using this model as well as problem solving and logical reasoning skills. 

We now look at agents that can plan _by interacting_ with the environment still knowing the model (model-based) of the environment such as its dynamics and rewards. The planning problem as will see, it will be described via a set of four equations called Bellman expectation and Bellman optimality equations that connect the values (utility) with each state or action with the policy. These equations can be solved by Dynamic Programming algorithms to produce the optimal policy (strategy) that the agent must adopt. 

Computationally we will go through approaches that solve the MDP as efficiently as possible - namely, the value and policy iteration algorithms.

![solving-mpd](images/solving-mdp.png#center)
*Solving MDP Problems*

>  Many of the algorithms presented here like policy and value iteration have been developed in [this](https://github.com/rlcode/reinforcement-learning) git repo that you should download and run while reading the notes. 