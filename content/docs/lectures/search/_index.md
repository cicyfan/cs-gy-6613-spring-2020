---
title: Lecture 7a - Problem Solving via Search
weight: 91
draft: true

---

# Problem Solving via Search

In [recursive state estimation]({{<ref "../pgm/recursive-state-estimation">}}) chapter we made two advances in our modeling tool set:

1. We introduced sequenced events (time) and the concept of a varying _state_ over such sequences.  
2. We saw how the agent state as dictated by an underlying dynamical model and and how to estimate it recursively using a graphical model that introduced a Bayesian probabilistic framework. We saw that many well known estimation algorithms such as the Kalman filter are specific cases of this framework. 

With this probabilistic reasoning in place, we can now track objects in the scene and ultimately assign symbols that represent them as a direct consequence of their unique attributes (e.g. location). Having symbolic representation of its agent's locale environment is not enough though as we need a compatible _global_ representation of the environment and additional semantics to denote _goals_. With such complementary representations we can hope that we can efficiently infer states that we _cannot perceive_ as well as plan ahead to satisfy our goals. We effectively zoom out from the task-specific _factored_ representation of the agent's state and we look at environment state that is _atomic_ i.e. it is not broken down into its individual variables. 

Atomic state representations of an environment are adequate for a a variety of tasks:  one striking use case is robotic navigation. There, the scene or environment takes the form of a global map and the goal is to move the embodied agent from a starting state to a goal state. If we assume that the global map takes the form of a grid with a suitable resolution, each grid tile represents a different atomic state than any other tile. Similar considerations can be made for other forms of the map e.g. a graph form. 

Given such state representation, search is one of the methods we use to find the action sequence that the agent must produce to reach a goal state. Note that in most cases, we are dealing with _informed_ rather than _blind_ search, where we are also given task-specific knowledge to find the solution as we will see shortly. 

## The A* Algorithm for Path Finding

One such example task is to find the path between a starting state and the goal state in a map. Not just any path that the _minimum cost_ path when given a map (discrete), starting state, goal state, cost function. The solution is as shown below. 

![path-finding](images/path-finding.png#center)
*Find the path in a map from a starting state (*) to a goal state (x)*

Lets look at an example cost function. 



The first task is to find the right data structures to represent it especially if we handcraft the algorithm.

[This](http://theory.stanford.edu/~amitp/GameProgramming/) is one of the better explanation of the A* algorithm 

## Demo

The [demo below](https://qiao.github.io/PathFinding.js/visual/) is instructive of the various search algorithms we will cover in class.

<iframe src="https://qiao.github.io/PathFinding.js/visual/" width="900" height="1200"></iframe>

