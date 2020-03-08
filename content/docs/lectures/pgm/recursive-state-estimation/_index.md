---
title: Recursive State Estimation
draft: true
weight: 85
---

# Recursive State Estimation

In [Inference in Graphical Models section]({{<ref "../bayesian-inference">}}) we have seen how sequential data belonging to just two evidential variables (captured via $p(x,y)$) can be treated by probabilistic models to infer (reason) about values of the posterior. Now we will expand on two fronts:

* Introduce the concept of _state_ $s$ that encapsulates multiple random variables and consider _dynamical systems_ with non-trivial non-linear dynamics (state transition models) common in robotics, medical diagnosis and many other fields.

* Introduce the time index $t$ explicitly in the aforementioned state evolution as represented via a graphical model. 

The perception subsystem, that processes sensor data produces noisy estimates (object detections etc.) - these can be attributed to the algorithmic elements of the subsystem or to imperfections in the sensors themselves. The model that captures the perception pipeline from sensors to estimates will be called _measurement model_ or _sensor model_. So in summary we have two abstractions / models that we need to be concerned about: the transition model of the environment state and the sensor model. 

Such expansion, will allow us to form using the Bayes rule, perhaps one of the most important contributions to the probabilistic modeling of dynamical systems: the recursive state estimator also known as _Bayes filter_ that affords the agent the ability to maintain an internal _belief_ of the current state of the environment.  

## Bayes Filter

We are introducing this algorithm, by considering a embodied agent (a robot) that moves in an environment characterized by a state $s$. 