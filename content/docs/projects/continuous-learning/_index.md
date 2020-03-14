---
title: Project 2 - Continual Learning for Robotic Perception
---

# Project 2 - Continual Learning for Robotic Perception

**This project is due March 29 at 11:59pm**

One of the greatest goals of AI is building an artificial continual learning agent which can construct a sophisticated understanding of the external world from its own experience through the adaptive, goal-oriented and incremental development of ever more complex skills and knowledge. Continual learning is essential in robotics where high dimensional data streams need to be constantly processed and where naïve continual learning strategies have been shown to suffer from _catastrophic forgetting_. 

|CORe50 Option   | Rotated MNIST Option    |
| --- | --- |
|  You will use [this](https://vlomonaco.github.io/core50/index.html) dataset and evaluate your method for New Class (NC) scenario.   |  Use the dataset provided [here](https://github.com/facebookresearch/GradientEpisodicMemory)  based on [this paper](http://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) |
|   Object recognition (classification).  | Object recognition (classification). |

The CORe50 option is more difficult than the MNIST option. Grading will happen relative to teams that selected the same option. The CORe50 option may require an AWS/Azure/Google cloud or NYU GPU compute  resource to run. MNIST should be able to run in Colab/Kaggle with standard free accounts. If you have access to compute, you will learn more from selecting the CORe50 option as this is closest to a real life dataset than the rotated MNIST which is not even testing for new classes - rather it tests CL on rotated version of classes it has seen before.

This is a very active area in AI right now - see [here](https://sites.google.com/view/clvision2020/challenge?authuser=0)
 