---
title: Planning Approaches
weight: 91
draft: true
---

# Planning Approaches

In this section we treat planning approaches that dominate in real-world applications. We focus on motion planning of autonomous mobile agents (self driving cars) as an example subdomain that we are all familiar with. There are [two main schools of thought in motion planning](https://arxiv.org/pdf/2003.06404.pdf) irrespectively of the Markovian assumptions regarding the transition or measurement models. 

|  Approach  |  Description   |     
| --- | --- | 
|  Modular Pipeline  |  We have seen the modular architecture in [the PGM chapter]({{<ref "../../pgm/pgm-intro">}}) where sensors, perception, planning and control subsystems work together to achieve the task at hand. |
|  End-to-End Pipeline |  In end-to-end approach the entire pipeline of transforming sensory inputs to driving commands is treated as a single learning task. We have seen this example in [the introductory chapter]({{<ref "../../ai-intro/course-introduction">}}) where Imitation Learning (IL) was used to determine from only pixel input the steering and acceleration controls of a vehicle. Apart from IL, Reinforcement Learning (RL) can also be used in a simulator setting to transfer the learned optimal policy in the real-world. |

![planning-approaches](images/planning-approaches.png#center)
*Modular vs End-to-End pipelines for real-world motion planning*

## Modular Pipeline

The modular pipeline as shown below, consists of four main modules:

NOTE: In the description below the motion planning module mentioned in the figure is called trajectory generation.

![modular-pipeline](images/modular-pipeline.png#center)
*Modular pipeline functional flow -from [here](https://arxiv.org/pdf/1604.07446.pdf).*

1. **Global (Route) planning**. In global route planning we are given a graph $G=(V, E)$ and the agent must solve the problem of finding a shortest path (cost) route from the source vertex $s$ to a destination or goal $g$.  The route planning scenarios can be fairly complicated involving a mixture of  road network restricted and unrestricted (eg. parking lot) modes. Route planning is usually decoupled from the other latency-sensitive subsystems of planning - it is being very infrequently executed.   
2. **Prediction**. This involves estimations of what other agents active in the same locale as the agent of interest will do next. 
3. **Behavioral decision making**. Makes a decision of the best action that the agent should make given the predictions of the other agents perceived intentions, signal and traffic sign detections. 
4. **Trajectory generation**. The action decided by the behavioral planning module, is translated to a trajectory the control subsystem will attempt to follow as close as possible. 

## End to end pipeline

