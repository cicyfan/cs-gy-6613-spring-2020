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

$$v_*(s) = \max_a \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_*(s^\prime) \right)$$

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

### Value Iteration Calculation Example

In another more elaborate example shown below (from [here](http://i-systems.github.io/HSE545/iAI/AI/topics/05_MDP/11_MDP.html))

![gridworld](images/gridworld.png#center)
*Gridworld to showcase the state-value calculation in Python code below. The states are numbered sequentially from top right.*

we can calculate the state-value function its the vector form - the function in this world maps the state space to the 11th dim real vector space  $v(s): \mathcal S \rightarrow \mathbb R^{11}$ aka the value function is a vector of size 11.

$$\mathbf v_{k+1} = \max_a \left( \mathcal R^a + \gamma \mathcal P^a \mathbf v_k \right) $$

{{<expand "Grid world value function estimation" >}}

```python

# Each of the 11 rows of the "matrix" P[s][a] has 4 tuples - one for each of the allowed actions. Each tuple / action is written in the format (probability, s') and is associated with the 3 possible next states that the agent may end up despite its intention to go to the desired state. 

P = {
 0: {0: [(0.9,0),(0.1,1),(0,4)], 1: [(0.8,1),(0.1,4),(0.1,0)], 2: [(0.8,4),(0.1,1),(0.1,0)], 3: [(0.9,0),(0.1,4)]},
 1: {0: [(0.8,1),(0.1,2),(0.1,0)], 1: [(0.8,2),(0.2,1)], 2: [(0.8,1),(0.1,0),(0.1,2)], 3: [(0.8,0),(0.2,1)]},
 2: {0: [(0.8,2),(0.1,3),(0.1,1)], 1: [(0.8,3),(0.1,5),(0.1,2)], 2: [(0.8,5),(0.1,1),(0.1,3)], 3: [(0.8,1),(0.1,2),(0.1,5)]},
 3: {0: [(0.9,3),(0.1,2)], 1: [(0.9,3),(0.1,6)], 2: [(0.8,6),(0.1,2),(0.1,3)], 3: [(0.8,2),(0.1,3),(0.1,6)]},
 4: {0: [(0.8,0),(0.2,4)], 1: [(0.8,4),(0.1,7),(0.1,0)], 2: [(0.8,7),(0.2,4)], 3: [(0.8,4),(0.1,0),(0.1,7)]},
 5: {0: [(0.8,2),(0.1,6),(0.1,5)], 1: [(0.8,6),(0.1,9),(0.1,2)], 2: [(0.8,9),(0.1,5),(0.1,6)], 3: [(0.8,5),(0.1,2),(0.1,9)]},
 6: {0: [(0.8,3),(0.1,6),(0.1,5)], 1: [(0.8,6),(0.1,10),(0.1,3)], 2: [(0.8,10),(0.1,5),(0.1,6)], 3: [(0.8,5),(0.1,3),(0.1,10)]},
 7: {0: [(0.8,4),(0.1,8),(0.1,7)], 1: [(0.8,8),(0.1,7),(0.1,4)], 2: [(0.9,7),(0.1,8)], 3: [(0.9,7),(0.1,4)]},
 8: {0: [(0.8,8),(0.1,9),(0.1,7)], 1: [(0.8,9),(0.2,8)], 2: [(0.8,8),(0.1,7),(0.1,9)], 3: [(0.8,7),(0.2,8)]},
 9: {0: [(0.8,5),(0.1,10),(0.1,8)], 1: [(0.8,9),(0.1,9),(0.1,5)], 2: [(0.8,9),(0.1,8),(0.1,10)], 3: [(0.8,8),(0.1,5),(0.1,9)]},
 10: {0: [(0.8,6),(0.1,10),(0.1,9)], 1: [(0.9,10),(0.1,6)], 2: [(0.9,10),(0.1,9)], 3: [(0.8,9),(0.1,6),(0.1,10)]}
}

R = [0, 0, 0, 1, 0, 0, -100, 0, 0, 0, 0]
gamma = 0.9

States = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Actions = [0, 1, 2, 3] # [north, east, south, west]

v = [0]*11

# value iteration

for i in range(100):
    for s in States:
        q_0 = sum(trans[0]*v[trans[1]] for trans in P[s][0])
        q_1 = sum(trans[0]*v[trans[1]] for trans in P[s][1])
        q_2 = sum(trans[0]*v[trans[1]] for trans in P[s][2])
        q_3 = sum(trans[0]*v[trans[1]] for trans in P[s][3])

        v[s] = R[s] + gamma*max(q_0, q_1, q_2, q_3)
    
print(v)

# [5.46991289990088, 6.313016781079707, 7.189835364530538, 8.668832766371658, 4.8028486314273, 3.346646443535637, -96.67286272722137, 4.161433444369266, 3.6539401768050603, 3.2220160316109103, 1.526193402980731]

# once v computed, we can calculate the optimal policy 
optPolicy = [0]*11

for s in States:       
    optPolicy[s] = np.argmax([sum([trans[0]*v[trans[1]] for trans in P[s][a]]) for a in Actions])

print(optPolicy)    
# [1, 1, 1, 0, 0, 3, 3, 0, 3, 3, 2]
```

## The SARSA RL Algorithm 


