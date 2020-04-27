---
title: SARSA
weight: 107
draft: false
---

# The SARSA DRL Algorithm 

The SARSA algorithm is a value-based algorithm and far more applicable in practice than policy-based since it tends to be more sample efficient - a general trait of many value-based algorithms despite the usual hybridization that is usually applicable today. Its name is attributed to the fact that we need to know the State-Action-Reward-State-Action before performing an update. There are two concepts that we need to grasp:

1. The first is a technique for learning the Q-function known via the [_temporal difference (TD) learning_]({{<ref "../drl-value">}}) we saw earlier. 

2. The second is a method for generating actions using the learned Q-function. 

The tree for SARSA is shown below:

![sarsa-update-tree](images/sarsa-update-tree.png#center)
*SARSA action-value backup update tree. The name SARSA is written as you rad from the top to the bottom of the tree :)*

Using this tree and following the value iteration section, we can write the value update equation as:

$$Q(S,A) = Q(S,A) + \alpha (R + \gamma Q(S^\prime, A^\prime)-Q(S,A))$$

Effectively the equation above updates the Q function by $\alpha$ times the direction of the TD-error.

Going back to the policy iteration instead of starting from a policy and a state value function we can start from a policy and action value function and follow the same principle of iterating over action value estimation and greedy policy improvement.

![generalized-policy-iteration](images/generalized-policy-iteration.png#center)
*Generalized policy iteration using action-value function*

The policy improvement step over $Q(s,a)$ is _model-free_

$$\pi^\prime = \argmax_{a \in \mathcal A} Q(s,a)$$

What SARSA does is basically the diagram above but with a twist. Instead of trying to evaluate the policy all the way using the DP, or over an episode using MC,  SARSA does policy improvement **over each time step** significantly increasing the iteration rate - this is figuratively shown below:

![sarsa-policy-iteration](images/sarsa-policy-iteration.png#center)
*SARSA on-policy control*

The idea is to increase the frequency of the so called $\epsilon$-Greedy policy improvement step where we select with probability $\epsilon$ a random action instead of the action that maximizes the $Q(s,a)$ function (greedy). 

The SARSA algorithm is summarized below:

![sarsa-on-policy-control-algorithm](images/sarsa-on-policy-control-algorithm.png#center)
*SARSA algorithm for on-policy control*

## SARSA Example
