---
title: Policy-based Deep RL
weight: 105
draft: true
---

# Policy-based Deep RL 

In this chapter is divided into two parts. In the first part, we develop the so called _planning_ problem (which is RL without learning) where we are dealing with a _known MDP_. This means that we know the transition and reward functions/models of the environment and we are after the optimal policy solutions. 

In the second part, we find optimal policy solutions when the MDP is _unknown_ and we need to _learn_ its underlying functions / models - also known as the  _model free_ control problem. Learning in this chapter follows the _on-policy_ approach where the agent learns these models "on the job", so to speak. 

## Dynamic Programming and Policy Iteration

In the [MDP]({{<ref "../mdp">}}) chapter we have derived the Bellman expectation _backup_ equations which allowed us to efficiently compute the value function. 

We have seen also that the Bellman optimality equations are non linear and need to be solved using iterative approaches - their solution will result in the optimal value function $v_*$ and the optimal policy $\pi_*$. Since the Bellman equations allow us to decompose recursively the problem into sub-problems, they in fact implement a general and exact approach called _dynamic programming_ which assumes full knowledge of the MDP.

In the policy iteration, given the policy $\pi$, we iterate two distinct steps as shown below:

![policy-iteration-steps](images/policy-iteration-steps.png#center)
*Policy iteration in solving the MDP - in each iteration we execute two steps, policy evaluation and policy improvement*

1. In the _evaluation_ (also called the _prediction_) step we estimate the state value function $v_\pi ~ \forall s \in \mathcal S$.

2. In the _improvement_ (also called the _control_) step we apply the greedy heuristic and elect a new policy based on the evaluation of the previous step. 

This is shown next,

![policy-iteration-convergence](images/policy-iteration-convergence.png#center)
*Policy and state value convergence to optimality in policy iteration. Up arrows are the evaluation steps while down arrows are the improvement steps.*

It can be shown that the policy iteration will converge to the optimal value function $v_*(s)$ and policy $\pi_*$. 

#### Policy Evaluation Step

The policy $\pi$ is evaluated when we have produced the state-value function $v_\pi(s)$ for all states. In other words when we know the expected discounted returns that each state can offer us. To do so we apply the Bellman expectation backup equations repeatedly. 

We start at $k=0$ by initializing all state-value function (a vactor) to $v_0(s)=0$. In each iteration $k+1$ we start with the state value function of the previous iteration $v_k(s)$ and apply the Bellman expectation backup as prescribed by the one step lookahead tree below that is decorated relative to what [we have seen]({{<ref "../mdp">}}) with the iteration information. This is called the synchronous backup formulation as we are updating all the elements of the value function vector at the same time. 

![policy-evaluation-tree](images/policy-evaluation-tree.png#center)
*Tree representation of the state-value function with one step look ahead across iterations.*

The Bellman expectation backup is given by,

$$v_{k+1}(s) = \sum_{a \in \mathcal A} \pi(a|s) \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_k(s^\prime) \right)$$

and in vector form,

$$\mathbf{v}^{k+1} = \mathbf{\mathcal R}^\pi + \gamma \mathbf{\mathcal P}^\pi \mathbf{v}^k$$

The following source code is instructive and standalone. It executes the policy evaluation for the Gridworld environment from the many that are part of the Gym RL python library. 
 
{{< expand "Policy Evaluation Python Code" "..." >}}

```python
# this code is from https://towardsdatascience.com/reinforcement-learning-demystified-solving-mdps-with-dynamic-programming-b52c8093c919

import numpy as np 
import gym.spaces
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, epsilon=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V_old = np.zeros(env.nS)

    while True:
        
        #new value function
        V_new = np.zeros(env.nS)
        #stopping condition
        delta = 0

        #loop over state space
        for s in range(env.nS):

            #To accumelate bellman expectation eqn
            v_fn = 0
            #get probability distribution over actions
            action_probs = policy[s]

            #loop over possible actions
            for a in range(env.nA):

                #get transitions
                [(prob, next_state, reward, done)] = env.P[s][a]
                #apply bellman expectatoin eqn
                v_fn += action_probs[a] * (reward + discount_factor * V_old[next_state])

            #get the biggest difference over state space
            delta = max(delta, abs(v_fn - V_old[s]))

            #update state-value
            V_new[s] = v_fn

        #the new value function
        V_old = V_new

        #if true value function
        if(delta < epsilon):
            break

    return np.array(V_old)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print(v)
print(expected_v)
```
{{</expand>}}

#### Policy Improvement Step

In the policy iteration step we are given the value function and simply apply the greedy heuristic to it.

$$\pi^\prime = \mathtt{greedy}(v_\pi)$$

It can be shown that this heuristic results into a policy that is better than the one the prediction step started ($\pi^\prime \ge \pi$) and this extends into multiple iterations. We can therefore converge into an optimal policy - the interested reader can follow [this](https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3) lecture for a justification. 

{{< expand "Policy Iteration Python Code" "..." >}}

```python
import numpy as np
import gym.spaces
from gridworld import GridworldEnv

env = GridworldEnv()

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
       
    """
    def one_step_lookahead(s, value_fn):

        actions = np.zeros(env.nA)

        for a in range(env.nA):

            [(prob, next_state, reward, done)] = env.P[s][a]
            actions[a] = prob * (reward + discount_factor * value_fn[next_state])
            
        return actions

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    actions_values = np.zeros(env.nA)

    while True:

        #evaluate the current policy
        value_fn = policy_eval_fn(policy, env)
       
        policy_stable = True

        #loop over state space
        for s in range(env.nS):


            #perform one step lookahead
            actions_values = one_step_lookahead(s, value_fn)
            
        	#maximize over possible actions 
            best_action = np.argmax(actions_values)

            #best action on current policy
            chosen_action = np.argmax(policy[s])

    		#if Bellman optimality equation not satisifed
            if(best_action != chosen_action):
                policy_stable = False

            #the new policy after acting greedily w.r.t value function
            policy[s] = np.eye(env.nA)[best_action]

        #if Bellman optimality eqn is satisfied
        if(policy_stable):
            return policy, value_fn

    
    

policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
```
{{</expand>}}

We already have seen that in the Gridworld environment, we may not _need_ to reach the optimal state value function $v_*(s)$ for an optimal policy to result, as shown in the next figure where the value function for the $k=3$ iteration results the same policy as the policy from a far more accurate value function (large k). 

![gridworld-policy-iterations](images/gridworld-policy-iterations.png#center)
*Convergence to optimal policy via separate prediction and policy improvement iterations*

We can therefore stop early and taking the argument to the limit, do the policy improvement step in _each_ iteration. This is equivalent to the value iteration that we will treat in a different chapter. 

In summary, we have seen that policy iteration solves the known MDPs. In the next section we remove the known MDP assumption and deal with the first Reinforcement Learning (RL) algorithm. 

## The REINFORCE RL Algorithm

Given that RL can be posed as an MDP, in this section we continue with a policy-based algorithm that learns the policy _directly_ by optimizing the objective function and can then map the states to actions.  The algorithm we treat here is an algorithm of fundamental importance - not that other more modern algorithms do not perform better but because its elements are instructive for understanding others. 

The algorithm is called REINFORCE and took its name from the fact that during _training_ actions that resulted in good outcomes should become more probable—these actions are positively _reinforced_. Conversely, actions which resulted in bad outcomes should become less probable. If learning is successful, over the course of many iterations, action probabilities produced by the policy, shift to a distribution that results in good performance in an environment. Action probabilities are changed by following the policy gradient, therefore REINFORCE is known as a policy gradient algorithm.
 
 The algorithm needs three components:

| Component | Description |
|---|---|
| Parametrized policy $\pi_\theta (a\|s)$ | The key idea of the algorithm is to learn a good policy, and this means doing function approximation. Neural networks are powerful and flexible function approximators, so we can represent a policy using a deep neural network (DNN) consisting of learnable parameters $\mathbf \theta$. This is often referred to as a policy network $\pi_θ$. We say that the policy is parametrized by  $\theta$. **Each specific set of values of the parameters of the policy network represents a particular policy**. To see why, consider $θ1 ≠ θ2$. For any given state $s$, different policy networks may output different sets of action probabilities, that is, $π_{θ1}(a|s) ≠ π_{θ2}(a|s)$. The mappings from states to action probabilities are different so we say that $π_{θ1}$ and $π_{θ2}$ are different policies. A single DNN is therefore capable of representing many different policies.|
| The objective to be maximized $J(\pi_\theta)$[^1]| At this point is nothing else other than the expected discounted _return_ over policy, just like in MDP. |
| Policy Gradient | A method for updating the policy parameters $\theta$. The policy gradient algorithm searches for a local maximum in $J(\pi_\theta)$:  $\max_\theta J(\pi_\theta)$. This is the common gradient ascent algorithm that we met in a similar form in neural network. $$\theta ← \theta + \alpha \nabla_\theta J(\pi_\theta)$$ where $\alpha$ is the learning rate.| 

Out of the three components the most complicated one is the policy gradient that can be shown to be given by the differentiable quantity: 

$$ \nabla_\theta J(\pi_\theta)= \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta (a|s) v_\pi (s) \right ]$$

We can approximate the value at state $s$ with the return over many sample trajectories $\tau$ that are sampled from the policy network. 

$$ \nabla_\theta J(\pi_\theta)= \mathbb{E}_{\tau \sim \pi_\theta} \left[ G_t \nabla_\theta \log \pi_\theta (a|s) \right ]$$

where $G_t$ is the _return_ - a quantity we have seen earlier albeit now the return is limited by the length of each trajectory,

$$G_t(\tau) = \sum_{k=0}^T\gamma^k R_{t+1+k}$$

The $\gamma$ is usually a hyper-parameter that we need to optimize usually iterating over many values in [0.01,...,0.99] and selecting the one with the best results. 

We also have an expectation in the gradient expression that we need to address.  The expectation $\mathbb{E}_{\tau \sim \pi_\theta}$ we need to with summation over _each_ trajectory. This is a straightforward approximation and is very commonly called Monte-Carlo. Effectively we are generating the right hand side (sampling) and sum as done in line 8 in the code below that summarizes the algorithm:

1: Initialize learning rate $\alpha$

2: Initialize weights θ of a policy network $π_θ$

3: for episode = 0, . . . , MAX_EPISODE do

4: &ensp;&ensp;Sample a trajectory using the policy network $\pi_\theta$, $τ = s_0, a_0, r_0, . . . , s_T, a_T, r_T$

5: &ensp;&ensp;Set $∇_θ J(π_θ) = 0$

6: &ensp;&ensp; for t = 0, . . . , T-1 do

7: &ensp;&ensp; &ensp;&ensp;         Calculate $G_t(τ)$

8: &ensp;&ensp; &ensp;&ensp;         $\nabla_\theta J(\pi_\theta) = \nabla_\theta J(\pi_\theta) + G_t (τ) \nabla_\theta \log \pi_\theta (a_t|s_t) $

9: &ensp;&ensp;     end for

10:&ensp;&ensp;      $θ = θ + α ∇_θ J(π_θ)$

11: end for

It is important that a trajectory is discarded after each parameter update—it cannot be reused. This is because REINFORCE is an _on-policy_ algorithm. Intuitively an on-policy algorithm "learns on the job" as evidently can be seen in line 10 where the parameter update equation plugs policy gradient that itself (line 8) directly depends on action probabilities $π_θ(a_t | s_t)$ generated by the _current_ policy $π_θ$ only and not some past policy $π_{θ′}$. Correspondingly, the return $G_t(τ)$ where $τ ~ π_θ$ must also be generated from $π_θ$, otherwise the action probabilities will be adjusted based on returns that the policy wouldn’t have generated.

### Policy Network
One of the key ingredients that DRL introduces is the policy network that is approximated with a DNN eg. a fully connected neural network with a number of hidden layers that is hyper-parameter (e.g. 2 RELU). 

1: Given a policy network ``net``, a ``Categorical`` (multinomial) distribution class, and a ``state``

2: Compute the output ``pdparams = net(state)``

3: Construct an instance of an action probability distribution  ``pd = Categorical(logits=pdparams)``

4: Use pd to sample an action, ``action = pd.sample()``

5: Use pd and action to compute the action log probability, ``log_prob = pd.log_prob(action)``

Other discrete distributions can be used and many actual libraries parametrize continuous distributions such as Gaussians. 

It is now instructive to see an stand-alone example in python for the so called ``CartPole-v0`` [^2]

![cartpole](images/cartpole.png#center)

{{<expand "REINFORCE Python Code" "..." >}}
```python
 1  from torch.distributions import Categorical
 2  import gym
 3  import numpy as np
 4  import torch
 5  import torch.nn as nn
 6  import torch.optim as optim
 7
 8  gamma = 0.99
 9
10  class Pi(nn.Module):
11      def __init__(self, in_dim, out_dim):
12          super(Pi, self).__init__()
13          layers = [
14              nn.Linear(in_dim, 64),
15              nn.ReLU(),
16              nn.Linear(64, out_dim),
17          ]
18          self.model = nn.Sequential(*layers)
19          self.onpolicy_reset()
20          self.train() # set training mode
21
22      def onpolicy_reset(self):
23          self.log_probs = []
24          self.rewards = []
25
26      def forward(self, x):
27          pdparam = self.model(x)
28          return pdparam
29
30      def act(self, state):
31          x = torch.from_numpy(state.astype(np.float32)) # to tensor
32          pdparam = self.forward(x) # forward pass
33          pd = Categorical(logits=pdparam) # probability distribution
34          action = pd.sample() # pi(a|s) in action via pd
35          log_prob = pd.log_prob(action) # log_prob of pi(a|s)
36          self.log_probs.append(log_prob) # store for training
37          return action.item()
38
39  def train(pi, optimizer):
40      # Inner gradient-ascent loop of REINFORCE algorithm
41      T = len(pi.rewards)
42      rets = np.empty(T, dtype=np.float32) # the returns
43      future_ret = 0.0
44      # compute the returns efficiently
45      for t in reversed(range(T)):
46          future_ret = pi.rewards[t] + gamma * future_ret
47          rets[t] = future_ret
48      rets = torch.tensor(rets)
49      log_probs = torch.stack(pi.log_probs)
50      loss = - log_probs * rets # gradient term; Negative for maximizing
51      loss = torch.sum(loss)
52      optimizer.zero_grad()
53      loss.backward() # backpropagate, compute gradients
54      optimizer.step() # gradient-ascent, update the weights
55      return loss
56
57  def main():
58      env = gym.make('CartPole-v0')
59      in_dim = env.observation_space.shape[0] # 4
60      out_dim = env.action_space.n # 2
61      pi = Pi(in_dim, out_dim) # policy pi_theta for REINFORCE
62      optimizer = optim.Adam(pi.parameters(), lr=0.01)
63      for epi in range(300):
64          state = env.reset()
65          for t in range(200): # cartpole max timestep is 200
66              action = pi.act(state)
67              state, reward, done, _ = env.step(action)
68              pi.rewards.append(reward)
69              env.render()
70              if done:
71                  break
72          loss = train(pi, optimizer) # train per episode
73          total_reward = sum(pi.rewards)
74          solved = total_reward > 195.0
75          pi.onpolicy_reset() # onpolicy: clear memory after training
76          print(f'Episode {epi}, loss: {loss}, \
77          total_reward: {total_reward}, solved: {solved}')
78
79  if __name__ == '__main__':
80      main()
```
{{</expand>}}

The REINFORCE algorithm presented here can generally be applied to continuous and discreet problems but it has been shown to possess high variance and sample-inefficiency. Several improvements have been proposed and the interested reader can refer to section 2.5.1 of the suggested book. 

[^1]: Notation wise, since we need to have a bit more flexibility in RL problems, we will use the symbol $J(\pi_\theta)$ as the objective function.
[^2]: Please note that SLM-Lab, the library that accompanies the suggested in the syllabus book, is a mature library and probably a good example of how to develop ML/RL libraries in python. You will learn a lot by reviewing the implementations under the ``agents/algorithms`` directory to get a feel of how RL problems are abstracted .  

