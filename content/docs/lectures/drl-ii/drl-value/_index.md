---
title: Value Iteration
weight: 106
draft: false
---

# Value Iteration

This chapter, is divided into two parts. In the first part, similar to the [policy-based DRL]({{<ref "../../drl-i/drl-policy">}}) that we presume the reader has gone through, we continue to investigate approaches for the _planning_ problem with a _known MDP_. 

In the second part, we find optimal policy solutions when the MDP is _unknown_ and we need to _learn_ its underlying functions - also known as the  _model free_ prediction problem.  

## Dynamic Programming and Value Iteration

The basic principle behind value-iteration is the principle that underlines dynamic programming and is called the _principle of optimality_ as applied to policies. According to this principle an _optimal_ policy can be divided into two components.

1. An optimal first action $a_*$.
2. An optimal policy from the successor state $s^\prime$.

More formally, a policy $\pi(a|s)$ achieves the optimal value from state $s$, $v_\pi(s) = v_*(s)$ iff for any state $s^\prime$ reachable from $s$, $v_\pi(s^\prime)=v_*(s)$. 

Effectively this principle allows us to decompose the problem into two sub-problems with one of them being straightforward to determine and use the Bellman **optimality equation** that provides the one step backup induction at each iteration.  

$$v_*(s) = \max_a \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_*(s^\prime) \right)$$

As an example if I want to move optimally towards a location in the room, I can make a optimal first step and at that point I can follow the optimal policy, that I was magically given, towards the desired final location. That optimal first step, think about making it by walking backwards from the goal. We start at the end of the problem where we know the final rewards and work backwards to all the states that correct to it in our look-ahead tree. 

![value-iteration-look-ahead-tree](images/value-iteration-look-ahead-tree.png#center)
*One step look-ahead tree representation of value iteration algorithm*

The "start from the end" intuition behind the equation is usually applied with no consideration as to if we are at the end or not. We just do the backup inductive step for each state.  In value iteration for synchronous backups, we start at $k=0$ from the value function $v_0(s)=0.0$ and at each iteration $k+1$ for all states $s \in \mathcal{S}$ we update the $v_{k+1}(s)$ from $v_k(s)$. As the iterations progress, the value function will converge to $v_*$.

The equation of value iteration is taken straight out of the Bellman optimality equation. 

$$v_{k+1}(s) = \max_a \left( \mathcal R_s^a + \gamma \sum_{s^\prime \in \mathcal S} \mathcal{P}^a_{ss^\prime} v_k(s^\prime) \right) $$

which can be written in a vector form as,

$$\mathbf v_{k+1} = \max_a \left( \mathcal R^a + \gamma \mathcal P^a \mathbf v_k \right) $$

Notice that we are not building an explicit policy at every iteration and also perhaps importantly, the intermediate value functions may _not_ correspond to a feasible policy. Before going into a more elaborate example, we can go back to the same simple world we have looked at in the [policy iteration]({{../../drl-i/drl-value}}) section and focus only on the state-value calculation using the formula above. 

![gridworld-value-iteration](images/gridworld-value-iteration-value-only.png#center)
*State values for an MDP with random policy (0.25 prob of taking any of the four available actions), $\gamma=1$, that rewards the agent with -1 at each transition except towards the goal states that are in the top left and bottom right corners*

We return to the tree representation of the value iteration with DP - this will be useful when we compare the DP with other value iteration approaches. 

$$V(S_t) = \mathbb E_\pi \left[R_{t+1} + \gamma V(S_{t+1}) \right]$$

![dp-value-iteration-tree](images/dp-value-iteration-tree.png#center)
*Backup tree with value iteration based on the DP approach - Notice that we do one step look ahead but we do not sample as we do in the other value iteration approaches.* 

### DP Value-iteration example

In example world shown below (from [here](http://i-systems.github.io/HSE545/iAI/AI/topics/05_MDP/11_MDP.html))

![gridworld](images/gridworld.png#center)
*Gridworld to showcase the state-value calculation in Python code below. The states are numbered sequentially from top right.*

we can calculate the state-value function its the vector form - the function in this world maps the state space to the 11th dim real vector space  $v(s): \mathcal S \rightarrow \mathbb R^{11}$ aka the value function is a vector of size 11.

$$\mathbf v_{k+1} = \max_a \left( \mathcal R^a + \gamma \mathcal P^a \mathbf v_k \right) $$

{{<expand "Grid world value iteration" >}}

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

{{</expand>}}

## Monte-Carlo (MC) Approximations

The state-value function was defined in the [MDP chapter]({{<ref "../../drl-i/mdp">}}) as the _expected_ return.

$$v_\pi(s) = \mathop{\mathbb{E}_\pi}(G_t | S_t=s)$$

In the discussion of the REINFORCE algorithm, we came across the approximation of the return, called _sample mean_ over a _sample_ episode / trajectory, 

$$G_t(\tau) = \sum_{k=0}^{T-1}\gamma^k R_{t+1+k}$$

We can therefore approximate the value function in what is in essence called _Monte-Carlo approximation_, by the sample mean of the returns over multiple over episodes / trajectories. In other words, to update each element of the state value function 

1. For each time step $t$ that state $s$ is visited in an episode
   * Increment a counter $N(s)$ of visitations  
   * Calculate the total return $S(s) = S(s) + G_t$
2. At the end of multiple episodes, the value is estimated as $V(s) = S(s) / N(s)$

As $N(s) \rightarrow ∞$ the estimate will converge to $V(s) \rightarrow v_\pi(s)$.

Notice that we started using capital letters for the _estimates_ of the value functions.  

Going back to the familiar tree structure its interesting to see what MC does to the value estimate, given its equation:

{{<hint danger>}}

$$V(S_t) = V(S_t) + \alpha(G_t - V(S_t))$$

{{</hint>}}

![mc-value-iteration-tree](images/mc-value-iteration-tree.png#center)
*Backup tree with value iteration based on the MC approach. MC samples a complete trajectory to the goal node T shown with red.*

## Temporal Difference (TD) Approximations

Instead of waiting for the value function to be estimated at the end of multiple episodes, we can use the incremental mean approximation as shown below to update the value function after each episode. 

$$ \mu_k = \frac{1}{k} \sum_{j=1}^k x_j = \frac{1}{k} \left( x_k + \sum_{j=1}^{k-1} x_j \right)$$ 
$$ = \frac{1}{k} \left(x_k + (k-1) \mu_{k-1}) \right) =  \mu_{k-1} + \frac{1}{k} ( x_k - \mu_{k-1} )$$

Using the incremental sample mean we can approximate the value function after each episode if for each state $S_t$ with return $G_t$,

$$ N(S_t) = N(S_t) +1 $$
$$ V(S_t) = V(S_t) + \alpha \left( G_t - V(S_t) \right)$$

where $\alpha = \frac{1}{N(S_t)}$ can be interpreted as the forgetting factor - it can also be any number $< 1$ to convert the sample mean into a running mean. 

Going back to the example of crossing the room optimally, we sample a trajectory, take a number steps and use an approximate value functions for the remaining trajectory. We repeat this as we go along effectively _bootstrapping_ the value function approximation with whatever we have experienced up to now. Mathematically, instead of using the _true_ return, TD uses a (biased) _estimated_ return called the _TD target_: $ R_{t+1} + \gamma V(S_{t+1})$ approximating the value function as:

{{<hint danger>}}

$$ V(S_t) = V(S_t) + \alpha \left( R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right)$$

{{</hint>}}

The difference below is called the _TD error_,

$$\delta_t = R_{t+1} + \gamma (V(S_{t+1}) - V(S_t))$$

We can now use an example that has no explicit actions but focuses on how TD differs from MC with respect to the rewards, as depicted in the figure below:

![td-driving-to-work-example](images/td-driving-to-work-example.png#center)
*Two value approximation methods: MC (left), TD (right) as converging in their predictions of the value of each of the states in the x-axis. The example is from a hypothetical commute from office back home. In MC you have to wait until the episode ended (reach the goal) to update the value function at each state of the trajectory. In contrast, TD updates the value function at each state based on the estimates of the total travel time. The goal state is "arrive home", while the reward function is time.*

As you can notice in the figure above the solid arrows in the MC case, adjust the predicted value of each state to the _actual_ return while in the TD case the value prediction happens every step in the way. We call TD for this reason an _online_ learning scheme. Another characteristic of TD is that it does not depend on reaching the goal, it _continuously_ learns. MC does depend on the goal and therefore is _episodic_. 

Finally, here is our tree for the TD behavior,  

![td-value-iteration-tree](images/td-value-iteration-tree.png#center)
*Backup tree for value iteration with the TD approach. TD samples a single step ahead as shown with red.* 

### The TD($\lambda$)

The TD approach of the previous section, can be extended to multiple steps. Instead of a single look ahead step we can take multiple successive look ahead steps (n), we will call this TD(n) for now, and at the end of the n-th step, we use the value function at that state to backup and get the value function at the state where we started. Effectively after n-steps our return will be:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{n-1}R_{t+n} + \gamma_n V(S_n)$$

and the TD(n) learning equation becomes

$$ V(S_t) = V(S_t) + \alpha \left( G^{(n)}_t - V(S_t) \right) $$

We now define the so called $\lambda$-return that combines all n-step return $G_t^{(n)}$ via the weighting function shown below as,

$$G_t^{(n)} = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

![lambda-weighting-function](images/lambda-weighting-function.png#center)
*$\lambda$ weighting function for TD($\lambda$)*

the TD(n) learning equation becomes

{{<hint danger>}}

$$ V(S_t) = V(S_t) + \alpha \left( G^\lambda_t - V(S_t) \right) $$

{{</hint>}}

When $\lambda=0$ we get TD(0) learning, while when $\lambda=1$ we get learning that is roughly equivalent to MC. 
