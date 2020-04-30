---
title: Lecture 9/10 - Deep Reinforcement Learning
weight: 110
draft: false
---

# Deep Reinforcement Learning

We started looking at different agent behavior architectures starting from the [planning agents]({{<ref "../planning">}}) where the _model_ of the environment is known and with _no interaction_ with it the agent improves its policy, using this model as well as problem solving and logical reasoning skills. We then looked at agents that can plan by interacting with the environment still knowing the model - this was covered in the [MDP]({{<ref "../mdp">}}) chapter. In this chapter we develop agents that can act in an _initially unknown_ environment and learn via their interactions with it, gradually improving their policy. 

Its been a long road to reach this point and we have just a short very short conclusion on the relationship of RL and the previously derived MDP solution. 

In the reinforcement learning problem setting, agents _do not know_ essential elements of the MDP $\mathcal M = <\mathcal S, \mathcal P, \mathcal R, \mathcal A, \gamma>$ that were assumed as given in the previous section. This includes the transition function, $P^a_{ss^\prime}$ and the reward function $\mathcal R_s^a$ that are essential as we have seen above to estimate the value function and optimize the policy.

The only way an agent can get information about these missing functions is through its experiences (states, actions, and rewards) in the environment—that is, the tuples ($S_t, A_t, R_{t+1}$). Provided that it can learn such functions, RL can be posed as an MDP policy optimization problem and many algorithms that we have looked at already, such as dynamic programming, policy and value iteration are different ways to solve what is unified approach to Reinforcement Learning shown below. 

![unified-view-rl](images/unified-view-rl.png#center)
*Different Approaches to solve RL as an MDP*

The MDP functions we dont know can be approximated using DNNs as we will see later, but for now the high level architecture is that of what is shown below. 

![drl-concept](images/drl-concept.png#center)
*DRL Principle - we will explain this figure later.*

Suffice to say that exploring DRL algorithms is a very long journey as shown below - we will cover only three key algorithms: REINFORCE, SARSA and DQN that can be used as design patterns for the others. 

![drl-algorithm-evolution](images/drl-algorithm-evolution.png#center)
*DRL algorithms - taxonomy and evolution*

> Apart from the notes here that are largely based on [David Silver's (Deep Mind) course material](https://www.davidsilver.uk/teaching/) and [video lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), the curious mind will be find additional resources: 
> * [in the suggested book](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Hands-dp-0135172381/dp/0135172381/ref=mt_paperback?_encoding=UTF8&me=&qid=) written by Google researchers as well as on [OpenAI's website](https://openai.com/resources/). The chapters we covered is 1-4. 
> * You may also want to watch Andrew Ng's, [2018 version of his ML class](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) that includes MDP and RL lectures.
