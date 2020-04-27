---
title: Lecture 9 - Deep Reinforcement Learning I
weight: 110
draft: false
---

# Deep Reinforcement Learning I 

![drl-concept](images/drl-concept.png#center)
*DRL Principle - we will explain this figure when we treat policy-based RL*

In this chapter we build upon what we learned in the [MDP]({{<ref "../mdp">}}) chapter and develop agents that can act in an initially unknown environment and learn via their interactions with it, gradually improving their policy.

These agents are different from the [planning agents]({{<ref "../planning">}}) where the _model_ of the environment is known and with _no interaction_ with it the agent improves its policy, using this model as well as problem solving and logical reasoning skills.

Apart from the notes here that are largely based on [David Silver's (Deep Mind) course material](https://www.davidsilver.uk/teaching/) and [video lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), the curious mind will be find additional resources [in the suggested book for this part of the course](https://www.amazon.com/Deep-Reinforcement-Learning-Python-Hands-dp-0135172381/dp/0135172381/ref=mt_paperback?_encoding=UTF8&me=&qid=) written by Google researchers as well as on [OpenAI's website](https://openai.com/resources/).

The algorithmic exploration in RL is a very long journey as shown below - we will cover only 2-3 algorithms that can be used as design patterns for the others. 

![drl-algorithm-evolution](images/drl-algorithm-evolution.png#center)
*DRL algorithms - taxonomy and evolution*