---
title: Syllabus
weight: 10
---

# Syllabus

The course schedule below highlights our journey to understand the multiple subsystems and how they can be connected together to create compelling but, currently, domain specific forms of intelligence. 

## Books

[Artificial Intelligence: A Modern Approach, by Stuart Russell, 3rd edition, 2010](https://www.amazon.com/Artificial-Intelligence-Approach-Stuart-Russell/dp/9332543518/ref=sr_1_2?crid=17NGBV1XXV150&keywords=ai+a+modern+approach&qid=1576432665&sprefix=ai+the+modern+appr%2Caps%2C158&sr=8-2) and also [here.](http://aima.cs.berkeley.edu/)

The publisher is about to release the [4th edition (2020) of this classic](https://www.amazon.com/Artificial-Intelligence-A-Modern-Approach/dp/0134610997/ref=sr_1_3?crid=17NGBV1XXV150&keywords=ai+a+modern+approach&qid=1576432686&sprefix=ai+the+modern+appr%2Caps%2C158&sr=8-3). We will be monitoring availability in bookstores but it does not seem likely this edition to appear on time for the Spring 2020 class.  

Other recommended texts are: 

(a) DRL: "Foundations of Deep Reinforcement Learning", by Graesser & Keng, 2020. 

(b) GERON: "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow", 2nd Edition, by Geron, 2019. 

(c) DL: https://www.deeplearningbook.org/ (free)

## Schedule

The schedule is based on [Academic Calendar Spring 2020](https://www.nyu.edu/registrar/calendars/university-academic-calendar.html): 

### Part I:  Perception and Machine Learning

1. **Lecture 1 (1/27/2020)** We start with an introduction to AI and present a systems approach towards it. We develop a map that will guide us through the rest of the course as we deep dive into each component embedded into _AI agents_. Reading: AIMA Chapters 1 & 2.  

2. **Lecture 2 (2/3/2020)**  The perception subsystem is the first stage of many AI systems including our brain. Its function is to process and fuse multimodal sensory inputs. Perception is implemented via a number of reflexive agents that map directly perceived state to an primitive action such as regressing on the frame coordinates of an object in the scene. We present _the supervised learning problem_ both for classification and regression, starting with classical ML algorithms. Reading: AIMA Chapter 18. 

3. **Lecture 3 (2/10/2020)**  We expand into _Deep neural networks_. DNNs are developed bottom up from the Perceptron algorithm. MLPs learn via optimization approaches such as Stochastic Gradient Descent.  We deep-dive into back-propagation - a fundamental algorithm that efficiently trains DNNs. Reading: DL Chapter 6

**3/16/2020**  Enjoy President's Day holiday.

4. **Lecture 4: (2/24/2020)** We dive into the most dominant DNN architecture today -  _Convolutional Neural Networks (CNNs)_. Reading: DL Chapter 9 & 10. 

5. **Lecture 5: (3/2/2020)** When agents move in the environment they need to abilities such as _scene understanding_.  We will go through few key perception building blocks such as Object Detection, Semantic and Instance Segmentation. Some of these building blocks (autoencoders) are instructive examples of representations learning that will be shown to be an essential tool in the construction of environment state representations. Reading: Various papers 
        
### Part II: Reasoning and Planning

6. **Lecture 6: (3/9/2020)**  In this lecture we introduce probabilistic models that process the outputs of perception (measurement / sensor model) and the state transitions and understand how the agent will track / update its belief state over time. This is a achieved with probabilistic recursive state estimation algorithms and dynamic bayesian networks. Reading: AIMA Chapters 14 & 15. 

**3/16/2020**  Enjoy your Spring Break.

7. **Lecture 7: (4/13/2020)** After the last lecture, the agent has a clear view of the environment state such as what and where the objects that surround it are, its able to track them as they potentially move. It needs to plan the best sequence of actions to reach its goal state and the approach we take here is that of _problem solving_. In fact planning and problem solving are inherently connected as concepts. If the goal state is feasible then the problem to solve  becomes that of  _search_. For instructive purposes we start from simple environmental conditions that are fully observed, known and deterministic. This is where the A* algorithm comes in. We then relax some of the assumptions and treat environments that are deterministic but the agent takes stochastic actions or when both the environment and agent actions are stochastic. We also investigate what happens when we do not just care about reaching our goal state, but when we, in addition, need to do so with optimality. Optimal planning under uncertainty is perhaps the cornerstone application today in robotics and other fields. Readings: Reading: AIMA Chapters 3 & 4 (problem solving)  AIMA Chapters 10 & 11 (planning) and selected papers.

**3/30/2020 - This is your Midterm Test (2h)** 

### Part III: Reinforcement Learning

8. **Lecture 8: (4/6/2020)** We now make a considerable extension to our assumptions: the utility of the agent now depends on a sequence of decisions and, further, the stochastic environment offers a feedback signal to the agent called _reward_. We review how the agent's policy, the sequence of actions, can be calculated when it fully observes its current state (MDP) and also when it can only partially do so (POMDP). We conclude with the basic taxonomy of the algorithmic space for RL problems.   Readings: AIMA Chapter 16 & 17, DRL Chapter 1. This lecture will be delivered by my colleague Gurudutt Hossangadi. 

**4/13/2020** Good luck with your midterm.

9.  **Lecture 9: (4/20/2020)**  The algorithms that learn optimal policies in such settings are known as Reinforcement Learning (RL). In this lecture we establish the connection between MDP and RL, by introducing the Bellman expectation backup and Bellman optimality equations. We then use these equations to derive the policy iteration algorithm that is behind the policy-based REINFORCE algorithm that is empowered by approximating the policy function using the Deep Neural Networks that we met in the perception subsystem. DRL Chapter 2.  
                
10.   **Lecture 10: (4/27/2020)**  In this lecture we aim to expand our DRL treatment on value-based methods and the principle of action-value iteration. We then look at algorithm incarnations of value-based methods such as State Action Reward State Action (SARSA) and if time allows off-policy DQN and understand how they learn the value functions. Readings: DRL Chapter 3 / 4. 

### Part IV: Natural Language Processing and Representations
                
11. **Lecture 11: (5/05/2020)**  NLP is the pinnacle of applied AI in every day life - we are all using natural language as the prime means of communicate between us and increasingly between us and robots. In this lecture we pose the NLP problem, understand its components and their mechanics. We then talk extensively about _language modeling_ and start with an approach based on the [RNN / LTSM]({{<ref "../lectures/rnn">}}) architecture. The later is used far beyond language modeling and expands into every use case that involves sequences. AIMA Chapter 23 and DL Chapter 10. 
        
12. **Lecture 12: (5/11/2020)**  We introduce the concept of _attention_ and go through the Transformer framework - perhaps the most successful architecture in NLP today. We also cover some key aspects of the course to prepare for the final test. 
        
**5/18/2020**  Good luck with your final test.
          