---
title: Lecture 12 - Natural Language Processing
draft: true
weight: 130
---

# Natural Language Processing

![national-library-greece](images/national-library-greece.jpg#center)
*“You shall know a word by the company it keeps” (J. R. Firth 1957: 11) - many modern discoveries are in fact rediscoveries from other works sometimes decades old. NLP is non an exception.*

In this chapter we will start the journey to discover how agents can process _and respond_ to input sources that contain natural language. Such inputs are all the the trillions of web pages, billions of captioned videos, real-time multi-modal speech and video etc.  We wil rewind developments in this space starting in 2012 to discover how J.R. Firth's words translate to the NLP space as it evolved over the last 8 years.

We start though with a overall architecture and 
## Language Models

In programming languages we have 
despite the fact that words in a sentence may appear very similar
Naturally expressed languages on the other hand can carry quite different meanings.  

{{< columns >}} 
## Same context, different meaning
"The food in this restaurant was good, not bad at all"

"The food in this restaurant was bad, not good at all."
<---> 

## Different context all together 
"The valuation of this bank eroded soon after the 2008 crisis"

"The bank of this river eroded after the 2008 floods"
{{< /columns >}}

Bag of words will fail to capture the different meaning especially in sentences like the restaurant reviews above that have the same distribution of words. One of key ideas that made NLP successful is the _distributional semantics_ that originated from Firth's work: A word’s meaning is given by the words that frequently appear close-by. When a word $w$ appears in a text, its context is the set of words that appear nearby (within a fixed-size window). Use the many contexts of $w$ to build up a representation of $w$. This is the main idea behind word2vec representations that we address next. 

## Word2Vec

In 2012, Thomas Mikolov, an intern at Microsoft, found a way to encode the meaning of words in a modest number of vector dimensions. Mikolov trained a neural network to predict word occurrences near each target word. In 2013, once at Google, Mikolov and his teammates released the software for creating these word vectors and called it Word2vec.



> Most of the material presented here are from the sources below:

* [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) - see also the [2019 version of the video lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
* [Natural Language Processing in Action](https://www.amazon.com/Natural-Language-Processing-Action-Understanding/dp/1617294632). This book takes a hands on perspective to NLP. It uses both nltk (suitable for students and researchers) and spacy (suitable for developers) for implementing some of the stages of the NLP pipelines.
* 