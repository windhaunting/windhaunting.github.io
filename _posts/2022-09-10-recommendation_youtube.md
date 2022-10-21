---
layout: post
title:  "Understanding of Deep Neural Networks for YouTube Recommendations"
published: false
mathjax: true
date:   2022-09-10 18:30:13 -0400
categories: default
tags: [Homomorphic Encryption, Privacy Preserving]
---

### Paper introduction

deep candidate generation model and then a separate deep ranking model.
Youtube recommendations have to deal with scalalibity, freshness and noises of user feedbacks.
The two-stage of this model can get a small personalized and accurate recommendation for users from a large of millions of videos.


<img src="/assets/images/2022_09_10//recommendation_youtube/recommendation_system_architecture.png" width="300">

<img src="/assets/images/2022_09_10//recommendation_youtube/deep_candidate_generation.png" width="300">

The purpose of candidate generation is to get a high recall.

Input features for candidate generation:
each user's YouTube activity history (IDs of video watches, search query tokens and demographics, geographic and device features.)

Offline evaluation: precision, recall, ranking loss...
Online evaluation: A/B testing measuring click through rate, watch time.

It modeles as extreme classification each video $$i$$ is a class.

It uses the implicit feedback [16] of watches to train the model, where a
user completing a video is a positive example.

Sample negative classes with importance weighting is used for classification speedup.

It uses training examples ages as additional features to deal with the freshness of recommendation.

It generates a  fixed number of training examples per user, effectively weighting
our users equally in the loss function, as this prevents a small cohort of highly active users from dominating the loss.

### Code testing
Youtube data is not public. 
There is a good implementation available in github:
[<span style="color:blue;"> Deep neural network recommendation tested in MovieLen dataset </span>](https://github.com/hyez/Deep-Youtube-Recommendations/blob/master/neural_net.ipynb
)
It uses the public movieLen dataset to test it.

