---
layout: post
title:  "Analysis of Deep Neural Networks for YouTube Recommendations"
published: true
mathjax: true
date:   2022-09-10 18:30:13 -0400
categories: default
tags: [Machine Learning, Deep Neural Networks, Recommendation]
---


Here we introduce a Youtube's recommendation system based on deep neural networks. It consists of a deep candidate generation model and then a separate deep ranking model. Youtube recommendations have to deal with scalability, freshness and noises of user feedbacks.
The two-stage of this model can get a small personalized and accurate recommendation for users from a large of millions of videos.

<img src="/assets/images/2022_09_10/recommendation_youtube/recommendation_system_architecture.png" width="300">
 
#### The candidate generation architecture
<img src="/assets/images/2022_09_10/recommendation_youtube/deep_candidate_generation.png" width="300">
 
 
* The purpose of candidate generation is to get a high recall.
 
* Input features for candidate generation:
each user's YouTube activity history (IDs of video watches, search query tokens and demographics, geographic and device features.)
 
* Offline evaluation: precision, recall, ranking loss... Online evaluation: A/B testing measuring click through rate, watch time.
 
* It models as extreme classification each video $$i$$ is a class.
 
* It uses the implicit feedback [16] of watches to train the model, where a
user completing a video is a positive example.
 
* It uses a user's history only to predict future watch instead of a held-out watch, which leaks future information.
 
* Sample negative classes with importance weighting are used for classification speedup.
 
* Training examples ages are used as additional features to deal with the freshness of recommendation.
 
* It generates a  fixed number of training examples per user, effectively weighting
our users equally in the loss function, as this prevents a small cohort of highly active users from dominating the loss.
 
 
#### The Ranking architecture
<img src="/assets/images/2022_09_10//recommendation_youtube/Deep_ranking_network_architecture.png" width="300">
 
* Ranking uses the similar deep neural network architecture as candidate generation, which assigns an independent score to each video impression using logistic regression.
 
* Proper normalization of continuous features was critical for convergence.
 
* It is to predict expected watch time given training examples that are either positive (the video impression was
clicked) or negative (the impression was not clicked).
 
#### Code implementation
Youtube data is not public.
We uses a small sampled movieLen dataset to simply implement a basic architecture.
 
[<span style="color:blue;"> Deep neural network recommendation tested in MovieLen dataset </span>](https://github.com/windhaunting/Machine-Learning-Deep-Learning-Codes-Practice/blob/main/recommendation_systems/deep_neural_network_recommendation.ipynb)
 
 
##### Reference:
Covington, P., Adams, J. and Sargin, E., 2016, September. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems (pp. 191-198).
