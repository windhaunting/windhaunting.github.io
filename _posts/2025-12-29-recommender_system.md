---
layout: post
title:  "Recommender systems"
published: true
# mathjax: true
date:   2025-12-15 21:30:13 -0400
categories: default
tags: [Machine Learning, Recommender system]
---


Here are my notes for recommender systems.

Several types of machine learning algorithms are commonly available for building recommender systems:

##### Collaborative Filtering:

(1) User-Based Collaborative Filtering: This approach recommends items to a user based on the preferences and behaviors of similar users. It finds users with similar tastes and recommends items liked by those similar users.

(2)  Item-Based Collaborative Filtering: Instead of finding similar users, this method identifies similar items and recommends items that are similar to those a user has already interacted with. The item-item similarity relies on calculating from user-item interactions because it leverages the patterns and preferences observed in the behavior of users to infer similarities between items. The underlying assumption is that items that are frequently interacted with by the same users are likely to be similar or share certain characteristics.

##### Matrix Factorization:

(1) Singular Value Decomposition (SVD): SVD factorizes the user-item interaction matrix into three matrices, which can be used to make recommendations. It is a foundational technique for matrix factorization-based recommenders.

(2) Factorization Machines:

Factorization machines are a generalization of matrix factorization methods and can capture complex interactions between users and items. They are known for their ability to handle feature engineering efficiently.
(3)  Matrix Factorization with Deep Learning: More advanced techniques use deep learning models to factorize the user-item interaction matrix, allowing for better generalization and handling of sparse data.


##### Content-Based Filtering:

This approach recommends items to users based on the content or characteristics of the items and the user's past preferences. It typically involves text analysis and feature extraction to understand the content of items.


##### Hybrid Recommender Systems:

Hybrid systems combine multiple recommendation techniques to improve recommendation quality. For example, combining collaborative filtering and content-based filtering can provide better results by leveraging both user behavior and item characteristics.


##### Association Rule Mining:

This technique identifies patterns and associations between different items that are often purchased or used together. It's commonly used in retail for market basket analysis.

##### Context-Aware Recommender Systems:

These systems take into account additional contextual information, such as time, location, or user behavior history, to make more relevant recommendations.


##### Session-Based Recommender Systems:

These systems are designed to make recommendations based on a user's current session or sequence of interactions, which is common in scenarios like e-commerce and content streaming platforms.
Reinforcement Learning:

##### Reinforcement learning:

This techniques can be used to optimize long-term user engagement by learning to make recommendations that maximize a predefined reward function.

##### Deep Learning for Recommender Systems:

Deep neural networks, including various architectures like FNN, RNN, CNN can be used to build sophisticated recommendation models. Neural collaborative filtering is an example of using neural networks for collaborative filtering.


 Neural collaborative filtering method:

Two Tower Neural Network is a collaborative filtering approach.



#### Modern Architecture of Recommender Systems in Industry
In most of industry production-grade Recommender systems, a multi-stage design is commonly used.
These stages include candidate retrieval, ranking and reranking. candidate retrieval tries to retrive highly relevant items, sometimes mix with some 
business constraints with high recall. ranking stage rank with smaller set of items with high precision. sometimes, ranking stage also include two stages- light ranking and heavy ranking to balance the recall and precision. reranking stage could further consider some business constraint and diversity, etc.


### Generative Recommender Systems
with the development of Generative AI, generative recommender system are actively in research and shows promissing results. It generates item IDs directly as tokens using a sequence model which treats recommendation as a sequence generation problem. I would like to cover in the later blogs.


## Some of the common problems in recommender systems
### Cold Start Problem:

Addressing the cold start problem in recommendation systems, particularly for new items or new users, requires innovative strategies to provide meaningful suggestions to users. Here are the top five methods.

* Content-Based Recommendations:

Utilize item features and metadata for new items. Leverage content similarities to make personalized suggestions.
Effective for items with rich attribute information.

* Hybrid Recommender Systems: Combine content-based and collaborative filtering methods.
Mitigate cold start issues by using content information for new items.
Enhance recommendation accuracy and coverage.
Popularity and Trend Analysis:

*  Recommend popular items to users while monitoring emerging trends.
Introduce trending items to users based on real-time popularity.
Balance recommendations between established and novel content.
Knowledge-Based Recommendations:

*  Incorporate domain knowledge or expert input for new items.
Utilize information about item characteristics, genres, or themes.
Effective when explicit item features are available.
Active Learning and User Feedback:

*  Encourage user interaction and feedback, especially for new items.
Dynamically adapt recommendations based on user responses.
Enhance the system's understanding of user preferences over time


### Long-Tail Problem
Few popular items dominate interactions, Most items (long tail) are rarely seen. 
*  Use Diversity-aware ranking
*  Reweighting / debiasing
*  Generative retrieval (better tail coverage)
We might also create a separete pipelin for retrieval and ranking historical interacted items that are in long tails, and then combine with a classical retrieval-ranking pipeline taht has good generalization but cover most head items.



## Metrics

### Recommendation System Metrics

#### Offline Metrics
When you're building a recommendation system, you start by testing it with existing data. Here's how you can see how well it's working:

- **Precision**: This is about how many of the things you recommend are actually good choices. If you recommend 10 movies and your friend likes 7, you've got good precision.

- **Recall**: Think of all the movies your friend would like. Recall measures how many of those you actually recommended. It's about making sure you catch as many good options as possible.

- **F1 Score**: Sometimes, you want a balance between making lots of good recommendations (recall) and making sure they’re all great (precision). The F1 score gives you a single number that balances both.

- **NDCG**: This checks not just if you’re recommending the right things, but if you’re putting the best ones at the top of the list where your friend will see them first.

- **MAP**: If you’re recommending a bunch of things, MAP looks at how well you’re doing across all of them.

- **Hit Rate**: You just want to make sure you’ve got at least one thing right for your friend. This is your “did I get a hit?” metric.

- **Coverage**: How many different people and products can your system handle? High coverage means it works for lots of users and items.

- **AUC**: This measures how well your system can tell the difference between something your friend will like and something they won’t.

#### Online Metrics
Once your recommendation system is up and running with real users, you want to see how it's doing in the wild. Here’s what you should keep an eye on:

- **Click-Through Rate (CTR)**: This is like counting how often people click on the things you recommend. A high CTR means your recommendations are catching their interest.

- **Conversion Rate**: After they click, do they do something important, like make a purchase? This metric tells you if your recommendations are not just interesting, but effective.

- **Average Order Value (AOV)**: If your recommendations lead to bigger purchases, that's great news. AOV measures how much people are spending after following your advice.

- **Session Duration**: How long do people stick around after seeing your recommendations? If they stay longer, it’s a good sign they’re engaged.

- **Bounce Rate**: If people leave without interacting at all, that’s not what you want. Keeping this number low is important.

- **User Retention**: Are people coming back again and again? This tells you if your recommendations are keeping users interested over time.

- **Engagement Rate**: How much are users interacting with what you recommend—like sharing, liking, or commenting? The more engagement, the better!

- **A/B Testing**: You can test out two different versions of your recommendations to see which one people like more.

- **Revenue per User**: This looks at how much money each user brings in after seeing your recommendations. It’s a direct way to see the impact of your system.
