---
layout: post
title:  "Summary of Predictive Model Performance: Offline and Online Evaluations"
published: true
date:   2022-11-19 20:30:13 -0400
categories: default
tags: [Machine learning, Data Science, Offline Metrics, Online Metrics, Predictive Models, Ads, CTR]
---

Here we summarize a paper from "Predictive Model Performance: Offline and Online Evaluations" from Microsoft. It analyzes the offline and online metric discrepancy problem, and the simulated metric for simulating online performance.

##### The Offline and online discrepancy problem:

One problem with the model evaluations in reality is that sometimes the improvement of model performance in offline evaluation does not get realized as much, or sometimes gets reversed in online evaluation.

Online testing even under the controlled environment is highly dynamic, of course, and many factors not considered during the offline modeling play a role in the results.

Another problem is to compare the performance of predictive models built with different kinds of data, especially data with rare events.
<br />
<br />

##### The Simulated metrics is a good choice to evaluate online performance:

A more significant problem with the offine evaluation metrics in practice is the discrepancy in performance between the offine and online testing.

Instead of using expensive and time consuming online evaluation, the performance of a model over the entire span
of feasible operating points can be simulated using the historic online user engagement data.

They implement auction simulation using sponsored search click logs data and produced various simulated metrics. Auction simulation, first, reruns ad auctions offline for the given query and selects a set of ads based on the new model prediction scores. During the simulation, user clicks are estimated using historic user clicks of the given (query, ad) pair available in the logs.
<br />
<br />


##### Common offline metrics for predictive models (especially ads click prediction):

Offline Metrics:

* Probability-based: AUC, MLE (Maximum Likelihood Estimator), etc
* Log Likelihood-based: RIG, cross-entropy, etc
* PE (prediction Error): MSE (Mean Square Error), MAE (Mean Absolute Error), RMSE (RootMean Square Error), etc
* DCG-based: DCG (Discounted Cumulative Gain), NDCG (Normalized DCG), RDCG (Relative DCG), etc
* IR (Information Retrieval): Precision/Recall, F-measure, AP(Average precision), MAP(Mean Average Precision), RBP (Rank-Based Precision), MRR (Mean Reciprocal Rank), etc
* Misc: everything else that does not belong to one of the other categories

Online metrics include model performance statistics:

* Ad impression yield
* Ad coverage
* User reaction metrics, such as CTR and the length of user sessions.
<br />
<br />


##### The summary of the peformance evaluation:

AUC alone is not sufficient enough to estimate the model performance reliably.

Both RIG and the AUC are highly sensitive to the class distribution of the evaluation data (I also obsereved from several previous projects that especially for unbalanced data, the ROC AUC is not a good indicator, the PR AUC is more reliable).

It is suggested to measure model performance in various quantiles, and carefully analyze how the change
of model behavior over the range of quantiles would impact in the online environment. One may review
various metrics together to discover any mismatch in the results, which may suggest some problems in the
metrics.

##### Reference:

Yi, J., Chen, Y., Li, J., Sett, S. and Yan, T.W., 2013, August. Predictive model performance: Offline and online evaluations. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1294-1302).