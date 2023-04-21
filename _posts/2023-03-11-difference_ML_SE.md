---
layout: post
title:  "The difference on working on machine learning and software engineering projects"
published: true
mathjax: true
date:   2022-03-11 18:30:13 -0400
categories: default
tags: [Machine Learning, Software Engineering, Job]
---

I have been working in the machine learning and data science (ML/DS) engineering area in industry for a year. 
I had been worked in the  software engineering (SE) area before. Some of the SE experience could be shifted to ML/DS,
but most of them are totally not transferrable.

More companies began to introduce ML/DS solutions to their existing systems and products to replace the traditional methods. 
Some of the common things between SE and ML/DS engineering:
(1) Do the objective and user requirement analysis 
(2) Desgin the system/model
(3) Develop and test
(4) Maintain and monitor
(5) Communicate and collaborate with other engineers or teams

However, the ML/DS engineering has its uniqueness. The ML engineering pipeline is a circular and repetitive cycle. It also relies on amounts of data.

(1) We are working on ML/DS to replace the traditional rule-based models.
The fancy deep and complex model is not the core.  The core is  improve the data quality for some existing ML models.
We spent lots of time on data retrieval and preprocessing.
There are a large amounts of data from different groups and department. There are lack of clear answer to which data might be availabe for our problem. Also the data quality is another concern. Therefore, to build an effective model,
our team focuses lots of time on data preprocessing

(2) Unlike the fixed function of input and output in SE, The features as an input of ML models are not easily constructed. Feature engineering is an important part for our projects. We are currently not working projects that needs no hand-engineered features like deep neural network based vision/image. Our projects need more feature engineering work that involve trial and error.

(3) The ML pipeline is more ambiguous and complicated than software engineering practice.
Traditional Software engineering has clear modularity and functionality, and constant input and output. Also, the code and data are separated and the versioning focus mostly on the code functionalities.
Machine learning's functional codes for modeling and data are almost confused together. The data are constantly changing and increasinly larger. Also different data input will lead to different output. For the test, you need to version data except from the functional codes.


(4) It is important to understand the key metrics for different teams,  especially the engineer/data scientist team and business team have different perspective and metrics to look at for a same problem. As an engineer, we need to understand the business and customer's care when we working on optimizing our metrics for a project.

(5) More monitoring and observation, and continous learning are needed after production for ML projects. There are data drift that needs continually adapt models to changing data distributions.



