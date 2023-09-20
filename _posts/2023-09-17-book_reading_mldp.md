---
layout: post
title:  "Booking reading -- Machine Learning Design Patterns"
published: true
mathjax: true
date: 2023-09-17 21:12:11 -0400
categories: default
tags: [Machine Learning, Design Pattern]
---


"Machine Learning Design Patterns" authored by Valliappa Lakshmanan, Sara Robinson, and Michael Munn, from Google, is a highly recommended read for individuals engaged in machine learning and data science. This book serves as an invaluable resource, offering a comprehensive overview of best practices and design patterns for machine learning development. Within its pages, readers will discover practical solutions to the common hurdles encountered in ML projects, encompassing critical areas such as data preparation, model training, and deployment. The book's extensive coverage spans essential topics, including data preparation, model selection and training, evaluation and validation, as well as deployment and monitoring.

Below is the summary from chapter 2.

### Chapter 2: Design Patterns for Data Representation

**Linear Scaling**:

In the realm of data scaling, four commonly adopted techniques emerge:

1. Min-max scaling: This method computes scaled values through the formula x1_scaled = (2*x1 - max_x1 - min_x1)/(max_x1 - min_x1).

2. Clipping: To tackle the issue of outliers, this approach substitutes them with "reasonable" values instead of estimating the minimum and maximum from the training dataset.

3. Z-score normalization: The scaled values are determined using the equation x1_scaled = (x1 - mean_x1)/stddev_x1.

4.Winsorizing**: Employing the empirical distribution in the training dataset, this technique confines the dataset within the bounds set by the 10th and 90th percentiles of the data values (or other percentiles as required). The winsorized value is subsequently subjected to min-max scaling.

The act of "centering" the data within the [-1, 1] range results in a more spherical error function. Consequently, models trained on transformed data tend to converge more rapidly, making them more time and cost-efficient to train. Additionally, the [-1, 1] range provides the highest level of floating-point precision available.


**Nonlinear Transformations**:

Picture a scenario where our data deviates from the typical uniform or bell curve distribution—it's skewed. In such instances, applying a nonlinear transformation to the input data before scaling proves advantageous. One commonly utilized technique involves taking the logarithm of the input values prior to scaling. Additionally, one should consider alternative transformations like sigmoid functions or polynomial expansions (such as square, square root, cube, cube root, and so forth). The success of a transformation can be gauged by its ability to render the distribution of transformed values more uniform or more closely resembling a normal distribution.

Categorical Variables:
For handling categorical variables, we can utilize techniques such as Label Encoding and One-Hot Encoding, among others.

#### Design Pattern 1: Hashed Feature

The Hashed Feature design pattern addresses three prevalent challenges associated with categorical features: dealing with incomplete vocabularies, managing large model sizes due to cardinality, and handling issues related to cold starts.

Feature Cross:
A widely embraced technique in feature engineering involves creating what's known as a "feature cross." Essentially, a feature cross is a synthetic feature formed by combining two or more categorical features to capture their interactions. This approach introduces nonlinearity into the model, empowering it to make predictions that go beyond the capabilities of individual features. Feature crosses also expedite the learning of relationships between features, even in simpler models like linear ones. Consequently, employing feature crosses explicitly can expedite model training, making it more cost-effective, and reduce model complexity, thus requiring less training data.

#### Design Pattern 2: Embeddings

Embeddings represent a learnable data format that transforms high-cardinality data into a lower-dimensional space while preserving the relevant information needed for the learning task. These embeddings play a central role in contemporary machine learning and manifest in various forms across the field.

#### Design Pattern 3: Feature Cross

This design pattern entails the explicit creation of separate features for every possible combination of input values. Feature crosses, formed by concatenating multiple categorical features, capture interactions between them, thereby introducing nonlinearity into the model. This enhances the model's predictive capabilities, surpassing what individual features can offer. Feature crosses expedite the model's ability to learn relationships between these features. Although more complex models like neural networks and trees can autonomously learn feature crosses, explicitly employing feature crosses allows us to rely on training just a linear model. Consequently, this expedites model training.

Feature crosses demonstrate strong performance even when dealing with extensive datasets. While augmenting layers in a deep neural network could theoretically introduce sufficient nonlinearity to understand the behavior of feature pairs, it substantially prolongs the training process. In certain instances, a linear model equipped with a feature cross, as trained in BigQuery ML, delivers comparable results to a deep neural network trained without a feature cross, all while significantly reducing training time. When selecting features for a feature cross, it is imperative to avoid combining highly correlated features.

#### Design Pattern 4: Multimodal Input

This pattern confronts the challenge of representing data that encompasses various data types or expresses the same information in complex manners. It achieves this by amalgamating all available data representations into a unified format.


### CHAPTER 3: Design Patterns for Problem Representation

#### Design Pattern 5: Reframing

The Reframing design pattern involves altering the representation of a machine learning problem's output. For instance, you can reframe a problem that appears to be a regression task into a classification task and vice versa. An example of reframing is binning the output values of a regression problem to transform it into a classification problem. Another approach is multitask learning, where both classification and regression tasks are combined into a single model using multiple prediction heads. However, when using reframing techniques, it's crucial to remain mindful of potential data limitations and the risk of introducing label bias. Multitask learning in neural networks typically involves either hard parameter sharing or soft parameter sharing.

#### Design Pattern 6: Multilabel

The Multilabel design pattern applies to scenarios where you can assign multiple labels to a single training example. To implement this, you utilize the sigmoid activation function in the final output layer, which results in each value in the sigmoid array ranging between 0 and 1. When working with the Multilabel design pattern, you need to use multi-hot encoding for your labels. Even in the case of binary classification, where your model has a one-element output, binary cross-entropy loss is used. Essentially, a multilabel problem with three classes is treated as three individual binary classification problems.

### Design Pattern 7: Ensembles

The Ensembles design pattern involves the use of techniques that combine multiple machine learning models and aggregate their outputs to make predictions. This approach addresses the error in an ML model, which comprises three components: irreducible error, bias error, and variance error. Ensembles aim to mitigate the bias-variance trade-off in machine learning, enhancing model performance. Some common ensemble methods include bagging, boosting, and stacking.

Bagging involves bootstrapping samples to introduce diversity.
Stacking utilizes different types of models and features.
Boosting trains weak learners sequentially to correct each other's errors.
There are also other ensemble techniques like those incorporating Bayesian approaches or combining neural architecture search and reinforcement learning, such as Google's AdaNet and AutoML techniques.

### Design Pattern 8: Cascade

The Cascade design pattern is employed when a machine learning problem can be effectively divided into a series of smaller ML problems. For instance, if you need to predict values in both typical and unusual circumstances, a single model might ignore the rare unusual events. In such cases, a cascade approach breaks the problem into four parts:

1. A classification model to identify the circumstance.
2. A model trained on unusual circumstances.
3. A separate model trained on typical circumstances.
4. A model to combine the outputs of the two separate models.
This approach ensures that both rare and common scenarios are considered. However, it's essential to avoid splitting an ML problem unnecessarily, and cascade should be reserved for situations where maintaining internal consistency among multiple models is necessary.

### Design Pattern 9: Neutral Class

In various classification scenarios, introducing a neutral class can be beneficial. Instead of training a binary classifier, a three-class classifier can be used, with classes for "Yes," "No," and "Maybe." The "Maybe" class serves as the neutral option. A neutral class is especially useful when experts disagree, in customer satisfaction assessment, as a means to improve embeddings, or when reframing the problem.

### Design Pattern 10: Rebalancing

The Rebalancing design pattern provides solutions for dealing with imbalanced datasets where one label dominates the majority. It addresses situations where there are few examples for specific classes. To overcome this issue, consider metrics other than accuracy, such as precision, recall, or F-measure. Average precision-recall can provide better insights for imbalanced datasets. Rebalancing methods include downsampling the majority class, applying class weights, upsampling the minority class, reframing the problem, and combining these techniques. Combining downsampling with ensemble methods is also a practical approach.

Chapters 2 and 3 focus on the initial steps of structuring machine learning problems, including input data formatting, model architecture options, and output representation. The next chapter will delve into design patterns for training machine learning models, advancing in the machine learning workflow.


## CHAPTER 4: Training Model Patterns

### Design Pattern 11: Useful Overfitting

Useful Overfitting is a design pattern where we consciously avoid employing generalization techniques because we intentionally want to overfit the training dataset. This pattern is recommended in situations where overfitting can yield benefits, and it suggests conducting machine learning without regularization, dropout, or even a validation dataset for early stopping.

Overfitting proves valuable under two specific conditions:

When there is no noise, meaning the labels are precise for all instances.
When you have access to the complete dataset, leaving no instances unaccounted for. In this scenario, overfitting essentially involves interpolating the dataset.
For instance, when tackling partial differential equations (PDEs) using machine learning to approximate solutions, overfitting becomes a desirable approach. By intentionally allowing the model to overfit the training data, we can obtain valuable insights. Unlike typical cases where overfitting leads to erroneous predictions on unseen data, in PDEs, we already know there won't be any unseen data, and the model is essentially approximating a solution across the entire input spectrum. If a neural network can learn a set of parameters that result in a zero loss function, those parameters define the actual solution to the PDE.

Some scenarios where Useful Overfitting is applicable include Monte Carlo methods, knowledge distillation from a larger neural network, and overfitting on a small batch as a sanity check for both the model code and data input pipeline.

### Design Pattern 12: Checkpoints

In the Checkpoints design pattern, we periodically save the complete state of the model during training, creating snapshots of partially trained models. These partially trained models can serve as the final model (in cases of early stopping) or as starting points for further training (in cases of machine failures or fine-tuning).

This pattern becomes especially valuable when training processes are lengthy, as it mitigates the risk of losing progress due to machine failures. Instead of starting from scratch in the event of an issue, we can resume training from an intermediate checkpoint.

### Design Pattern 13: Transfer Learning

Transfer Learning involves taking a portion of a pre-trained model, preserving its weights, and incorporating these non-trainable layers into a new model tailored to address a similar problem, often on a smaller dataset.

Two common approaches to transfer learning are fine-tuning and feature extraction, each suited to specific scenarios and needs.

### Design Pattern 14: Distribution Strategy

The Distribution Strategy pattern entails performing training at scale across multiple workers, often incorporating caching, hardware acceleration, and parallelization.

This distribution strategy is typically carried out through two primary methods: data parallelism and model parallelism. In data parallelism, computation is distributed across different machines, with each worker training on distinct subsets of the training data. In contrast, model parallelism involves dividing the model itself, with various workers handling computations for different portions of the model.

Furthermore, data parallelism can be executed either synchronously or asynchronously, depending on the specific requirements of the training process. Additionally, to improve training time, one can leverage specialized hardware such as GPUs, Google TPUs, Microsoft Azure FPGAs, address I/O limitations with high-speed interconnects, and optimize batch size, particularly in the context of synchronous data parallelism, especially when dealing with large models.

### Design Pattern 15: Hyperparameter Tuning

Hyperparameter Tuning involves subjecting the training loop to an optimization process to discover the ideal set of model hyperparameters.

Hyperparameters encompass any parameters within the model that a builder can control. These hyperparameters can be categorized into two groups: those related to model architecture and those related to model training. Model architecture hyperparameters govern aspects like the number of layers and neurons per layer, shaping the underlying mathematical function of the machine learning model. Meanwhile, training-related parameters, such as the number of epochs, learning rate, and batch size, control the training loop and often influence how the gradient descent optimizer operates.

Methods for optimizing hyperparameters include manual tuning, grid search, random search, Bayesian optimization, and genetic algorithms. These approaches help identify the hyperparameters that yield the best model performance for a given task.


## CHAPTER 5: Design Patterns for Resilient Serving

### Design Pattern 16: Stateless Serving Function

The Stateless Serving Function design pattern empowers a production ML system to efficiently handle thousands to millions of prediction requests per second in a synchronous manner. This system revolves around a stateless function that encapsulates the architecture and weights of a trained model.

Typically, servers establish a pool of stateless components, utilizing them to cater to client requests as they arrive. In contrast, stateful components need to maintain each client's conversational state, requiring intricate management, and consequently, they tend to be costly and challenging to handle. In enterprise applications, architects strive to minimize the use of stateful components. For instance, web applications often rely on REST APIs, transmitting state from the client to the server with each call.

Machine learning models capture significant state during training, including elements like the epoch number and learning rate. These are part of a model's state and must be retained since the learning rate commonly decreases with successive epochs. By mandating that models be exported as stateless functions, we compel model framework creators to manage these stateful variables separately, omitting them from the exported file. While the adoption of stateless functions simplifies server code and enhances scalability, it can introduce complexity into client-side code.

### Design Pattern 17: Batch Serving

The Batch Serving design pattern leverages distributed data processing infrastructure, typically used for large-scale data analysis, to perform inference on numerous instances simultaneously.

There are scenarios where predictions need to be made asynchronously for substantial volumes of data. For example, deciding whether to reorder a stock-keeping unit (SKU) might be a task carried out hourly, rather than every time a customer purchases the SKU.

The Batch Serving design pattern employs distributed data processing frameworks like MapReduce, Apache Spark, BigQuery, Apache Beam, and others to perform ML inference asynchronously on a large number of instances.

Lambda architecture:

A production ML system that supports both online serving and batch serving adopts a Lambda architecture. This allows ML practitioners to balance between latency, facilitated by the Stateless Serving Function pattern, and throughput, enabled by the Batch Serving pattern.

### Design Pattern 18: Continued Model Evaluation

The Continued Model Evaluation design pattern addresses the challenge of detecting and responding when a deployed model is no longer suitable for its intended purpose.

Two primary reasons models degrade over time are concept drift and data drift. Concept drift occurs when the relationship between model inputs and targets changes. Data drift, on the other hand, pertains to changes in the data used for prediction compared to the training data.

To identify model deterioration, it is essential to continuously monitor the model's predictive performance over time, using the same evaluation metrics as during development.

Scheduled Retraining

Determining the frequency of retraining depends on factors such as the business use case, the arrival of new data, and the cost implications of the retraining pipeline. One cost-effective tactic to gauge the impact of data and concept drift is to train a model using outdated data and assess its performance on more recent data.

### Design Pattern 19: Two-Phase Predictions

The Two-Phase Predictions design pattern offers a solution for maintaining the performance of large, complex models when deploying them on distributed devices. It involves splitting use cases into two phases, with the simpler phase executed at the edge.

Deploying machine learning models cannot always rely on stable internet connections for end-users. In such cases, models are deployed directly on user devices, requiring them to be smaller and posing trade-offs between model complexity, size, update frequency, accuracy, and low latency.

With the Two-Phase Predictions pattern, the problem is divided into two parts. A smaller, more straightforward model is initially deployed on the device, addressing the primary task with relatively high accuracy. A second, more complex model is deployed in the cloud and activated only when necessary. Implementing this pattern depends on the ability to divide the problem into two segments with varying complexity levels.

Another approach is to provide offline support for specific use cases, ensuring certain parts of the application remain functional for users with limited internet connectivity.

### Design Pattern 20: Keyed Predictions

In typical scenarios, machine learning models are trained on a set of input features consistent with those provided during real-time deployment. However, the Keyed Predictions design pattern introduces the concept of clients supplying a key along with their input, particularly useful for scenarios where handling large input-output datasets is involved.

Consider a model deployed as a web service, receiving a file containing a million inputs and returning a file with a million output predictions. Without client-supplied keys, it becomes challenging to match specific outputs to their corresponding inputs.

The solution lies in using pass-through keys, where each input is associated with a key supplied by the client. Client-supplied keys prove beneficial not only in batch prediction scenarios but also in asynchronous serving and continuous evaluation situations.


## CHAPTER 6: Design Patterns for Reproducibility

### Design Pattern 21: Transform

The Transform design pattern, especially in BigQuery ML, streamlines the process of deploying an ML model into production by meticulously segregating inputs, features, and transformations.

In cases where the chosen framework lacks inherent support for the Transform pattern, it's imperative to structure the model architecture to ensure that the transformations applied during training can be faithfully replicated during the serving phase. This can be achieved by preserving the transformation details within the model graph or by maintaining a repository of transformed features.

Alternatively, another approach to mitigate the training-serving skew problem is the utilization of the Feature Store pattern. The feature store entails a coordinated computation engine and a repository housing transformed feature data.

### Design Pattern 22: Repeatable Splitting

To ensure the repeatability and reproducibility of data sampling, it's vital to employ a well-distributed column and a deterministic hash function for partitioning available data into training, validation, and test datasets.

However, there are situations where a sequential split of data is warranted, particularly when there are strong correlations between consecutive instances. For instance, in weather forecasting, consecutive days exhibit high correlations, making it inappropriate to place October 12 in the training dataset and October 13 in the testing dataset, given the possibility of significant leakage (e.g., due to a hurricane on October 12). To address this, incorporating seasonality is essential, involving the inclusion of days from all seasons in each split. For instance, the first 20 days of every month could be placed in the training dataset, followed by the next 5 days in the validation dataset, and the last 5 days in the testing dataset.

### Design Pattern 23: Bridged Schema

The Bridged Schema design pattern offers a mechanism to adapt training data, originally structured under an older data schema, to a newer, improved schema. This pattern proves valuable because when data providers enhance their data feeds, it often takes time to accumulate enough data in the updated schema to adequately train a new model. The Bridged Schema pattern enables the utilization of the newer data to the fullest extent possible while supplementing it with relevant older data to enhance model accuracy.

Critical to this pattern's success is the comparison of the performance of the newer model, trained on bridged examples, against the older, unaltered model using an evaluation dataset.

### Design Pattern 24: Windowed Inference

The Windowed Inference design pattern addresses models that demand a continuous sequence of instances for conducting inference. This pattern functions by externalizing the model's state and invoking it through a stream analytics pipeline. It is particularly useful when a machine learning model relies on features computed from time-windowed aggregates. By externalizing the model state within a stream pipeline, the Windowed Inference design pattern ensures that features calculated dynamically over time windows can be consistently reproduced between training and serving, preventing training-serving skew, especially in cases involving temporal aggregate features.

To handle inference for models requiring a sequence of instances, the solution lies in implementing stateful stream processing, allowing for the maintenance of the model's state over time.

### Design Pattern 25: Workflow Pipeline

The Workflow Pipeline design pattern addresses the challenge of creating an end-to-end reproducible pipeline by containerizing and orchestrating various steps within the machine learning process. Containerization can be carried out explicitly or facilitated through frameworks designed for this purpose.

In recent years, the shift from monolithic applications to a microservices architecture has gained traction. In this approach, business logic is broken down into smaller, independently developed and deployed components, enhancing manageability.

To effectively scale machine learning processes, each step in the ML workflow can be transformed into a separate, containerized service. Containers ensure consistent code execution across different environments, yielding predictable behavior across runs. These containerized steps can then be seamlessly connected to form a pipeline, which can be triggered with a simple REST API call.

Several tools are available for creating pipelines, including Cloud AI Platform Pipelines, TensorFlow Extended (TFX), Kubeflow Pipelines (KFP), MLflow, and Apache Airflow.

### Design Pattern 26: Feature Store

The Feature Store design pattern simplifies the management and reuse of features across projects by decoupling feature creation from model development.

While ad hoc feature creation may suffice for one-off model development, it becomes impractical and problematic as organizations scale. Challenges include the lack of feature reusability, data governance complications, and difficulties in sharing features across teams and projects, leading to training-serving skew.

The solution is the establishment of a shared feature store, serving as a centralized repository for feature datasets used in machine learning model construction. It acts as the bridge between data engineers responsible for feature creation pipelines and data scientists crafting models. To mitigate training-serving skew risks arising from inconsistent feature engineering, feature stores ensure that feature data is written to both an online and an offline database, separating upstream feature engineering from model serving and ensuring point-in-time correctness.

### Design Pattern 27: Model Versioning

In the Model Versioning design pattern, backward compatibility is achieved by deploying altered models as microservices with distinct REST endpoints. This becomes a vital prerequisite for implementing many other patterns.

As highlighted earlier with data drift, models can become outdated over time, necessitating regular updates to align with an organization's evolving goals and changing data environments. Updating models in production impacts their behavior on new data, necessitating a solution for keeping production models current while ensuring backward compatibility for existing users.

To address this challenge effectively, multiple versions of a model can be deployed concurrently, each associated with a distinct REST endpoint. This approach ensures backward compatibility, allowing users reliant on older versions to continue using the service. Versioning also facilitates granular performance monitoring and analytics tracking across model versions, enabling accurate decisions about when to retire a specific version. Moreover, this design pattern supports A/B testing for model updates with a subset of users.



### Chapter 7: Responsible AI

#### Design Pattern 28: Heuristic Benchmark

The Heuristic Benchmark pattern serves as a valuable tool for assessing the performance of an ML model by comparing it to a straightforward and easily comprehensible heuristic. This approach aids in conveying the model's performance to business decision-makers in a clear and understandable manner.

However, it's important to note that the use of a heuristic benchmark is not recommended when an established operational practice is already in place. In such cases, the model should be evaluated against the existing standard, regardless of whether it employs ML techniques or not. The existing operational practice, which may not necessarily rely on ML, represents the prevailing approach used to address the problem.

#### Design Pattern 29: Explainable Predictions

The Explainable Predictions design pattern plays a crucial role in enhancing user trust in ML systems by providing users with insights into how and why models arrive at specific predictions. While models like decision trees inherently offer interpretability, the complex architectures of deep neural networks often make them challenging to explain. Nonetheless, understanding predictions, especially the factors and features influencing model behavior, is vital across all model types.

To address the inherent opacity of ML models, there is a growing area of research dedicated to developing techniques for interpreting and conveying the rationale behind model predictions. This field, often referred to as explainability or model understanding, continues to evolve rapidly. The interpretability of a model's predictions can vary significantly based on the model's architecture and the nature of its training data. Explainability also plays a role in uncovering biases within ML models.

Simple models like decision trees are typically easier to interpret because their learned weights directly reveal the factors influencing predictions. For instance, in a linear regression model with independent numeric input features, the weights often offer direct interpretability.

Open-source libraries such as SHAP provide Python APIs for obtaining feature attributions across various model types, based on the concept of Shapley Value.

However, it's essential to acknowledge the limitations of explainability. For instance, explanations are a direct reflection of training data, the model, and the chosen baseline. Therefore, if the training dataset is an inaccurate representation of relevant groups or if the chosen baseline is ill-suited for the problem at hand, the quality of explanations may be compromised.

Additionally, two alternative approaches to explainability include counterfactual analysis and example-based explanations. Counterfactual analysis focuses on identifying dataset examples with similar features that yield different predictions, often using tools like the What-If Tool. Example-based explanations compare new examples and predictions to similar instances in the training dataset, providing insights into how training data influences model behavior. These approaches are particularly effective for image or text data, offering intuitive insights into model predictions.

However, it's crucial to recognize that the relationships explained by these methods are context-bound, representative only within the framework of the training data, model, and specified baseline value.

#### Design Pattern 30: Fairness Lens

The Fairness Lens design pattern recommends the adoption of preprocessing and postprocessing techniques to ensure that model predictions are fair and unbiased, catering to different user groups and scenarios.

Data bias, a common challenge, can either be naturally occurring or problematic bias, which arises when different groups are impacted unevenly due to statistical properties in the original dataset. Experimenter bias can introduce bias during the data labeling process, while the choice of objective functions during model training may also contribute to bias.

Before applying the tools outlined in this section, it's prudent to conduct a thorough analysis of both the dataset and the prediction task to assess the potential for problematic bias. This involves examining the model's impact on various user groups and understanding how these groups will be affected. If problematic bias is anticipated, the technical strategies discussed in this section offer a starting point for mitigating such bias.

Tools like the What-If Tool are invaluable for conducting bias analysis, providing an open-source platform for evaluating datasets and models. These tools can be effectively utilized from various Python notebook environments, aiding in the essential task of addressing bias in ML models.

$$
Attention(Q, K, V) = softmax(sim(Q, K^T))*V

where sim(Q, K) = Q*K/sqrt(d_k)
$$

$Q$, $K$, $V$ dimensions are $n$ x $d$,  Attention's dimension is $n$ x $d$.


## Reference:
* Lakshmanan, Valliappa, Sara Robinson, and Michael Munn. Machine learning design patterns. O'Reilly Media, 2020.
* https://github.com/GoogleCloudPlatform/ml-design-patterns/tree/master







