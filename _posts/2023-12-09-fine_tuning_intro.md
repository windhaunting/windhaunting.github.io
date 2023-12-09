---
layout: post
title: "Essentials of Fine-tuning LLM Models"
published: true
mathjax: true
date: 2023-12-09 15:15:16 -0400
categories: default
tags: [GenAI, LLM, NLP, Transfer Learning, Fine-tuning]
---

With the advent of large pre-trained language models like BERT and GPT-3, fine-tuning has emerged as a widely adopted method for transfer learning research. This approach entails customizing a pre-trained model for a specific task through training on a more modest dataset containing task-specific labeled data.

Transfer learning is the practice of using a pre-trained model created for a specific task as a starting point for a related task. It entails taking advantage of the feature representations learned by the pre-trained model and applying them to a new model. This new model is then fine-tuned or trained further using a smaller, task-specific dataset. Transfer learning is particularly beneficial when working with limited data for the new task, as it allows leveraging knowledge gained from a larger dataset used in the original task.

### Fine-tuning:

Fine-tuning, a subset of transfer learning, entails adjusting the weights of a pre-trained model on a task-specific dataset. It leverages the knowledge acquired during pre-training as a starting point but refines the model's parameters to better suit the characteristics of the new data. The degree of adjustment during fine-tuning is contingent on factors like the volume of available data and the similarity between the original and target tasks. This iterative process allows the model to adapt its learned features to the intricacies of the specific domain, optimizing performance for the task at hand. Fine-tuning strikes a balance between leveraging general knowledge from pre-training and tailoring the model to the nuances of a particular application.

##### Parameter efficient fine-tuning (PEFT):

As models continue to grow in size, performing complete fine-tuning on consumer-grade hardware becomes impractical. Moreover, the cost of storing and deploying individually fine-tuned models for each downstream task is significantly high, given that these fine-tuned models have the same size as the initial pre-trained model. Parameter-Efficient Fine-tuning (PEFT) approaches aim to tackle both of these challenges.

PEFT focuses on optimizing model performance with limited computational resources. It involves adapting a pre-trained model to a specific task while minimizing the number of parameters. Strategies such as knowledge distillation, architecture search, and neural architecture optimization are employed to achieve optimal performance with reduced computational demands. This approach is especially valuable in scenarios where computational resources are constrained, aiming to strike a balance between model efficiency and task-specific accuracy.

PEFT strives to fine-tune only a small subset of the model's parameters, delivering comparable performance to full fine-tuning but with a substantial reduction in computational demands. The common methods for PEFT are shown below:

- Additive: used to add new tunable layers to the model, keeping the foundation model weights frozen and updating only the new layer weights. Examples include soft prompt (adding virtual token/vectors into an input sequence and then fine-tuning these vectors), prompt tuning (involves using a small trainable model before using the LLM to encode the text prompt and generate task-specific virtual tokens), prefix tuning (tune prefixes for every layer), and Adapter in transformer (adapters are small learned layers inserted within each layer of a pre-trained model).

- Selective: update a few foundation model layers (e.g., BitFit, which only updates bias parameters; Diff Pruning, which creates task-specific diff vectors and only updates them).

- Re-parameterization: Decompose weight matrix updates into smaller-rank matrices (e.g., LoRA - low rank adaptation, Rank decomposition; IA(3), etc.).


#### Popular FT Methods:

**LoRA (Low Rank Adaptation) - Rank Decomposition:**

LoRA adopts a more parameter-efficient approach, where the task-specific parameter increment ΔΦ = ΔΦ(Θ) is further encoded by a much smaller-sized set of parameters Θ.

For a pre-trained weight matrix \(W_0 \in \mathbb{R}^{d\times k}\), we constrain its update by representing the latter with a low-rank decomposition
\[W_0 + \Delta W = W_0 + BA\]
where \(B \in \mathbb{R}^{d\times r}\), \(A \in \mathbb{R}^{r\times k}\).

**LoRA + int8 quantization:**

You can combine low-rank adaptation with int8 quantization to further optimize memory usage and speed up inference on hardware with specialized instructions for int8 operations, such as modern CPUs & AI accelerators.

**QLoRA:**

This efficient fine-tuning approach utilizes 4-bit quantization and introduces innovations like the NF4 data type, double quantization, and paged optimizers to optimize memory usage without compromising the performance of full 16-bit fine-tuning task performance.

**IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations):**

It is intended to improve over LoRA. It rescales inner activations with learned vectors.


#### Limitations of PEFT:

- Difficult to match the performance of full fine-tuning.
- It's not always efficient for inference.
- It still needs the training complexity of forward and backward, and storing the massive foundation models.

Fine-tuning is a dynamic field of study within active learning, with ongoing research and emerging techniques that are worth paying attention to.

### Practical ways to fine-tuning

1. **Use OpenAI fine-tuning:**

   If you have the OpenAI API subscription, you could fine-tune on the OpenAI's Fine-tuned Model of GPT-3.5 or GPT-4 (in the experimental stage so far). You could refer to the [OpenAI fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) page or this [ChatGPT3.5 fine-tuning video](https://www.youtube.com/watch?v=W4Q9bKLNYiQ&ab_channel=AllAboutAI) for detailed steps.

2. **Use Google PaLM 2 fine-tuning:**

   Google Vertex AI provided the GenAI model with text, coding, chat foundation model, and you could fine-tune on it. You could refer to this [Vertex AI Fine-tuning](https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models) with a step-by-step tutorial or use [PaLM 2 fine-tuning with Scikit-LLM](https://medium.com/@iryna230520/fine-tune-google-palm-2-with-scikit-llm-d41b0aa673a5).

3. **Other open-source ways:**

   There is a relatively high cost to use OpenAI and Google Vertex AI for fine-tuning models. Hugging Face provides lots of foundational models and a library for fine-tuning with several algorithms. For example, you could fine-tune a Transformer or llama 2 model with [LoRA algorithms through Hugging Face](https://www.philschmid.de/fine-tune-flan-t5-peft).


### Reference:

- Ruder, Sebastian, Jonas Pfeiffer, and Ivan Vulić. "Modular and Parameter-Efficient Fine-Tuning for NLP Models." In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts, pp. 23-29. 2022.

- Pfeiffer, Jonas, Andreas Rücklé, Clifton Poth, Aishwarya Kamath, Ivan Vulić, Sebastian Ruder, Kyunghyun Cho, and Iryna Gurevych. "Adapterhub: A framework for adapting transformers." arXiv preprint arXiv:2007.07779 (2020).

- Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

- Liao, Baohao, Yan Meng, and Christof Monz. "Parameter-Efficient Fine-Tuning without Introducing New Latency." arXiv preprint arXiv:2305.16742 (2023).

- Liu, Haokun, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A. Raffel. "Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning." Advances in Neural Information Processing Systems 35 (2022): 1950-1965.

- Dettmers, Tim, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. "Qlora: Efficient finetuning of quantized llms." arXiv preprint arXiv:2305.14314 (2023).

-  https://huggingface.co/blog/peft
