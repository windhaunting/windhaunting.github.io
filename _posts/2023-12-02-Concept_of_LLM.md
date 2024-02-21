---
layout: post
title:  "Concepts of Large Language Model"
published: true
# mathjax: true
date:   2023-12-02 21:30:13 -0400
categories: default
tags: [Machine Learning, GenAI, Large Language Models, LLM]
---


### LLMs (Large Language Models)
Large language models (LLMs) are language models that can recognize, summarize, translate, predict, and generate content using very large datasets. They take text as input and predict what words or phrases are likely to come next. They are built using complex neural networks and trained on massive amounts of text data.

### Foundational models
In the context of LLMs, foundational models are the initial versions or iterations of these models upon which subsequent versions are built or derived. For example, the Transformer model is used as the foundational model for the GPT model. Many open-source LLMs in the Hugging Face models are created based on foundational models such as BERT, LlaMa, BLOOM, Gemma, etc."

### Prompt techniques

##### AutoPrompt:
It explores the potential of automating generating prompts for language models on lots of tasks based on a gradient-guided search [1].

##### Optimizer by prompt

The idea is from DeepMind's paper: Optimization by PROmpting (OPRO) [2], a method that uses AI large language models (LLM) as optimizers.

There are some disadvantages of using prompt Engineering. 

- **Prompt Sensitivity:**
  - LLMs are sensitive to prompt wording, necessitating careful crafting for desired responses.

- **Limited Context Understanding:**
  - LLMs may struggle with broader context, resulting in contextually incorrect responses and difficulty understanding nuanced user queries.

- **Inefficiency in Prompt Processing:**
  - Processing prompts for each inference can lead to inefficiencies in model performance.

- **Performance Challenges:**
  - Generally, prompt engineering may result in poorer performance compared to fine-tuning approaches.


### RAG

Retrieval Augmented Generation (RAG): It is used to complement the limited context problem of prompt engineering. RAG utilizes external knowledge to provide context-rich answers to a query [3]. There are several evolved paradigms around RAG, including Naive RAG, Advanced RAG, and Modular RAG [4].

### Fine-tuning:
Fine-tuning replies on pre-training model to refine the model’s parameters to better suit the characteristics of the new data [5]. For the detail, please refer to the blog, <span style="color:blue;"> Essentials of Fine-tuning LLM Models </span>](https://windhaunting.github.io/2023/12/09/fine_tuning_intro){:target="_blank"}


### Improve the efficency and reduce resources

##### Quantization: 

Quantization is used to reduce the size of LLMs by modifying the precision of their weights [6]. Quantization significantly decreases the model's size by reducing the number of bits required for each model weight. A typical scenario would be the reduction of the weights from FP16 (16-bit Floating-point) to INT4 (4-bit Integer). This allows models to run on cheaper hardware and/or at higher speeds. However, by reducing the precision of the weights, the overall quality of the LLM can also be impacted.

##### FrugalGPT
It can adaptively select which generative LLMs to use for different queries to reduce cost and improve accuracy [7, 8]. It tries to balance between cost and accuracy when it comes to the adoption of different LLM technologies.


#### Reference:

[1] Shin, Taylor, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. "Autoprompt: Eliciting knowledge from language models with automatically generated prompts." arXiv preprint arXiv:2010.15980 (2020).

[2] Yang, Chengrun, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. "Large language models as optimizers." arXiv preprint arXiv:2309.03409 (2023).

[3] Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

[4] https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2 (Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation)

[5] Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

[6] Frantar, Elias, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).


[7] Chen, Lingjiao, Matei Zaharia, and James Zou. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv preprint arXiv:2305.05176 (2023).

[8] https://medium.com/@ronnyh/frugalgpt-a-game-changer-in-ai-for-small-businesses-d8d385cb13d