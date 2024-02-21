


LLM (large lanage model)


Foundational models:

### Prompt techniques:

##### AutoPrompt:
It explores the potential of automating generating prompts for language models on lots of tasks based on a gradient-guided search..

##### Optimizer by prompt

The idea is from DeepMind's paper: Optimization by PROmpting (OPRO), a method that uses AI large language models (LLM) as optimizers.


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

Retrieval Augmented Generation (RAG): It is used to complement prompt engineering's limited context problem. It uses external knowledge for context-rich answers to a query.  There are some evolved paradigm around RAG that Naive RAG,
advanced RAG, and modular RAG.

### Fine-tuning:
Fine-tuning replies on pre-training model to refines the model’s parameters to better suit the characteristics of the new data. For detail, please refer to the blog, <span style="color:blue;"> Essentials of Fine-tuning LLM Models </span>](https://windhaunting.github.io/2023/12/09/fine_tuning_intro){:target="_blank"}


###  Improve the efficency and reduce resources

##### Quantization: 

It is used to reduce the size of LLM by modifying the precision of their weights.
Quantization significantly decreases the model's size by reducing the number of bits required for each model weight. A typical scenario would be the reduction of the weights from FP16 (16-bit Floating-point) to INT4 (4-bit Integer). This allows for models to run on cheaper hardware and/or with higher speed. By reducing the precision of the weights, the overall quality of the LLM can also suffer some impact. 

##### FrugalGPT
It can adaptively selects which generative LLMs to use for different queries to reduce cost and improve accuracy.



Reference:

 Shin, Taylor, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. "Autoprompt: Eliciting knowledge from language models with automatically generated prompts." arXiv preprint arXiv:2010.15980 (2020).

Yang, Chengrun, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. "Large language models as optimizers." arXiv preprint arXiv:2309.03409 (2023).

Lewis, Patrick, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler et al. "Retrieval-augmented generation for knowledge-intensive nlp tasks." Advances in Neural Information Processing Systems 33 (2020): 9459-9474.

https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2 (Retrieval-Augmented Generation (RAG): From Theory to LangChain Implementation)

Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. "Lora: Low-rank adaptation of large language models." arXiv preprint arXiv:2106.09685 (2021).

Chen, Lingjiao, Matei Zaharia, and James Zou. "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance." arXiv preprint arXiv:2305.05176 (2023).

Frantar, Elias, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. "Gptq: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

