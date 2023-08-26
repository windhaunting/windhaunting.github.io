---
layout: post
title:  "Unlocking Innovation: My First Industry Hackathon Adventure with Walmart and Google"
published: true
mathjax: true
date: 2023-08-26 18:15:16 -0400
categories: default
tags: [Machine Learning, GenAI, LLM, Chatbot, Google, Walmart]
---

My first hackthon in a company, organized by Walmart and Google, was an exciting venture into the world of innovation and technology. Taking place over four days, starting from Monday and ending on Thursday, it was a whirlwind of creativity and collaboration.

Google took the stage with a spotlight on their Google Cloud Vertex GenAI platform and models – our toolkit for the hackathon. With this powerful resource at our disposal, we embarked on our development journey. The GenAI offer provided a plethora of capabilities, from text and code processing to chat-bot functionalities and image processing.

The kickoff day was a learning extravaganza as Walmart and Google experts introduced us to the hackathon and the intricacies of Vertex GenAI. Armed with newfound knowledge, each team was granted the opportunity to create one or more instances on the Vertex AI. The task at hand was to develop a use case around these instances, culminating in a demo and presentation on Friday morning. However, only the top 10 teams would have the privilege of presenting to a panel of experts from both Google and Walmart. Throughout the rest of the hackathon, we benefited from office hours with Google experts and a dedicated communication channel to enhance collaboration.

Initially, I rallied three colleagues to form our dream team, but when the hackathon commenced, it was just myself and another member, Richa, who embarked on this adventure.

Navigating privacy and security policies, we honed in on a use case that involved analyzing public customer product reviews from Walmart's website. This approach offered insights that could benefit Walmart associates, helping them enhance products and customer service. Our first step was to construct a review analyzer, summarizing product reviews across various dimensions. As our project evolved, we transformed it into an interactive service chatbot, designed to provide associates with insights beyond the ordinary.

Leveraging the power of the LLM (Large Language Model), our chatbot delivered impressive answers tailored to associate user queries, thanks to our carefully crafted prompts and fine-tuning parameters.

Yet, every journey has its challenges. Ours came in the form of managing a substantial volume of reviews. To address this, we employed a method akin to map-reduction, systematically crawling and storing reviews in Google Cloud GCS buckets in batches based on products and review counts. With these batches processed, we amalgamated insights for each associate user's questions. In light of time constraints, we considered the storage of reviews in a vector database, complete with embeddings, along with compression techniques to optimize search efficiency and minimize delays.

As the deadline loomed, we submitted our presentation and demo, marking the end of a remarkable journey. Looking back, this hackathon was a truly enriching experience. Challenges were overcome, knowledge was gained, and innovation thrived. A heartfelt thank you to Walmart and Google for providing this exceptional opportunity.

In conclusion, my inaugural hackathon was a rollercoaster of learning, innovation, and teamwork. The road was paved with challenges, but the camaraderie and growth made every moment worthwhile. Here's to the next adventure, continued innovation, and the lasting impact of this hackathon. Cheers to Walmart, Google, and the power of seizing opportunities!

