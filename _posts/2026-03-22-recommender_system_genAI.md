---
layout: post
title: "Generation is the New Retrieval: A Tour of Generative Recommender Systems"
published: false
date: 2026-03-22
categories: default
tags: [generative retrieval, RecSys, LLM, RAG, DSI]
---


# Generation is the New Retrieval: A Tour of Generative Recommender Systems

Recommender systems have been on a quiet but weird journey over the last few years.

For ages, the recipe was simple: learn embeddings for users and items, find nearest neighbors (FAISS, Annoy, whatever), then run a heavy ranker on top. Classic.

But then LLMs happened, and people started asking a strange question:

> Do we even need to *retrieve* items? What if we just *generate* them instead?

Turns out, that question spawned a whole family of research—from sequential recommendation to LLM ranking to fully generative retrieval. Let me walk you through the papers and ideas that matter.

---

## 1. Where it started: Sequential recommendation

Before “generative retrieval” was a buzzword, we already had Transformers modeling user clicks. Two papers basically set the stage:

- **SASRec** (Kang & McAuley, 2018) – Takes a user’s history and predicts the *next* item, GPT‑style. Simple, effective, still widely used.
- **BERT4Rec** (Sun et al., 2019) – Uses masked item prediction to catch bidirectional context. Fancy, but harder to deploy online.

The key insight back then: treat item sequences like language. But both still need a retrieval step after the model runs. You generate a user vector, then go hunt in an item index. That separation became the bottleneck that later work wanted to kill.

---

## 2. LLMs enter the chat: From ranking to generation

Once LLMs got good, researchers couldn’t help themselves. They asked: *why not just use a language model directly as a recommender?*

Two broad directions emerged.

### Zero‑shot ranking

There’s a paper by Hou et al. (2023-ish) on “LLMs as Zero‑Shot Rankers”. The idea is embarrassingly simple:  
You dump the user’s history and some candidate items into a prompt, and ask the LLM to pick the best. No training, no fine‑tuning. It actually works okay for cold‑start or semantic matches, but it’s slow and can’t scale to millions of candidates.

### Generative recommendation

Then Bao et al. took it further. Instead of ranking existing candidates, they made the LLM **generate the next item directly**. That flips the problem: you’re not selecting from a fixed catalog anymore. The model could, in theory, invent a reasonable item it’s barely seen before.

Cool, but also a bit scary (hallucination is real).

---

## 3. The crazy one: Removing the index entirely

This is where things get radical. Two papers from 2021–2022 asked: *what if the model itself is the index?*

### Differentiable Search Index (DSI) – Tay et al., Google 2022

They replaced a full search index with a Transformer. Every document gets a learned ID. The model learns to map queries → document IDs. No FAISS, no ANN. Just the weights.

### Autoregressive Entity Retrieval (AER) – De Cao et al., 2021

Similar vibe, but focused on entities (products, knowledge graph entries). You treat each entity ID as a token sequence, and generate it token‑by‑token given a query.

So retrieval becomes:  
`P(item | query) = product over tokens of P(token | query, previous tokens)`

No nearest neighbor search. Just decoding.

In theory, this scales better with corpus size because cost depends on ID length, not number of items. In practice, decoding is slower than a dot product. But for batch or low‑latency‑tolerant use cases, it’s fascinating.

---

## 4. The pragmatic hero: RAG

Let’s be real. Most of us can’t ditch our indexes overnight. That’s where **Retrieval‑Augmented Generation (RAG)** comes in (Lewis et al., 2020).

RAG doesn’t replace retrieval. It just makes it smarter:

1. Retrieve top‑k candidates with your old‑school index (BM25, FAISS, whatever).
2. Feed them into a generative model to produce the final answer or ranking.

This is the dominant architecture in production today. You get scalability from the index, reasoning from the LLM, and grounding from the retrieved items. Win‑win.

---

## 5. Generative ranking: A smaller step that actually works

Not everyone wants to throw away their retrieval pipeline. Some just want to improve the ranking stage.

Idea: replace your pointwise scoring function with LLM probabilities.  
`score(item) = log P(item | user_context)`

You can do pointwise (score each item) or listwise (generate the whole ordered list). This is surprisingly effective in reranking, and it’s much easier to deploy than full generative retrieval.

---

## 6. What’s still hard

Pure generative retrieval isn’t winning production hearts (yet). Why?

- **Decoding latency** – Beam search is not cheap.
- **Item space size** – Once you have millions or billions of items, generating the right ID becomes a constraint satisfaction nightmare.
- **ID design** – You can’t use random IDs. They need structure, but designing that structure is more art than science.

That said, new work on **structured IDs** (hierarchical codes, tree‑based tokenization) and **attribute‑level generation** (predict brand, category, color, then the item) is making progress. For e‑commerce where attributes matter as much as the item itself, this is a promising direction.

---

## 7. So, how do you actually design item IDs for generative retrieval?

I promised a follow‑up, so here it is.

If you’ve read the papers on DSI or Autoregressive Entity Retrieval, you’ll notice they breeze over one painfully practical detail: *where do the item IDs come from?*

You can’t just use random UUIDs. The model would have to memorize a meaningless string for every item, which defeats the whole point of generalization. The IDs need **structure** – some pattern the model can learn and exploit.

Over the last couple of years, people have tried a few different strategies. Here’s what works (and what doesn’t).

---

### Approach 1: Hierarchical IDs (the safe bet)

This is the most common trick. Split the ID into levels that mirror your category tree.

Example for e‑commerce:
[level1: category] → [level2: subcategory] → [level3: product type] → [level4: numeric id]

text

So a coffee maker might become:
Home → Kitchen Appliances → Coffee Makers → 00472

text

Why this helps:  
The model learns that “Home → Kitchen Appliances → Coffee Makers” is a common prefix. Even if it never saw product `00472` during training, it can guess the right prefix based on the query. That’s the whole point of generalization.

Downside:  
If your category tree is messy or changes often (hello, marketplace sellers inventing new categories), you’re in for a world of pain.

---

### Approach 2: Semantic IDs from product titles or descriptions

Instead of hand‑crafting categories, you can cluster items using a pretrained embedding (say, from a BERT model fine‑tuned on product titles). Then assign each cluster a token, and each item a cluster‑specific ID.

One paper that explored this direction is **“Semantic IDs for Generative Retrieval”** (though not as famous as DSI – it’s more of an emerging trick). The idea:

- Embed all items → cluster them → each cluster becomes a token.
- The ID is `[cluster_token] + [item_index_in_cluster]`.

The model learns to predict the semantic cluster first, then the specific item. This works surprisingly well because similar items live in similar clusters.

---

### Approach 3: Learned IDs (the fancy but fragile one)

DSI tried this: treat the IDs as learnable parameters. You randomly initialize an ID sequence for each item and train the model to predict them end‑to‑end.

In theory, the model learns whatever ID structure helps it retrieve correctly. In practice, it’s a black box. You get IDs that make no sense to a human, and the model sometimes memorizes instead of generalizing. I’ve seen teams waste weeks debugging this.

My take: skip learned IDs unless you have tons of data and a strong regularization budget.

---

### Approach 4: Attribute‑level generation (the new hotness)

Some recent work (not yet a single canonical paper) suggests: don’t generate item IDs at all. Generate the **attributes** that define an item, then map attributes to actual items via a lookup table.

So for a query “bluetooth speaker under $50”, the model generates:
brand: Anker, category: Speakers, max_price: 50, color: black

text

Then a simple key‑value store returns the matching item.

Why this is clever:

- The model doesn’t have to memorize millions of IDs.
- It only needs to learn attribute distributions – much easier.
- The mapping from attributes to items can be done with a fast hash table.

The tradeoff: if two items have identical attributes, you need a fallback (like a numeric suffix). But in e‑commerce, that’s rare enough to ignore.

---

## What I’d recommend for a real project

If you’re building a generative retrieval system tomorrow:

1. **Start with hierarchical IDs** based on your existing category taxonomy. It’s boring but it works.
2. **Embed a sanity check** – if the model generates an invalid prefix (e.g., “Electronics → Clothing”), mask it during beam search. Constrained decoding saves you from nonsense.
3. **Consider attribute generation** if your catalog is huge (millions+) and your data is attribute‑rich. It’s more future‑proof.
4. **Avoid pure learned IDs** unless you enjoy sadness.

And one more thing: no matter the ID design, you still need a **fallback retrieval mechanism**. Generative retrieval will fail for tail queries. Keep a small ANN index around for low‑probability predictions. Hybrid systems win again.

---

## 8. Where I think we’re headed

Despite the hype, I don’t see pure generative retrieval killing dense retrieval in the next year or two. But the trajectory is clear:

- Recommendation is becoming sequence modeling (SASRec, BERT4Rec started this).
- Retrieval is becoming conditional generation (DSI, AER).
- Ranking is becoming probabilistic decoding (LLM rankers).
- Production systems are converging on **hybrid RAG architectures** – retrieve a bunch, then generate/rerank.

It’s messy, but it’s also the most interesting change in RecSys since the deep learning wave.

---

## References

1. **SASRec** – Kang, W. C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *IEEE ICDM*. [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)

2. **BERT4Rec** – Sun, F., et al. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. *CIKM*. [arXiv:1904.06690](https://arxiv.org/abs/1904.06690)

3. **LLMs as Zero-Shot Rankers** – Hou, Y., et al. (2023). Large Language Models are Zero-Shot Rankers for Recommender Systems. *ECIR*. [arXiv:2305.08845](https://arxiv.org/abs/2305.08845)

4. **Generative Recommendation (Bao et al.)** – Bao, K., et al. (2023). Generative Recommendation: Towards Next-Generation Recommender Paradigm. *arXiv preprint*. [arXiv:2304.03516](https://arxiv.org/abs/2304.03516)

5. **Autoregressive Entity Retrieval (AER)** – De Cao, N., et al. (2021). Autoregressive Entity Retrieval. *ICLR*. [arXiv:2010.00904](https://arxiv.org/abs/2010.00904)

6. **Differentiable Search Index (DSI)** – Tay, Y., et al. (2022). Transformer Memory as a Differentiable Search Index. *NeurIPS*. [arXiv:2202.06991](https://arxiv.org/abs/2202.06991)

7. **RAG** – Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

