---
layout: post
title: guts of the transformer
date: 2025-10-30 10:00:00
description: 
tags: transformers language-models
categories: cliff-notes
---

## The Guts

A standard Transformer is a encoder-decoder model.

There are **4 main steps** in the encoder:

1. Word Embedding: Because numbers not strings.
2. Positional Encoding: Because order matters and transformers have no inherent sense of order like a recurrent model.
3. Self-Attention: Because for each word in the input sequence we need to know how relevant every other word in the input sequence is to it.
4. Residual Connections between 2 and 3: Because deep models forget earlier layers otherwise.

There are **5 main steps** in the decoder:

1. Word Embedding: Because numbers not strings.
2. Positional Encoding: Because order matters and transformers have no inherent sense of order like a recurrent model.
3. Self-Attention: Because for each word in the output sequence we need to know how relevant every other word in the output sequence is to it.
4. Encoder-Decoder Attention: Because for every word in the output sequence we need to know how relevant each input word is to it.
5. Residual Connections between 2 and 3, and between 3 and 4: Because deep models forget earlier layers otherwise.

Bunch of normalization stuff in between.

## The Variants

| Types | Details | Example| Usecase |
|-------|---------|---------|---------|
| Decoder-only | Uses Masked Self-Attention instead of regular Self-Attention (in step 3 of the decoder) because we want each word to attend to only the previous words, not future words i.e. _auto-regressive_. | GPT | Expansion, Given a prompt, generate a response. |
| Encoder-only | Uses regular Self-Attention (in step 3 of the encoder) because we want each word to attend to all other words in the input sequence whether before or after it. | BERT | Extraction, Given some text, generate a context-aware embedding useful for downstream tasks like classification, clustering, similarity search for RAG, etc. |
