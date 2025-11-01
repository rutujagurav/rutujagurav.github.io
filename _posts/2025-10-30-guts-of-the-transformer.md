---
layout: post
title: Guts of the Transformer
date: 2025-10-30 00:00:00
description: 
tags: musings
categories: 
---

A standard Transformer is a encoder-decoder model.

There are 4 main steps in the encoder:

1. Word Embedding: Because numbers not strings.
2. Positional Encoding: Because order matters and transformers have no inherent sense of order like a recurrent model.
3. Self-Attention: Because for each word in the input sequence we need to know how relevant every other word in the input sequence is to it.
4. Residual Connections between 2 and 3: Because deep models forget earlier layers otherwise.

There are 5 main steps in the decoder:

1. Word Embedding: Because numbers not strings.
2. Positional Encoding: Because order matters and transformers have no inherent sense of order like a recurrent model.
3. Self-Attention: Because for each word in the output sequence we need to know how relevant every other word in the output sequence is to it.
4. Encoder-Decoder Attention: Because the output words need to know how relevant each input word is to it.
5. Residual Connections between 2 and 3, and between 3 and 4: Because deep models forget earlier layers otherwise.

Bunch of normalization stuff in between.

A decoder-only transformer, like GPT, uses Masked Self-Attention instead of regular Self-Attention in step 3 of the decoder because we want each word to attend to only the previous words, not future words. Hence, they are referred to as auto-regressive models.
