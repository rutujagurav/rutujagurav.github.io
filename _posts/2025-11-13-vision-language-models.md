---
layout: post
title: Vision Language Models (VLMs)
date: 2025-11-13 00:00:00
description: 
tags: transformers vision-language-models
categories: cliff-notes
---

# Vision Language Models (VLMs)

Popular Models: Most popular LLMs have a VLM counterpart - Gemini 2.0 Flash, GPT-4o, LLaVA, Qwen 2.5-VL, etc.

The 4 key approaches to VLMs are -

1. Contrastive Learning: Push embedings of matched image-text pairs closer and non-matched pairs farther apart.
2. Masking: Mask a word in the text and provide a unmasked image as context to predict the masked word.
3. Generative Modeling: Text-to-image (Imagen, Midjourney, DALL-E, SORA), image-to-text.
4. Pretrained Models: A pretrained LLM and a pretrained vision encoder.

# References
[1] https://www.ibm.com/think/topics/vision-language-models