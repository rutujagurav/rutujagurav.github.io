---
layout: post
title: what's going on with sktime?
date: 2024-04-15 15:00:00
description: the saga of sktime and aeon
tags: uff musings sktime aeon
categories: oh-what-a-mess
---

It is super late at night and I have fallen into an internet rabbit hole of the timeseries machine learning community. I've been working with a lot of time domain data for a while now and of the several good python libraries available, `sktime` has been one that I haven't quite used too much. My go-to ones have been - 1. `tsfresh` for the feature extraction capabilities, 2. `tslearn` for standard machine learning algorithms, 3. `stumpy` for all things matrix profile and 4. `tsai` for deep learning based algorithms.

I have been playing around with a short project idea of examining clustering perfromance of all available timeseries datasets in the mighty UCR/UEA timeseries classification archive using various timeseries feature-sets like the ones available in `tsfresh` and the increasingly popular Catch-22 features available in `pycatch22` package[^1], so imagine my delight when I went to [timeseriesclassification.com](https://timeseriesclassification.com) to fetch all the datasets and read _"The scikit-learn compatible aeon toolkit contains the state of the art algorithms for time series classification. All of the datasets and results stored here are directly accessible in code using aeon."_ I thought, "Awesome, eveything I need in one place!". So I go check out `aeon` and it is fantastic. But my brain must have done a random access of some forgotten recess of my mind because I found myself thinking, "Huh, this looked familiar. Almost like `sktime`...". And indeed that's when I fell into the current rabbit hole from whence I write this post.

Turns out there is some drama-llama stuff around this whole `sktime` vs. `aeon` saga. Turns out [`aeon` is a fork of sktime](https://news.ycombinator.com/item?id=36432369) created by Tony Bagnall of UEA who departed(?) from `sktime` after [another core developer](https://github.com/sktime/community-org/issues/45) allegedly took over the project and kicked others out. The whole thing sounds like a bit of a mess. Anyhoo, I have no horse in the race. Preliminary examination suggests either library is fine for my purposes. I am going to go with `aeon` for the aforementioned short project. And this Alice needs to climb out this hole and go to bed now.

[^1]: I know, I know, some of the datasets available in the UCR/UEA archive are not amenable to features based classification perhaps and are more separable in terms of shapes but I digress.