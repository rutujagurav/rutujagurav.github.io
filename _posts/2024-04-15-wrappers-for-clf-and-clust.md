---
layout: post
title: convenience wrappers
date: 2024-04-15 15:00:00
description: wrappers for sklearn et al for classification and clustering 
tags: classification scikit-learn wrappers
categories: my-little-helpers
---
# Getting Started
Every time I start a new machine learning project, I find myself going through the same tedious process of trial and error of setting up a grid search to find the _right_ model along with the _right_ set of hyperparameters for the model that optimize one or more of the laundry list of _metrics-of-interest_ and repeating every combination of _free parameters_ in this pipeline a bunch of times and finally making a lot of plots to get the lay of the land so to speak. So, over the years, I have developed a set of convenience wrappers around the mighty `scikit-learn` library that I use to make this process a bit more streamlined and published them as `clfutils4r` for classification tasks and `clustutils4r` for clustering tasks (the 'r' being my initial not the language...err, should have thought this through, huh?). I thought I would share them here in case they are useful to anyone else. 

## Classification
The premise is this: Someone hands you a classification dataset. You want to know the standard metrics on various classifiers available in `scikit-learn` and you want to know them _now_. You don't want to spend time writing boilerplate code setting up a grid search and you don't want to spend time making plots. Here is minimally complete example of how you can do it with essentially 2 function calls:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

## Load dataset: Example - breast cancer prediction
data = datasets.load_breast_cancer()
class_names = [str(x) for x in data.target_names]
feature_names = [str(x) for x in data.feature_names]
X, y = data.data, data.target

## Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

## Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

## Grid search for best model
from clfutils4r.gridsearch_classification import gridsearch_classification
save_dir = "gridsearch_results"
os.makedirs(save_dir, exist_ok=True)
best_model, grid_search_results = gridsearch_classification(X=X_train,                    # training dataset
                                                            gt_labels=y_train,            # ground truth labels
                                                            best_model_metric="F1",       # metric to use to choose the best model
                                                            show=True,                    # whether to display the plots; this is used in a notebook
                                                            save=True, save_dir=save_dir  # whether to save the plots
                                                        )

## Predict on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

## Evaluate best model on test set
from clfutils4r.eval_classification import eval_classification
eval_classification(y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba,  # ground truth labels, predicted labels, predicted probabilities
                    class_names=class_names, feature_names=feature_names,
                    titlestr="Breast Cancer Classification",
                    make_metrics_plots=True, # make a variety of classification metrics plots
                    make_shap_plot=True, shap_nsamples=100, # do Shapley analysis for model explainability
                    show=True,  
                    save=True, RESULTS_DIR=os.getcwd()+'/test_results')

```

Let's dive a bit deeper into these two convenience functions.
 
### `gridsearch_classification`

This will produce a whole bunch of useful outputs including the best model which you can use as you choose downstream and the results of the grid search. The data is stored in neat folder structure in JSON files and is visualized with a _Parallel Co-ordinates Plot_. I find this type of plot very useful to get a quick view of the grid search.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/parcoord_plot.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The parallel co-ordinates plot that is produced by `gridsearch_classification` function for the _K Nearest Neighbors_ classifier for the example above.
</div>

### `eval_classification`

The primary output is the classic sklearn _classification report_. Sometimes that's all you need but by setting the `make_metrics_plots` to `True` you can choose to make a variety of other plots that I find useful for understanding the performance of the model. These include the familiar plots of the confusion matrix, the ROC curve, the precision-recall curve as well as some more exotic ones I found in `scikit-plot` that are exclusive to binary classification like the KS statistic plot, the cumulative gains curve and the lift curve. You can also choose to do Shapley analysis to _explain_ the model predictions by setting the `make_shap_plot` parameter to `True` and specifying the number of samples to use for the analysis with the `shap_nsamples` parameter. I love the fantastic `shap` package so I just wrapped the _KernelExplainer_ and _summary_plot_ in this function.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/confusion_matrix.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/classwise_roc_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/classwise_pr_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The classic evaluation plots produced by eval_classification() on the test set for the best model returned by gridsearch_classification() for the example above.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ks_stat.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/cumul_gain.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/lift_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Some more exotic evaluation plots exclusive to binary classification produced by eval_classification() on the test set for the best model returned by gridsearch_classification() for the example above.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/shap_summary_plot.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The Shapley analysis summary plot produced by shap package for the best model returned by gridsearch_classification() for the example above.
</div>