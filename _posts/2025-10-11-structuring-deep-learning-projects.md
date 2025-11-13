---
layout: post
title: Guts of the Transformer
date: 2025-10-30 00:00:00
description: 
tags: 
categories: my-little-helpers
---


At some point a deep learning project leaps out of the single Jupyter notebook and into a more structured codebase on its way to deployment, either standalone or as part of a larger system. Over the years, through trial, error and sleuthing on github, I have settled on a basic, cookie-cutter structure that works pretty well for me for most low TRL academic projects. Here is a quick overview of the structure along with a brief description of each folder/file.

## The Structure

``` plain
my_deep_learning_project/
│├── src/
│   ├── __init__.py
│   ├── dataloading/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py
│   │   ├── proposed_models.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── train.py
|   |   ├── metrics.py
|   |   ├── evaluate.py
|   |── notebooks/
│   |   ├── EDA.ipynb
│   |   ├── ResultsReview.ipynb
|   ├── utils.py
│├── results/
│├── data/
|├── stdout/
│├── requirements.txt
│├── .gitignore
│├── README.md
```

## The Breakdown

- `src/`: This is the main source code directory containing all the code for data loading, model definitions, experiments, and utilities.
  - `dataloading/`: Contains code related to datasets and preprocessing. The `dataset.py` file typically defines custom dataset classes. A benchmarking suite can have multiple datasets each of which can be defined in dedicated `<name>_dataset.py` files.
  - `models/`: Contains implementations of the models. `baselines.py` includes standard models for comparison, while `proposed_models.py` contains the implementation of the new model(s) being proposed. Alternatively, each model can be defined in its own file viz. `<model_name>.py`.
  - `training/`: Contains scripts for training and evaluating models. `train.py` handles the training loop, `metrics.py` defines evaluation metrics.
  - `evaluation/`: Contains `evaluate.py`  for evaluating trained models on test datasets.
  - `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA), demos, reviewing results, etc.
  - `utils.py`: A utility module for helper functions used across the project.
- `results/`: Directory to store model outputs, logs, and visualizations.
- `data/`: Directory for raw and processed datasets.
- `stdout/`: Directory to store standard output logs from training and evaluation runs.
- `requirements.txt`: Lists the Python dependencies for the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## The Example

Check out this template at my repo [cookiecutterdl4r](https://github.com/rutujagurav/cookiecutterdl4r).
