# Schizophrenia Prediction Model (Logistic Regression)

## Overview

This is a conceptual machine learning project that explores the use of logistic regression to classify schizophrenia diagnoses based on a set of simplified psychiatric and cognitive parameters. The dataset used is entirely synthetic and was generated with the help of ChatGPT for educational purposes only.

The goal of this project was to understand the process of building a basic ML pipeline in Python, including data generation, preprocessing, feature engineering, model training, and evaluation. The high accuracy achieved is a reflection of the deterministic rules used to assign class labels in the synthetic dataset and should not be interpreted as indicative of real-world diagnostic performance.

## Features Used

* PANSS positive score
* PANSS negative score
* PANSS general psychopathology score
* Working memory score
* Reaction time
* Age
* Gender (binary encoded)

## Dataset

The dataset was not collected from real patients. It was generated synthetically using randomized values within plausible clinical ranges. Label assignment (diagnosis of schizophrenia) was based on arbitrary threshold rules applied to these synthetic features. The feature values and rules were adapted and implemented with the assistance of ChatGPT.

## Model

* Type: Logistic Regression
* Library: scikit-learn
* Preprocessing: StandardScaler for feature normalization

## Results

The model achieved an accuracy of approximately 90.6% on the synthetic test set. This result is expected due to the structured and noise-free nature of the dataset. It does not reflect real-world performance and should not be considered predictive in any clinical context.

## Disclaimer

This project is for educational purposes only. The dataset is synthetic, and the model is not intended for any clinical or diagnostic use. Interpretations drawn from this project are limited to the simulated context and do not represent actual psychiatric data or outcomes.

## Future Work

* Incorporating noise and variability to better reflect real-world complexity
* Experimenting with other classifiers (e.g., SVM, decision trees)
* Developing synthetic datasets with probabilistic label generation

## Author

This project was developed as a learning exercise by an undergraduate student exploring the intersection of machine learning and computational psychiatry.&#x20;
