# XGB-ToxPredict

<p align="center">

  <img src="doc/Fig.1.png" alt="XGB-ToxPredict Architecture" width="900">

</p>

XGB-ToxPredict is an XGBoost-based machine learning pipeline for predicting treatment-related toxicity in patients with hepatocellular carcinoma (HCC). The framework adopts a hierarchical two-stage classification strategy that first identifies patients at risk of developing any toxicity and then stratifies those patients according to toxicity severity.
The pipeline consists of two sequential models:
* Model 1 (M1): Any Toxicity Prediction – Predicts whether a patient will experience treatment-related toxicity (Grade > 0) using the complete patient cohort.
* Model 2 (M2): Severe Toxicity Prediction – Applied only to patients predicted to have toxicity by M1. This model is trained on patients with Grades 1–5 and predicts severe toxicity (Grade ≥3), distinguishing mild/moderate toxicity from severe cases.

This cascaded design decomposes toxicity prediction into two clinically meaningful binary classification tasks, improving interpretability while allowing independent optimization of each prediction stage.
