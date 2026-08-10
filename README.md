# XGB-ToxPredict

<p align="center">

  <img src="doc/Fig.1.png" alt="XGB-ToxPredict Architecture" width="900">

</p>

XGB-ToxPredict is an XGBoost-based machine learning pipeline for predicting treatment-related toxicity in patients with hepatocellular carcinoma (HCC). The framework adopts a hierarchical two-stage classification strategy that first identifies patients at risk of developing any toxicity and then stratifies those patients according to toxicity severity.
The pipeline consists of two sequential models:
* Model 1 (M1): Any Toxicity Prediction – Predicts whether a patient will experience treatment-related toxicity (Grade > 0) using the complete patient cohort.
* Model 2 (M2): Severe Toxicity Prediction – Applied only to patients predicted to have toxicity by M1. This model is trained on patients with Grades 1–5 and predicts severe toxicity (Grade ≥3), distinguishing mild/moderate toxicity from severe cases.

This cascaded design decomposes toxicity prediction into two clinically meaningful binary classification tasks, improving interpretability while allowing independent optimization of each prediction stage.

## Pipeline Overview

### Model 1 (M1): Any Toxicity Prediction

-   Negative class: **Grade = 0**
-   Positive class: **Grade \> 0**

### Model 2 (M2): Severe Toxicity Prediction

-   Negative class: **Grade 1--2**
-   Positive class: **Grade ≥ 3**

## Features

-   XGBoost-based classification
-   Hierarchical two-stage prediction
-   Automated preprocessing
-   Configurable feature selection
-   Cross-validation
-   Probability calibration
-   Automatic threshold optimization
-   ROC, PR and calibration curves
-   Confusion matrix
-   SHAP explainability
-   YAML configuration
-   Reproducible experiments

## Installation

### Clone

``` bash
git clone https://github.com/<username>/XGB-ToxPredict.git
cd XGB-ToxPredict
```

### Create environment

``` bash
conda create -n xgb-toxpredict python=3.9 -y
conda activate xgb-toxpredict
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preprocessing

``` bash
cd Pre-process
python pre_process.py
```

or

``` bash
python pre_process.py --config_path config/config.yaml
```

## Training

``` bash
python train.py --config_path configs/m1.yaml
python train.py --config_path configs/m2.yaml
```

## Testing

``` bash
python test.py --config_path configs/m1.yaml
python test.py --config_path configs/m2.yaml
```

## Output

The pipeline generates:

-   ROC curve
-   Precision--Recall curve
-   Calibration curve
-   Confusion matrix
-   Excel report with predictions and metrics
-   Feature importance
-   SHAP plots

## License

MIT License.

## Citation

Citation information will be added after publication.

## Contact

**Falah Jabar**

University Hospital of North Norway (UNN)

