# XGB-ToxPredict

<p align="center">

  <img src="doc/Fig.1.png" alt="XGB-ToxPredict Architecture" width="900">

</p>

XGB-ToxPredict is an XGBoost-based machine learning pipeline for predicting treatment-related toxicity in patients with hepatocellular carcinoma (HCC). The framework adopts a hierarchical two-stage classification strategy that first identifies patients at risk of developing any toxicity and then stratifies those patients according to toxicity severity.
The pipeline consists of two sequential models:
* Model 1 (M1): Any Toxicity Prediction – Predicts whether a patient will experience treatment-related toxicity (Grade > 0) using the complete patient cohort.
* Model 2 (M2): Severe Toxicity Prediction – Applied only to patients predicted to have toxicity by M1. This model is trained on patients with Grades 1–5 and predicts severe toxicity (Grade ≥3), distinguishing mild/moderate toxicity from severe cases.

This cascaded design decomposes toxicity prediction into two clinically meaningful binary classification tasks, improving interpretability while allowing independent optimization of each prediction stage.


# Data Pre-processing

Before training the machine learning models, the clinical dataset must be cleaned and pre-processed.

## Step 1: Prepare the Dataset

An example input template is provided in:

```text
Pre-process/dataset/example.xlsx
```

The dataset should follow this structure:

| Column | Description |
|--------|-------------|
| First column | Patient ID |
| Middle columns | Clinical features |
| Last column | Target variable |

 The target should be encoded as:

| Value | Meaning |
|------:|---------|
| 0 | No toxicity (Grade 0) |
| 1 | Toxicity (Grades 1–5) |

Use the provided template as a guide and replace the example data with your own dataset while preserving the same format.

---

## Step 2: Configure the Pre-processing Pipeline

Open the configuration file:

```text
Pre-process/config/config_pre.yaml
```

Each configuration parameter is documented with comments. Carefully review and modify the settings according to your dataset, including:

- Input dataset path
- Patient ID column
- Target column
- Feature types (categorical/numerical)
- Missing value handling
- Imputation method
- Feature selection options
- Output directory

---

## Step 3: Run Pre-processing

Navigate to the preprocessing directory:

```bash
cd Pre-process
```

Run the preprocessing pipeline:

```bash
python pre_process.py --config_path config/config_pre.yaml
```

---

## Step 4: Review the Results

All preprocessing outputs, exploratory analyses, and visualizations are saved in:

```text
Data_explatory/
```

The generated reports include:

- Missing data analysis
- Feature distributions
- Target distribution
- Correlation analysis
- Data summaries
- Additional exploratory visualizations

These reports provide a comprehensive overview of the dataset before model development.

---

## Step 5: Use the Processed Dataset

After preprocessing is complete, the final cleaned and imputed dataset is generated automatically.

For example:

```text
patient_imputed_final_MissForest.xlsx
```

This file contains the processed dataset and should be used as the input for training and evaluating the XGB-ToxPredict models.




## Pipeline Overview

### Model 1 (M1): Any Toxicity Prediction

-   Negative class: **Grade = 0**
-   Positive class: **Grade \> 0**

### Model 2 (M2): Severe Toxicity Prediction

-   Negative class: **Grade < 3**
-   Positive class: **Grade ≥ 3**

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
python pre_process.py --config_path config/config_pre.yaml
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

