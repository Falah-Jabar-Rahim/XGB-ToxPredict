# XGB-ToxPredict

<p align="center">

  <img src="doc/Fig.1.png" alt="XGB-ToxPredict Architecture" width="900">

</p>

XGB-ToxPredict is an XGBoost-based machine learning pipeline for predicting treatment-related toxicity in patients with hepatocellular carcinoma (HCC). The framework adopts a hierarchical two-stage classification strategy that first identifies patients at risk of developing any toxicity and then stratifies those patients according to toxicity severity.
The pipeline consists of two sequential models:
* Model 1 (M1): Any Toxicity Prediction – Predicts whether a patient will experience treatment-related toxicity (Grade > 0) using the complete patient cohort.
* Model 2 (M2): Severe Toxicity Prediction – Applied only to patients predicted to have toxicity by M1. This model is trained on patients with Grades 1–5 and predicts severe toxicity (Grade ≥3), distinguishing mild/moderate toxicity from severe cases.

This cascaded design decomposes toxicity prediction into two clinically meaningful binary classification tasks, improving interpretability while allowing independent optimization of each prediction stage.



### Installation

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

# Preparing the Model Datasets

After preprocessing, two datasets should be created for the hierarchical prediction pipeline.

## M1: Any Toxicity Prediction

Model 1 is trained and tested using **all patients**.

The target variable should be defined as:

| Toxicity Grade | M1 Target |
|---------------:|:---------:|
| Grade 0 | 0 |
| Grades 1–5 | 1 |

Thus, Model 1 learns to predict whether a patient will experience **any treatment-related toxicity**.

The resulting dataset should contain:

- Patient ID
- Selected clinical features
- Binary target (`Grade≥3`)

Example:
```text
M1/dataset/train.xlsx
M1/dataset/test.xlsx
```
---

## M2: Severe Toxicity Prediction

Model 2 is trained and tested **only on patients who experienced toxicity** (Grades 1–5).

Patients with **Grade 0** are excluded.

The target variable is then redefined as:

| Toxicity Grade | M2 Target |
|---------------:|:---------:|
| Grades 1–2 | 0 |
| Grades 3–5 | 1 |

Thus, Model 2 learns to distinguish between:

- Mild/Moderate toxicity (Grades 1–2)
- Severe toxicity (Grades 3–5)

The resulting dataset should contain:

- Patient ID
- Selected clinical features
- Binary target (`Grade≥3`)

Example:
```text
M2/dataset/train.xlsx
M2/dataset/test.xlsx
```
---

## Dataset Summary

| Model | Included Patients | Target |
|--------|-------------------|--------|
| **M1** | All patients | Grade 0 vs Grades 1–5 |
| **M2** | Only patients with toxicity (Grades 1–5) | Grades 1–2 vs Grades 3–5 |

The same preprocessing pipeline should be applied to both datasets to ensure consistent feature engineering and data quality.

# Usage

After preprocessing and preparing the datasets for **M1** and **M2**, the models can be trained and evaluated independently or used together in the hierarchical prediction pipeline.

## Train M1 (Any Toxicity Prediction)

```bash
python -m stages.train --config configs/m1.yaml
```

This trains the M1 model to predict **whether a patient will develop treatment-related toxicity (Grade > 0)**.

---

## Train M2 (Severe Toxicity Prediction)

```bash
python -m stages.train --config configs/m2.yaml
```

This trains the M2 model using only patients with toxicity (Grades 1–5) to predict **severe toxicity (Grade ≥ 3)**.

---

## Evaluate M1

```bash
python -m stages.test --config configs/m1.yaml
```

Evaluates the trained M1 model on an independent test dataset and generates:

- ROC curve
- Precision–Recall curve
- Calibration curve
- Confusion matrix
- Classification metrics
- Feature importance and SHAP visualizations
- Patient-level predictions

---

## Evaluate M2

```bash
python -m stages.test --config configs/m2.yaml
```

Evaluates the trained M2 model using the corresponding independent test dataset and generates the same evaluation reports.

---
## Run the Complete Hierarchical Pipeline

```bash
python -m hierarchical.predict --config configs/hierarchical.yaml
```

The hierarchical pipeline uses a single input dataset containing **all patients**. The workflow then proceeds automatically:

1. **Model 1 (M1)** predicts whether each patient is likely to develop treatment-related toxicity.
2. Patients predicted as **No Toxicity** are assigned the final outcome **Grade 0**.
3. Patients predicted as **Toxicity** are forwarded to **Model 2 (M2)**.
4. **Model 2 (M2)** predicts whether toxicity is:
   - **Grade 1–2 (Mild/Moderate Toxicity)**, or
   - **Grade 3–5 (Severe Toxicity)**.

The final output provides a hierarchical prediction for every patient in the input dataset.

### Workflow

```text
                All Patients
                     │
                     ▼
        ┌────────────────────────┐
        │ Model 1                │
        │ Any Toxicity Prediction│
        └────────────────────────┘
             │              │
             │              │
      No Toxicity      Toxicity
       (Grade 0)            │
             │              ▼
             │      ┌────────────────────────┐
             │      │ Model 2                │
             │      │ Severe Toxicity        │
             │      └────────────────────────┘
             │            │
             │            │
             ▼            ▼
         Grade 0     Grade 1–2 or Grade 3–5
```

**Note:** Unlike Model 2 training, which uses only patients with toxicity (Grades 1–5), the hierarchical prediction pipeline always accepts a dataset containing **all patients**, allowing the two models to operate sequentially as they would in clinical practice.



## License

MIT License.

## Citation

Citation information will be added after publication.

## Contact

**Falah Jabar**

University Hospital of North Norway (UNN)

