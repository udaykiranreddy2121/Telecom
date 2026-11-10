# Telecom Customer Churn Analysis

An exploratory data analysis and machine-learning project that studies customer churn patterns and trains reproducible classification models to identify customers who may leave a telecom service.

## Dataset

The notebooks and training script expect a CSV named `Telecom Churn.csv`. The dataset used in the original analysis contains 3,333 records and 21 columns, including customer account details, service plans, usage, support calls and the `churn` target.

The CSV is not currently stored in this repository. Obtain it from your authorised course or dataset source and place it in the project root. Do not commit data that you do not have permission to redistribute.

## Project contents

- `Exploratory_Data_Analysis.ipynb` — initial data inspection and exploration.
- `Visual_Data_Analysis.ipynb` — visual exploration of churn-related patterns.
- `train_model.py` — reproducible model training and evaluation pipeline.
- `requirements.txt` — Python dependencies.

## Setup

```bash
git clone https://github.com/udaykiranreddy2121/Telecom.git
cd Telecom
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate models

Place `Telecom Churn.csv` in the repository root, then run:

```bash
python train_model.py
```

Or specify another path:

```bash
python train_model.py --data "path/to/Telecom Churn.csv"
```

The pipeline excludes the phone-number identifier, creates a stratified 80/20 train/test split, fits preprocessing on training data only, and compares a majority-class baseline, balanced logistic regression and balanced random forest. It reports accuracy, precision, recall, F1, ROC-AUC, average precision and a confusion matrix. The best non-baseline model by churn-class F1 is saved.

Outputs are written to `artifacts/`:
- `telecom_churn_model.joblib`
- `metrics.json`

## Results from the supplied dataset

The supplied 3,333-row dataset was evaluated using a stratified 80/20 split (random state 42). Random Forest achieved **93.55% accuracy**, **83.75% precision**, **69.07% recall**, **0.7571 F1**, **0.8916 ROC-AUC**, and **0.7826 average precision** for churn. Its confusion matrix was TN=557, FP=13, FN=30, TP=67. Logistic Regression achieved 0.4525 churn-class F1; the majority baseline detected no churn cases despite 85.46% accuracy.

A separate five-fold stratified cross-validation of Random Forest produced mean F1 **0.8008** (std 0.0221), mean ROC-AUC **0.9112** (std 0.0253), and mean average precision **0.8541** (std 0.0278). Full rounded results are in [metrics.json](metrics.json).

## Evaluation and responsible interpretation

Churn datasets can be imbalanced, so accuracy alone is not sufficient. Review churn-class precision and recall, F1, ROC-AUC and average precision together. Holdout and cross-validation metrics are estimates on this dataset, not guarantees of future performance. Validate on newer customer data before operational use. Model predictions should support, not automatically determine, customer-retention actions.

## Current limitations

- The dataset is not included; users must supply an authorised copy.
- The training script uses a single stratified holdout split, not a full hyperparameter search.
- The original notebooks may contain Colab-specific file paths and saved outputs; use the training script for the reproducible workflow.

## Future improvements

- Add cross-validation and threshold tuning based on retention-team capacity.
- Add feature importance or model explainability with SHAP.
- Add a small inference API or dashboard after validating the model.
- Add tests for target normalisation and the training pipeline.

## License

No license is specified yet. Add a license file if you want others to reuse the code.
