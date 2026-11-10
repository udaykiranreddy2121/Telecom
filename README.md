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

## Evaluation and responsible interpretation

Churn datasets can be imbalanced, so accuracy alone is not sufficient. Review churn-class precision and recall, F1, ROC-AUC and average precision together. Holdout metrics are estimates on one split; use cross-validation and an untouched future test set before making business decisions. Model predictions should support, not automatically determine, customer-retention actions.

## Current limitations

- The dataset is not included; users must supply an authorised copy.
- The training script uses a single stratified holdout split, not a full hyperparameter search.
- No measured model performance is claimed here. Run the script on the dataset to generate actual metrics.
- The original notebooks may contain Colab-specific file paths and saved outputs; use the training script for the reproducible workflow.

## Future improvements

- Add cross-validation and threshold tuning based on retention-team capacity.
- Add feature importance or model explainability with SHAP.
- Add a small inference API or dashboard after validating the model.
- Add tests for target normalisation and the training pipeline.

## License

No license is specified yet. Add a license file if you want others to reuse the code.
