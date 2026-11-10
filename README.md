# Telecom Customer Churn Analysis

An exploratory data analysis project using a telecom customer dataset to understand customer churn patterns and identify factors associated with customers leaving a service.

## Dataset

The notebook uses a telecom churn CSV with 3,333 customer records and 21 columns, including:

- Customer profile: state, account length, area code
- Service choices: international plan, voice-mail plan
- Usage: day, evening, night and international minutes/calls/charges
- Support interactions: customer service calls
- Target: churn

**Dataset note:** The notebook currently loads `Telecom Churn.csv` from a Google Colab path. That dataset file is not included in this repository listing. Download the dataset from your authorised source and place it in the project folder, or update the notebook's data path. Do not commit private or licensed data.

## Project workflow

1. Load the dataset with pandas.
2. Inspect dimensions, columns, data types and missing values.
3. Explore churn distribution and customer usage patterns.
4. Visualise relationships between service usage, plans, support calls and churn.
5. Use the findings to inform a future churn-prediction model.

## Tech stack

Python, Jupyter Notebook, pandas, NumPy, data visualisation.

## Run locally

```bash
git clone https://github.com/udaykiranreddy2121/Telecom.git
cd Telecom
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

Open `Exploratory_Data_Analysis.ipynb` or `Visual_Data_Analysis.ipynb`.

## Repository contents

- `Exploratory_Data_Analysis.ipynb` — initial data exploration and inspection.
- `Visual_Data_Analysis.ipynb` — visual exploration of churn-related patterns.

## Important interpretation note

Exploratory relationships do not establish causation. Any predictive model should be evaluated with a stratified train/test split and metrics appropriate for class imbalance, such as precision, recall, F1-score and ROC-AUC. Do not report model accuracy or business impact unless it has been measured and documented.

## Next steps

- Make notebook paths relative instead of using Colab-only paths.
- Remove personally identifying fields such as phone number from modelling features.
- Add a reproducible modelling notebook with preprocessing, a baseline and cross-validation.
- Document measured results and limitations.

## License

No license is specified yet. Add a license file if you want others to reuse the code.
