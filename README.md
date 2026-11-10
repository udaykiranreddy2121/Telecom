# Telecom Customer Churn Analysis

An exploratory data analysis project using a telecom customer dataset to understand customer usage patterns and investigate factors associated with customer churn.

## Project objective

Explore customer account, calling, plan, and service-interaction data to identify patterns that may help a telecom provider understand and reduce customer churn.

## Dataset

The notebook references a file named `Telecom Churn.csv`. The dataset is **not currently included in this repository**, so download or obtain the dataset separately and place it in the project folder before running the notebook.

The notebook output shows a dataset with 3,333 records and 21 columns, including:
- Customer account information: state, account length, area code
- Plan information: international plan and voice-mail plan
- Usage: day, evening, night, and international calls/minutes/charges
- Service interactions: customer service calls
- Target: churn

The notebook includes customer phone numbers in the source dataset/output. Avoid republishing raw personal or customer-identifying fields; remove or mask phone numbers before sharing datasets or screenshots.

## Repository contents

```text
Telecom/
├── Exploratory_Data_Analysis.ipynb
├── Visual_Data_Analysis.ipynb
└── README.md
```

## Tools and libraries

- Python
- pandas
- NumPy
- Jupyter Notebook / Google Colab
- Matplotlib / Seaborn (confirm exact plotting imports in the visualization notebook before running)

## Setup

Use Python 3.9 or later.

```bash
git clone https://github.com/udaykiranreddy2121/Telecom.git
cd Telecom
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the core analysis dependencies:

```bash
python -m pip install pandas numpy jupyter matplotlib seaborn scikit-learn
```

Place `Telecom Churn.csv` in the project directory. The current EDA notebook references a Google Colab path (`/content/Telecom Churn.csv`); update the data-loading cell to use a local relative path, for example:

```python
import pandas as pd

data = pd.read_csv("Telecom Churn.csv")
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `Exploratory_Data_Analysis.ipynb` first, followed by `Visual_Data_Analysis.ipynb`.

## Analysis workflow

The exploratory notebook currently:
1. Loads the telecom churn CSV into a pandas DataFrame.
2. Inspects sample rows, shape, columns, and data types.
3. Examines customer plan, usage, billing, and service-call variables.
4. Converts the churn target from boolean to an integer representation.
5. Explores the dataset to support subsequent visual analysis.

The repository also contains a separate visualization notebook. Review its charts and conclusions directly in that notebook.

## Important distinction

The current repository evidence supports describing this as a **customer churn exploratory and visual analysis project**. Do not describe it as a completed predictive machine-learning model or claim accuracy, precision, recall, or business impact unless a reproducible training/evaluation workflow and its measured results are added.

## Recommended next steps

- Add a sanitized dataset source/link and document its license.
- Replace hard-coded Colab file paths with configurable relative paths.
- Remove or mask phone-number identifiers from outputs before publishing.
- Add a `requirements.txt` file.
- Add a separate modelling notebook with a stratified train/test split, baseline model, appropriate metrics, and leakage checks if predictive modelling is part of the intended project.
- Add clear chart captions and evidence-based findings.

## License

No license file was present when this README was prepared. Add a license only after choosing the terms under which you want others to use the project.
