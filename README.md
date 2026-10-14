# AI Fraud Risk Scoring Engine for Financial Institutions

**Vaal University of Technology** | Diploma: Information Technology
**Subject:** Business Analysis 3.2 (AIBUY3A) | **Theme:** An AI Solution for Industries | **Industry:** Financial Services

An end-to-end Python prototype that scores card transactions for fraud risk. It combines a rule-based baseline, machine learning, deep learning, time-series analysis, NLP and a fraud-investigation chatbot, and presents the results in an interactive dashboard. It is a decision-support tool: it ranks transactions for human review and does not make final fraud decisions.

## Team

| Member | Student number | Role |
|---|---|---|
| OE Mukhwakhwa | 224463845 | Group Leader / Business Analyst |
| L Manhlangu | 224572865 | Data Scientist |
| KA Ramanya | 223189227 | Machine Learning Engineer |
| AM Murulana | 224633546 | Research and Advanced Analytics |
| NM Mahanuki | 224509373 | Application Developer |

## Dataset

`data/transactions.csv` has 88,192 simulated card transactions (100 customers, year 2025) with a real `is_fraud` label (917 fraud cases, about 1%). It was generated with the open-source [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) tool (MIT licence). Names, SSNs, card numbers and street addresses were removed.

The data is simulated, so results show how the method behaves, not how it would perform on a real bank's data. See `docs/DATASET.md` for the column list and how to regenerate it.

## Quick start

Run these from the project root folder:

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

- `python main.py` runs every stage and prints the model results.
- `streamlit run dashboard/app.py` opens the dashboard.
- Optional: use a virtual environment first (`python -m venv venv`, then `venv\Scripts\activate` on Windows or `source venv/bin/activate` on Mac/Linux).

## Project structure

```
├── data/transactions.csv          Labelled transaction dataset
├── src/
│   ├── data_preprocessing.py      Load, validate and clean the data
│   ├── fraud_labeling.py          Risk indicators and rule-engine baseline
│   ├── feature_engineering.py     Feature matrix for the models
│   ├── eda.py                     Exploratory data analysis and charts
│   ├── time_series_analysis.py    Daily trends, rolling average, forecast, hourly fraud rate
│   ├── nlp_module.py              Transaction narratives, TF-IDF, keyword search
│   ├── ml_models.py               Logistic Regression, Random Forest, Isolation Forest
│   ├── deep_learning_model.py     ANN classifier and autoencoder
│   ├── pipeline.py                Builds and caches the scored transaction table
│   └── chatbot.py                 Fraud investigation chatbot
├── dashboard/app.py               Streamlit dashboard
├── docs/                          Project Report
├── main.py                        Runs the full pipeline
└── requirements.txt
```


## What each part does

| Part | Purpose |
|---|---|
| Data preprocessing | Cleans the data and adds age and distance from home |
| Risk indicators | Explainable rules (large amount, night-time, new merchant and so on), used as a baseline |
| Machine learning | Compares Logistic Regression, Random Forest and Isolation Forest with 5-fold cross-validation |
| Deep learning | An ANN classifier and an autoencoder that learns normal behaviour without fraud labels |
| Time series | Fraud and spending patterns over days and hours, plus a simple 7-day forecast |
| NLP | Turns each transaction into a plain-language description that can be searched |
| Chatbot | Answers questions such as `summary`, `high risk`, `riskiest hour`, `account AC0042` |
| Dashboard | KPIs, charts, transaction explorer, chatbot and model information |


## Running one module on its own

```bash
python -m src.data_preprocessing
python -m src.fraud_labeling
python -m src.feature_engineering
python -m src.eda
python -m src.time_series_analysis
python -m src.nlp_module
python -m src.ml_models
python -m src.deep_learning_model
python -m src.chatbot
```

