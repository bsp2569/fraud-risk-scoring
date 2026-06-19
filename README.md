# Fraud Risk Scoring & Transaction Anomaly Detection

A resume-ready Data Science project for fraud analytics, payment risk scoring, transaction anomaly detection, and dashboarding.

## What this project does

This project builds an end-to-end fraud risk analytics system:

1. Generates a realistic synthetic transaction dataset.
2. Engineers fraud-risk features such as transaction velocity, location mismatch, failed transaction ratio, merchant category risk, and time-of-day risk.
3. Trains supervised ML models to classify fraud risk.
4. Trains an unsupervised anomaly detection model using Isolation Forest.
5. Produces model evaluation metrics.
6. Launches a Streamlit dashboard for fraud monitoring and transaction-level risk scoring.

## Tech Stack

Python, SQL-ready Pandas transformations, Scikit-learn, Random Forest, Gradient Boosting, Isolation Forest, Streamlit, Plotly, Joblib.

## Project Structure

```text
fraud_risk_scoring_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── synthetic_transactions.csv
│
├── models/
│   ├── fraud_classifier.joblib
│   ├── anomaly_detector.joblib
│   └── feature_columns.json
│
├── reports/
│   └── model_metrics.json
│
└── src/
    ├── generate_data.py
    ├── feature_engineering.py
    └── train_model.py
```

## How to Run

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate dataset

```bash
python src/generate_data.py
```

### 4. Train models

```bash
python src/train_model.py
```

### 5. Run dashboard

```bash
streamlit run app.py
```

## Resume Bullet Points

You can add this to your resume:

**Fraud Risk Scoring & Transaction Anomaly Detection**  
**Tech Stack:** Python, SQL, Pandas, Scikit-learn, Random Forest, Gradient Boosting, Isolation Forest, Streamlit

- Built an end-to-end fraud risk analytics solution to detect suspicious transactions using customer, merchant, payment, location, and transaction behavior data.
- Engineered risk features including transaction velocity, amount deviation, failed transaction ratio, merchant category risk, location mismatch, and time-of-day spending behavior.
- Applied Random Forest, Gradient Boosting, and Isolation Forest models to classify fraud risk and detect anomalous transaction patterns.
- Built a Streamlit dashboard to monitor fraud trends, high-risk merchants, transaction anomalies, model risk scores, and investigation priorities.

## Suggested GitHub README Additions

Add screenshots from your Streamlit app:
- Fraud trend chart
- Risk score distribution
- High-risk transaction table
- Merchant category risk chart
- Confusion matrix / model metrics
