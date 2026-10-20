# GlucoTwin
### A Personalized Glucose Digital Twin — Research Prototype

GlucoTwin is an educational research prototype exploring how a
personalized digital twin can represent glucose measurements,
forecast short-term glucose trends, and identify potentially
significant glucose events using synthetic patient data.

The project combines data generation, machine learning,
evaluation, and an interactive Streamlit dashboard in one
reproducible workflow.

> **Research prototype only:** GlucoTwin uses synthetic data and
> has not been clinically validated. It is not a medical device
> and must not be used for diagnosis, treatment, or medication
> decisions.

---

## 1. Project Objectives

- Build synthetic patient and glucose time-series datasets.
- Explore short-term glucose forecasting.
- Classify potentially significant glucose events.
- Compare model performance with baseline methods.
- Present results through an interactive dashboard.
- Document limitations and opportunities for future research.

## 2. Key Features

- Synthetic patient records and glucose measurements.
- Interactive glucose history visualization.
- Short-term glucose forecast visualization.
- Forecast evaluation against a persistence baseline.
- Glucose event classification and evaluation.
- Model performance summaries and visualizations.
- Data exploration and downloadable records.
- Reproducible data-generation and model-training scripts.

## 3. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Git and GitHub

## 4. System Architecture

```text
Synthetic Data Generation
          |
          v
Data Validation and Preparation
          |
          v
|                      |
v                      v
Glucose Forecasting    Event Classification
|                      |
v                      v
Forecast Evaluation    Event Evaluation
|                      |
           |
           v
    Streamlit Dashboard
           |
           v
 Interactive Visualization