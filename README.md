# GlucoTwin
### A Personalized Glucose Digital Twin — Research Prototype

GlucoTwin is an educational research prototype exploring how a personalized digital twin can represent glucose measurements, forecast short-term glucose trends, and identify potentially significant glucose events using synthetic patient data.

The project combines synthetic data generation, data validation, machine learning, model evaluation, and an interactive Streamlit dashboard.

> **Important Safety Notice:** GlucoTwin uses synthetic data and has not been clinically validated. It is not a medical device and must not be used for diagnosis, treatment, insulin dosing, medication decisions, or other clinical purposes. Its outputs are for educational and research exploration only.

---

## 1. Participant and Team Details

- **Participant Name:** Edwin Vinod
- **Participation Type:** Individual / Solo Project
- **Team Members:** None — individual submission
- **Institution:** Indira Gandhi Institute of Engineering and Technology
- **Location:** Nellikuzhy, Kothamangalam, Ernakulam, Kerala, India
- **Incubator:** Not applicable

## 2. Project Title

**GlucoTwin — A Personalized Glucose Digital Twin: Research Prototype**

GlucoTwin explores how digital-twin-inspired concepts and machine learning workflows can be used to organize and analyze synthetic glucose time-series data in an interactive research environment.

## 3. Problem Statement

Glucose measurements form time-series data that can be explored to understand historical patterns, forecast short-term trends, and identify potentially significant events.

Developing a research workflow that combines these activities can help demonstrate how digital-twin-inspired systems may organize and analyze time-dependent health data.

GlucoTwin explores this concept through synthetic glucose data, forecasting experiments, event classification, evaluation scripts, and interactive visualizations.

The project is an educational research prototype and does not claim clinical effectiveness.

## 4. Healthcare Use Case

GlucoTwin explores a research-oriented workflow for glucose data analysis.

The prototype demonstrates:

- Visualization of historical glucose measurements.
- Exploration of short-term glucose forecasts.
- Experimental glucose event classification.
- Comparison of forecasting performance against a persistence baseline.
- Evaluation summaries for forecasting and event classification.
- Interactive exploration of synthetic patient data.

The intended use case is educational exploration of personalized glucose analytics and digital-twin concepts.

GlucoTwin does not provide medical advice, diagnose health conditions, recommend treatment, or calculate insulin doses. It has not been validated for real-world patient care.

## 5. Project Objectives

- Generate and explore synthetic patient records and glucose time-series data.
- Explore short-term glucose forecasting.
- Experiment with glucose event classification.
- Evaluate model performance against appropriate baselines.
- Visualize patient histories, forecasts, events, and evaluation summaries.
- Document the limitations and future research opportunities of a glucose digital twin.

## 6. Key Features

- Synthetic patient and glucose datasets.
- Interactive glucose history visualization.
- Short-term glucose forecast visualization.
- Forecast evaluation against a persistence baseline.
- Experimental glucose event classification and evaluation.
- Model performance summaries.
- Data exploration and downloadable records, where supported by the dashboard.
- Python scripts for data validation, training, and evaluation.
- Interactive Streamlit dashboard.
- Research limitations and safety disclaimer.

## 7. Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas and NumPy
- **Machine Learning Framework:** Scikit-learn
- **Visualization:** Plotly
- **Interactive Dashboard:** Streamlit
- **Version Control:** Git
- **Code Hosting:** GitHub
- **Documentation:** Markdown and PDF

## 8. AI/ML Model and Framework Details

GlucoTwin contains separate machine learning workflows for glucose forecasting and event classification.

### 8.1 Glucose Forecasting

The forecasting workflow explores short-term glucose trend prediction using synthetic time-series data.

Forecast evaluation includes comparison against a persistence baseline.

Relevant implementation files:

- `train_model.py`
- `evaluate_forecast.py`

### 8.2 Glucose Event Classification

The event-classification workflow explores identifying potentially significant glucose events using generated data.

Model outputs and event thresholds are experimental and have not been clinically validated.

Relevant implementation files:

- `train_event_model.py`
- `evaluate_events.py`
- `analyze_event_thresholds.py`

### 8.3 Data Validation

The project includes a validation workflow for checking generated data before analysis.

Relevant implementation file:

- `validate_data.py`

### 8.4 Model Evaluation

Separate scripts support forecasting evaluation, event evaluation, and threshold analysis.

Refer to the source code and `EVALUATION_REPORT.md` for the actual algorithms, parameters, evaluation methods, and results.

No clinical accuracy or generalization to real patients is claimed.

## 9. System Architecture

The prototype follows this research workflow:

```text
Synthetic Data Generation
           |
           v
Data Validation and Preparation
           |
           +----------------------+
           |                      |
           v                      v
 Glucose Forecasting      Event Classification
           |                      |
           v                      v
 Forecast Evaluation       Event Evaluation
           |                      |
           +-----------+----------+
                       |
                       v
              Streamlit Dashboard
                       |
                       v
          Interactive Visualization

The application brings together the data-generation pipeline, forecasting and event-classification workflows, evaluation scripts, and dashboard.
See glucowin.architecture.pdf for the architecture document.

10. Repository Structure


GlucoTwin/
├── app.py
├── data/
├── docs/
├── analyze_event_thresholds.py
├── evaluate_events.py
├── evaluate_forecast.py
├── train_event_model.py
├── train_model.py
├── validate_data.py
├── EVALUATION_REPORT.md
├── GlucoTwin_Submission.pdf
├── glucowin.architecture.pdf
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore

The data/ directory contains the project's synthetic datasets and related generated files. The docs/ directory is reserved for additional documentation.

11. Getting Started

Requirements
Python compatible with the versions specified in requirements.txt
Git
A terminal or Visual Studio Code
A supported desktop operating system

11.1 Clone the Repository

git clone https://github.com/EDDY05ee/GlucoTwin.git
cd GlucoTwin

11.2 Create and Activate a Virtual Environment
On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

11.3 Install Dependencies

python -m pip install -r requirements.txt

11.4 Run the Dashboard

streamlit run app.py

18. Acknowledgment

GlucoTwin was developed as an individual educational research prototype exploring digital-twin-inspired glucose analytics, synthetic time-series data, machine learning, model evaluation, and interactive visualization.

Project Links

Participant: Edwin Vinod
GitHub Repository: https://github.com/EDDY05ee/GlucoTwin⁠
Demonstration Video: https://youtu.be/MOqSYF24CZE⁠
DEMO VIDEO : https://youtu.be/ixA3Khw0cLM

Architecture Document: glucowin.architecture.pdf
Submission Document: GlucoTwin_Submission.pdf
Evaluation Report: EVALUATION_REPORT.md

Disclaimer

GlucoTwin is an educational research prototype, not a medical device. It has not been clinically validated. Do not use it to diagnose conditions, make treatment decisions, determine insulin doses, or replace advice from a qualified healthcare professional.