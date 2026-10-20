# GlucoTwin
### A Personalized Glucose Digital Twin — Research Prototype

GlucoTwin is an educational research prototype exploring how a personalized digital twin can represent glucose measurements, forecast short-term glucose trends, and identify potentially significant glucose events using synthetic patient data.

The project combines synthetic data generation, data validation, machine learning, model evaluation, and an interactive Streamlit dashboard.

> **Important safety notice:** GlucoTwin uses synthetic data and has not been clinically validated. It is not a medical device and must not be used for diagnosis, treatment, insulin dosing, or other medication decisions. Its outputs are for educational and research exploration only.

## 1. Project Objectives

- Generate synthetic patient records and glucose time-series data.
- Explore short-term glucose forecasting.
- Experiment with glucose event classification.
- Evaluate model performance against appropriate baselines.
- Visualize patient histories, forecasts, events, and evaluation summaries.
- Document the limitations and future research opportunities of a glucose digital twin.

## 2. Key Features

- Synthetic patient and glucose datasets.
- Interactive glucose history visualization.
- Short-term glucose forecast visualization.
- Forecast evaluation against a persistence baseline.
- Glucose event classification and evaluation.
- Model performance summaries.
- Data exploration and downloadable records.
- Scripts for data generation, validation, training, and evaluation.
- Research limitations and safety disclaimer.

## 3. Technology Stack

- Python
- Pandas and NumPy
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
```

The application brings together the data-generation pipeline, forecasting and event-classification workflows, evaluation scripts, and dashboard.

See `glucowin.architecture.pdf` for the architecture document.

## 5. Repository Structure

```text
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
```

The `data/` directory contains the project's synthetic datasets and related generated files. The `docs/` directory is reserved for additional documentation.

## 6. Getting Started

### Requirements

- Python compatible with the versions specified in `requirements.txt`
- Git
- A terminal or Visual Studio Code
- A supported desktop operating system

### Installation

Clone the repository:

```bash
git clone https://github.com/EDDY05ee/GlucoTwin.git
cd GlucoTwin
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Run the dashboard

```powershell
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the dashboard.

## 7. Data and Evaluation

The project uses synthetic data for experimentation rather than real patient records.

Relevant project scripts include:

- `validate_data.py` — data validation.
- `train_model.py` — glucose forecasting model workflow.
- `train_event_model.py` — event-classification model workflow.
- `evaluate_forecast.py` — forecast evaluation.
- `evaluate_events.py` — event evaluation.
- `analyze_event_thresholds.py` — event-threshold analysis.

Consult `EVALUATION_REPORT.md` for the documented evaluation methods and results. Results should be interpreted in the context of the synthetic dataset and the project's research limitations.

No clinical effectiveness or real-world patient benefit is claimed.

## 8. Limitations

- The data are synthetic and may not represent real patient populations.
- Model performance on synthetic data does not establish clinical accuracy.
- Forecasts and event classifications may be wrong.
- The prototype has not been clinically validated.
- The project is not intended for real-world medical decisions.
- Performance may change when data distributions, sampling patterns, or patient characteristics change.

## 9. Future Work

Possible future research directions include:

- Evaluating on appropriately licensed, ethically sourced real-world datasets.
- Improving patient-specific forecasting and uncertainty estimation.
- Testing generalization across different data distributions.
- Strengthening model validation and reproducibility.
- Conducting appropriate clinical, privacy, and safety reviews before any real-world healthcare use.

These are future directions, not claims of completed functionality.

## 10. Project Documents

- [Submission document](GlucoTwin_Submission.pdf)
- [Architecture document](glucowin.architecture.pdf)
- [Evaluation report](EVALUATION_REPORT.md)

## 11. Demonstration

**Demo video:** To be added after recording and uploading the project demonstration.

Once the demonstration is uploaded, add the verified YouTube Unlisted URL here.

## 12. License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 13. Disclaimer

GlucoTwin is an educational research prototype, not a medical device. It has not been clinically validated. Do not use it to diagnose conditions, make treatment decisions, determine insulin doses, or replace advice from a qualified healthcare professional.