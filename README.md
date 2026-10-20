# GlucoTwin
## A Personalized Glucose Digital Twin — Research Prototype

GlucoTwin is an educational research prototype exploring how a personalized digital twin can represent glucose measurements, forecast short-term glucose trends, and identify potentially significant glucose events using synthetic patient data.

The project combines synthetic data generation, data validation, machine learning, model evaluation, event analysis, and an interactive Streamlit dashboard.

> **Important Safety Notice:** GlucoTwin is an experimental educational prototype. It is not a medical device, is not clinically validated, and must not be used for diagnosis, treatment, medication decisions, or emergency medical decisions. All included patient data is synthetic.

---

## 1. Participant and Team Details

- **Participant Name:** Edwin Vinod
- **Participation Type:** Individual / Solo
- **Team Members:** None
- **Institution:** Indira Gandhi Institute of Engineering and Technology, Nellikuzhy, Kothamangalam, Ernakulam, Kerala, India
- **Incubator:** Not applicable

## 2. Project Title

**GlucoTwin: A Personalized Glucose Digital Twin for Glucose Trend Forecasting and Event Analysis**

## 3. Problem Statement

Glucose measurements can vary over time because of multiple factors. Looking at individual readings alone may not provide a clear view of changing trends or potentially significant events.

GlucoTwin explores how a software-based digital twin can represent a synthetic individual's glucose history, estimate short-term glucose trends, and identify candidate events for further investigation.

The prototype demonstrates a research workflow for data preparation, forecasting, event analysis, evaluation, and visualization.

## 4. Healthcare Use Case

The proposed research use case is an exploratory glucose-monitoring environment in which a digital representation of a synthetic individual is used to:

- Visualize historical glucose measurements.
- Explore short-term glucose forecasts.
- Flag potentially significant glucose patterns for analysis.
- Review model performance and limitations.
- Investigate how personalized health-data models could be evaluated in future research.

This prototype does not provide medical advice or make clinical decisions.

## 5. Project Objectives

1. Build a reproducible research prototype using synthetic glucose data.
2. Represent an individual's glucose history in a digital-twin-style workflow.
3. Train and evaluate a machine-learning model for glucose forecasting.
4. Explore candidate glucose-event detection methods.
5. Validate data and assess model performance.
6. Present the results through an interactive dashboard.
7. Document the architecture, methodology, limitations, and future improvements.

## 6. Key Features

- **Synthetic Data:** Uses synthetic glucose measurements for experimentation.
- **Data Validation:** Checks the dataset before model training and evaluation.
- **Glucose Forecasting:** Explores short-term glucose prediction using a trained model.
- **Event Analysis:** Investigates potentially significant glucose patterns.
- **Model Evaluation:** Includes scripts for evaluating forecasting and event-analysis performance.
- **Interactive Dashboard:** Uses Streamlit to explore data, forecasts, events, and model information.
- **Research Documentation:** Includes evaluation findings, architecture documentation, and project limitations.
- **Privacy-Aware Development:** Uses synthetic data rather than real patient records.

## 7. Technology Stack

- **Programming Language:** Python
- **Dashboard:** Streamlit
- **Data Processing:** Pandas and NumPy
- **Machine Learning:** Scikit-learn and libraries listed in `requirements.txt`
- **Data Visualization:** Libraries specified in `requirements.txt`
- **Development Environment:** Visual Studio Code
- **Version Control:** Git and GitHub
- **Data Source:** Synthetic glucose data

## 8. AI and Machine Learning

The project separates data preparation, forecasting, event analysis, and evaluation into individual scripts.

- `train_model.py` — trains the glucose forecasting model.
- `evaluate_forecast.py` — evaluates forecasting performance.
- `train_event_model.py` — trains or prepares the event-analysis model.
- `evaluate_events.py` — evaluates event-analysis performance.
- `analyze_event_thresholds.py` — investigates the effects of event thresholds.
- `validate_data.py` — checks the input data.

These workflows help explore model behavior and understand the limitations of machine-learning predictions on synthetic data.

**Research limitation:** Performance on synthetic data does not establish clinical accuracy or prove that the model will generalize to real patients.

## 9. System Architecture

The prototype follows this general workflow:

```text
Synthetic Glucose Data
          |
          v
    Data Validation
          |
          v
   Data Preparation
          |
          v
   +------+------+
   |             |
   v             v
Forecasting   Event Analysis
   |             |
   v             v
Forecast       Candidate
Evaluation     Event Evaluation
   |             |
   +------+------+
          |
          v
  Interactive Streamlit
       Dashboard
          |
          v
 Results, Visualizations,
 and Research Exploration
```

The architecture describes the research workflow. It does not represent a clinically deployed healthcare system.

## 10. Repository Structure

The main repository components include:

```text
GlucoTwin/
├── app.py
├── data/
├── train_model.py
├── train_event_model.py
├── evaluate_forecast.py
├── evaluate_events.py
├── analyze_event_thresholds.py
├── validate_data.py
├── EVALUATION_REPORT.md
├── README.md
├── requirements.txt
├── LICENSE
├── GlucoTwin_Submission.pdf
└── glucowin.architecture.pdf
```

The exact contents may evolve as the project is updated.

## 11. How to Run the Project

### Prerequisites

- Python 3.12 or a compatible version.
- Git (optional if downloading the repository as a ZIP).
- A terminal or Visual Studio Code.

### Step 1: Clone the Repository

```bash
git clone https://github.com/EDDY05ee/GlucoTwin.git
cd GlucoTwin
```

### Step 2: Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

For Windows Command Prompt:

```bat
.venv\Scripts\activate
```

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Launch the Dashboard

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to explore the dashboard.

## 12. Data and Evaluation

GlucoTwin uses synthetic glucose data to demonstrate its workflow without requiring real patient information.

The project includes scripts for:

- Checking data validity.
- Training the forecasting model.
- Evaluating forecasting performance.
- Training and evaluating event-analysis methods.
- Exploring event-threshold choices.

Read the [Evaluation Report](EVALUATION_REPORT.md) for the project's documented evaluation findings.

Results should be interpreted in the context of the dataset, experimental setup, evaluation method, and documented limitations. Synthetic-data results alone are not evidence of clinical effectiveness.

## 13. Interactive Dashboard

The Streamlit dashboard organizes the research prototype into six areas:

1. **Glucose History** — explores historical glucose measurements.
2. **Forecast** — displays the prototype's glucose forecasting results.
3. **Events** — explores candidate glucose events.
4. **Data Explorer** — inspects the available dataset.
5. **Model Summary** — presents model-related information.
6. **About** — describes the project and its limitations.

The dashboard is designed for research exploration and demonstration, not clinical use.

## 14. Project Documentation

- [GitHub Repository](https://github.com/EDDY05ee/GlucoTwin)
- [Evaluation Report](https://github.com/EDDY05ee/GlucoTwin/blob/main/EVALUATION_REPORT.md)
- [Submission PDF](https://github.com/EDDY05ee/GlucoTwin/blob/main/GlucoTwin_Submission.pdf)
- [Architecture PDF](https://github.com/EDDY05ee/GlucoTwin/blob/main/glucowin.architecture.pdf)

These links require the corresponding files to exist in the public repository.

## 15. Demo Videos

Two GlucoTwin demonstration videos are available on YouTube:

### Demo Video 1

[▶ Watch GlucoTwin Demo — Video 1](https://youtu.be/ixA3Khw0cLM)

### Demo Video 2

[▶ Watch GlucoTwin Demo — Video 2](https://youtu.be/MOqSYF24CZE)

Both links are provided for project demonstration and review. Please check each video to confirm its contents, duration, and audio before using it for competition submission.

## 16. Limitations and Safety

- The prototype is intended for educational and research purposes.
- The project uses synthetic data and does not establish performance on real-world patient data.
- Forecasts and event flags may be inaccurate.
- No clinical validation or regulatory approval is claimed.
- The prototype must not be used to diagnose, treat, or manage a medical condition.
- It must not be used to make medication, insulin-dosing, or emergency-care decisions.
- Any future healthcare deployment would require appropriate real-world validation, privacy and security safeguards, clinical oversight, and applicable regulatory review.

## 17. Future Work

Potential future improvements include:

- Testing on appropriately licensed, de-identified real-world datasets.
- Evaluating forecasting performance across different individuals and time periods.
- Improving uncertainty estimation and reporting.
- Investigating personalized model adaptation.
- Conducting rigorous event-detection evaluation.
- Improving reproducibility, testing, and documentation.
- Exploring secure and privacy-preserving healthcare data workflows.

These are proposed future directions and are not claims of completed features.

## 18. License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for the applicable terms.

## 19. Acknowledgment

Developed as an educational research prototype for the **Digital Twin Challenge 2026**.

## Disclaimer

GlucoTwin is a research and educational prototype, not a medical device or a substitute for professional medical advice. Its predictions and event flags are experimental and have not been clinically validated. Do not use this project for diagnosis, treatment, medication decisions, or emergency medical decisions.