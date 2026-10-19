# Privacy & Ethics in Healthcare AI

A responsible-AI healthcare analytics project focused on privacy-preserving, fair, and explainable prediction of hospital readmission risk using synthetic clinical and administrative data.

This repository combines data preprocessing, privacy safeguards, model evaluation, fairness auditing, explainability, and compliance-oriented documentation for a healthcare use case. The project is designed for research, experimentation, and educational use.

## Overview

The project aims to demonstrate how an ML pipeline can be built and evaluated with an emphasis on:

- Privacy protection and de-identification
- Fairness and bias assessment
- Model robustness and validation
- Explainable AI (XAI)
- Compliance awareness for healthcare regulations
- Reproducible testing and documentation

The primary modeling task is a clinical readmission risk prediction problem, using synthetic patient-level data similar to UCI-inspired healthcare data sources.

## Project Goals

- Create a privacy-aware healthcare ML pipeline
- Remove direct identifiers and pseudonymize patient records
- Build and compare predictive models for readmission risk
- Evaluate fairness and model robustness under realistic conditions
- Explain predictions using XAI methods
- Document the system in a model card format
- Provide a reproducible test suite for core pipeline components

## Repository Structure

```text
Privacy-and-ethics/
├── .gitignore
├── README.md
├── src/
│   ├── pipeline_core.py              # Core pipeline logic and model functions
│   ├── test_pipeline.py             # Pytest suite for validation
│   ├── model_card.md                # Model documentation / card
│   ├── responsible_ai_healthcare.ipynb  # Notebook with analysis and workflow
│   ├── data/
│   │   ├── 1_clinical_treatment_synthetic.csv
│   │   ├── 2_hospital_readmission_synthetic.csv
│   │   ├── demo_clinical.csv
│   │   ├── demo_readmission.csv
│   │   ├── EU_AI_ACT.pdf
│   │   ├── GDPR.pdf
│   │   ├── HIPAA.pdf
│   │   └── architecture/
│   │       ├── Architecture_diagram.png
│   │       └── Basic Linear Pipeline.png
│   ├── Output/
│   │   ├── cleaned_master.csv
│   │   ├── merged_master.csv
│   │   ├── integrated_patient_master.csv
│   │   ├── processed_admin_hospital_stays.csv
│   │   ├── processed_clinical_lab_results.csv
│   │   ├── model_predictions_output.csv
│   │   └── 4_scrubbed_dataset.csv
│   ├── visualization/
│   │   ├── adversarial_robustness.png
│   │   ├── clinical_markers.png
│   │   ├── correlation_heatmap.png
│   │   ├── demographic_analysis.png
│   │   ├── differential_privacy_analysis.png
│   │   ├── fairness_metrics.png
│   │   ├── feature_importance_prescreening.png
│   │   ├── gbm_evaluation.png
│   │   ├── imbalance_strategy_comparison.png
│   │   ├── learning_curves.png
│   │   ├── lime_explanations.png
│   │   ├── model_comparison.png
│   │   ├── model_evaluation.png
│   │   ├── pdp_ice_plots.png
│   │   ├── permutation_importance.png
│   │   ├── shap_global.png
│   │   ├── shap_local_waterfall.png
│   │   ├── target_distribution.png
│   │   └── xai_method_comparison.png
│   └── chroma_db/
│       └── ...
└── .DS_Store
```

## Key Features

### Privacy and de-identification
- SHA-256 patient ID generation
- Removal of direct identifiers and PII columns
- Pseudonymization strategy aligned with privacy-preserving healthcare practice
- Synthetic data use for experimentation and safe research workflows

### Data engineering
- IQR outlier clipping
- Composite feature engineering from clinical markers
- Low-variance feature filtering
- Missing value handling via median imputation

### Modeling
- Random Forest classifier with class balancing
- SMOTE-based oversampling for imbalanced classes
- Evaluation metrics including:
  - weighted F1 score
  - ROC-AUC
  - prediction flip rate for robustness checks

### Fairness and ethics
- Bias-related analysis and documentation
- Review of potential proxy features and demographic disparities
- Clinical decision-support framing rather than autonomous clinical decision-making

### Explainability
- SHAP
- LIME
- Permutation importance
- Partial dependence and ICE plots
- Model comparison charts and interpretability metrics

### Regulatory context
Included documentation and source files reference:
- HIPAA
- GDPR
- EU AI Act

The project is aligned with privacy, ethical review, and high-risk AI governance principles in healthcare contexts.

## Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- SHAP
- LIME
- Fairlearn
- diffprivlib
- Matplotlib / Seaborn
- Jupyter Notebook
- pytest

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/Gowrikamahesh17/Privacy-and-ethics.git
cd Privacy-and-ethics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -U pip
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn jupyter pytest shap lime fairlearn diffprivlib
```

If a project-specific requirements file is added later, use:

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the core pipeline tests

From the repository root:

```bash
pytest src/test_pipeline.py -q
```

### Open the notebook

```bash
jupyter notebook src/responsible_ai_healthcare.ipynb
```

### Inspect the model code

```bash
python -c "import src.pipeline_core as pc; print('Loaded pipeline_core successfully')"
```

## Data Description

The project includes synthetic datasets designed for experimentation with privacy-preserving and responsible-healthcare AI workflows.

### Input data
- `src/data/1_clinical_treatment_synthetic.csv`
- `src/data/2_hospital_readmission_synthetic.csv`
- `src/data/demo_clinical.csv`
- `src/data/demo_readmission.csv`

These datasets include synthetic clinical variables such as:
- lab values
- treatment information
- demographics
- hospital readmission outcomes
- administrative metadata

### Output data
Generated outputs in `src/Output/` include processed and cleaned tabular datasets used for training and evaluation.

Examples:
- `cleaned_master.csv`
- `merged_master.csv`
- `integrated_patient_master.csv`
- `processed_admin_hospital_stays.csv`
- `processed_clinical_lab_results.csv`
- `model_predictions_output.csv`

## Model and Evaluation

The core model logic is implemented in `src/pipeline_core.py`.

### Core functions include:
- patient pseudonymization (`generate_patient_id`, `scrub_pii`)
- outlier clipping (`cap_outliers_iqr`)
- composite feature creation (`add_composite_features`)
- low-variance filtering (`drop_low_variance`)
- feature preparation (`prepare_features`)
- class imbalance handling (`apply_smote`)
- model training (`train_random_forest`)
- evaluation (`evaluate_model`)
- robustness check (`prediction_flip_rate`)

### Key metrics
The project evaluates:
- Weighted F1 score
- ROC-AUC
- Class imbalance ratio
- Prediction stability under noisy features

## Ethical and Privacy Notes

This project is intended for educational and research use, not for autonomous clinical decisions.

Important principles in this repository:
- patient identifiers are pseudonymized rather than kept in plaintext
- sensitive columns are removed in processed datasets
- fairness and robustness checks are part of the workflow
- predictions should be reviewed by qualified clinicians
- use on real patient data requires proper institutional review and governance approval

## Documentation

- `src/model_card.md` contains a model card for the healthcare prediction system
- `src/responsible_ai_healthcare.ipynb` contains the notebook-based analysis
- the repository includes legal and compliance reference documents in `src/data/`

## Testing

The repository includes a Pytest suite covering:
- privacy functions
- feature engineering
- imbalance handling
- model evaluation
- robustness checks
- data validation

Example:

```bash
pytest src/test_pipeline.py -q
```

## Limitations

This project has several important limitations:
- it uses synthetic data rather than real clinical records
- real-world deployment would require formal clinical validation
- fairness and privacy safeguards must be audited in the target deployment environment
- data drift and evolving clinical patterns require regular retraining and monitoring

## Contribution

Contributions are welcome. Suggested improvements include:
- adding a formal `requirements.txt`
- providing a reproducible CLI entry point
- extending fairness evaluation with demographic groups
- documenting deployment and monitoring procedures
- integrating automated CI checks

## License

No explicit license file is present in this repository at the moment. Please treat this as a research and education project unless a license is added later.

## Disclaimer

This project is for demonstration, learning, and experimentation. It is not a production clinical decision system and should not be used as a substitute for medical judgment or regulatory approval.

## Contact

Repository owner:
- Gowrikamahesh17

Project URL:
- https://github.com/Gowrikamahesh17/Privacy-and-ethics

