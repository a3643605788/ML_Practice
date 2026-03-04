# Machine Learning Practice & Pipeline Implementation 

This repository showcases my practical implementation of end-to-end Machine Learning projects. As a **Software Engineer** transitioning into **ML Engineering**, my focus is on building **modular, scalable, and production-ready pipelines** rather than just running isolated scripts.

---

## Core Objectives
* **Engineering Excellence:** Applying clean code principles to ML workflows.
* **Pipeline Automation:** Integrating preprocessing, training, and evaluation into unified pipelines.
* **Model Serving:** Transitioning from local experiments to deployable APIs.

---

## Tech Stack
* **Language:** Python 3.x
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Pipelines, Feature Engineering, Classification)
* **Deep Learning (Current focus):** PyTorch (CNNs for Image Classification)
* **Tools:** Git, Docker, Kaggle API

---

## Project Structure
A structured directory for better maintainability and reproducibility:

```
├── data/               # Datasets (Raw and Processed)
├── notebooks/          # EDA and Experimental Research
├── src/                # Production-grade Source Code
│   ├── preprocess.py   # Feature engineering and cleaning
│   ├── train.py        # Model training logic
│   └── utils.py        # Helper functions
├── models/             # Serialized models (.joblib / .pth)
└── requirements.txt    # Environment dependencies
```

## Featured Project: Titanic Survival Pipeline
An automated classification system to predict passenger survival using structured data.

### Key Technical Implementations
* **Custom Transformers:** Built Scikit-learn compatible classes for specialized data cleaning and feature engineering to ensure code reusability.
* **Pipeline Integration:** Leveraged `sklearn.pipeline` to encapsulate the entire workflow, effectively preventing **data leakage** between training and validation sets.
* **Model Evaluation:** Conducted rigorous performance analysis by comparing multiple algorithms (Random Forest, Logistic Regression) using **ROC-AUC** and **F1-Score**.
* **Hyperparameter Tuning:** Systematically optimized model performance through automated `GridSearchCV`.

---

## Quick Start

### 1. Setup Environment

# Clone the repository
```
git clone [https://github.com/a3643605788/ML_Practice.git](https://github.com/a3643605788/ML_Practice.git)
cd ML_Practice
pip install -r requirements.txt
```

### 2. Run Training Pipeline
```
python src/train.py
```

## Engineering Mindset
I believe ML models are only as good as the systems they run on. My approach includes:

* Modular Design: Separating data logic from model logic.
* Reproducibility: Ensuring results are consistent across different environments.
* Scalability: Building scripts that can easily transition from 100 to 100,000 data points.