# Heart Disease Prediction using Machine Learning

This project focuses on predicting the presence of heart disease using health, lifestyle,
and demographic indicators. A complete machine learning pipeline is implemented, including
data preprocessing, model training, evaluation, and visualization.

---

## 1. Problem Statement
The objective of this project is to predict whether an individual has heart disease based on
structured health-related data.

This is formulated as a **binary classification problem**:
- `1` – Heart disease present
- `0` – No heart disease

---

## 2. Data Description
- **Dataset file:** `data/heart.csv`
- **Source:** Health Indicators / Heart Disease Dataset (Kaggle)
- **Target variable:** Heart disease indicator
- **Number of samples:** 500+ records

### Feature Overview
The dataset includes:
- **Health indicators:** high blood pressure, high cholesterol, diabetes, stroke
- **Lifestyle factors:** smoking, alcohol consumption, physical activity
- **Physical & mental health:** general health, mental health days, physical health days
- **Demographics:** age group, sex, education level, income

---

## 3. Project Structure

AI-HEARTDISEASE-PROJECT/
├── data/
│ └── heart.csv
│
├── reports/
│ ├── figures/
│ │ ├── confusion_matrix_dt.png
│ │ ├── confusion_matrix_logreg.png
│ │ ├── confusion_matrix_rf.png
│ │ ├── rf_feature_importance.png
│ │ ├── roc_dt.png
│ │ ├── roc_logreg.png
│ │ └── roc_rf.png
│ │
│ └── results_summary.csv
│
├── src/
│ └── main.py
│
├── README.md
└── requirements.txt


---

## 4. Methodology

### Data Loading
- Data is loaded from `data/heart.csv`
- File existence is validated before processing

### Data Preprocessing
A preprocessing pipeline is constructed using `scikit-learn`:
- Missing values handled with `SimpleImputer`
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- All transformations combined using `ColumnTransformer`

### Models Trained
The following models are trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Model Training
- Dataset split into 80% training and 20% testing
- Hyperparameter tuning performed using `GridSearchCV`
- Fixed random seed used for reproducibility

---

## 5. Results

### Evaluation Metrics
Each model is evaluated using:
- Accuracy
- F1-score
- Confusion Matrix
- ROC Curve
- ROC-AUC score

### Output Files
All results are saved under the `reports/` directory:
- Confusion matrices for all models
- ROC curves for Logistic Regression, Decision Tree, and Random Forest
- Random Forest feature importance plot
- Overall performance metrics stored in `results_summary.csv`

---

## 6. Conclusion
The experimental results show that machine learning models can effectively predict heart
disease risk using structured health data.

- **Random Forest** achieved the best overall performance
- **Logistic Regression** provided a strong and interpretable baseline
- **Decision Tree** was easy to interpret but showed signs of overfitting

This project demonstrates the importance of preprocessing pipelines and ensemble methods
in healthcare-related prediction tasks.
