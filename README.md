# Classification Algorithm Comparison Project

## Overview
This project compares the performance of three classification algorithms—**Random Forest**, **KNN**, and **LSTM**—on a given dataset. The goal is to evaluate these models using 10-fold cross-validation and determine the best-performing algorithm based on metrics such as F1-score, TSS, HSS, and ROC-AUC.

## Objectives
- Implement **Random Forest**, **KNN**, and **LSTM** classifiers using Python libraries.
- Perform **10-fold cross-validation** for robust evaluation.
- Manually calculate key metrics like TP, TN, FP, FN, F1-score, TSS, HSS, etc.
- Compare models using **tabular results** and **ROC curves**.
- Identify the best algorithm and provide insights into its performance.

## Dataset
The dataset used in this project is the **Red Wine Quality Dataset**. The target variable is binary, classifying wine as "good" (quality > 5) or "not good" (quality ≤ 5).

## Methodology
1. **Data Preprocessing**:
   - Normalized feature values using `MinMaxScaler`.
   - Converted the target variable to a binary classification problem.

2. **Algorithms**:
   - **Random Forest**: Ensemble-based model for handling complex data.
   - **KNN**: Instance-based learner sensitive to local relationships.
   - **LSTM**: Deep learning model designed for sequential data.

3. **Evaluation**:
   - Performed 10-fold cross-validation for all models.
   - Calculated metrics for each fold and averaged results across folds.
   - Visualized performance with ROC curves and tabular summaries.

## Results
- **Random Forest** showed the highest F1-score and AUC, making it the most reliable and consistent model.
- **KNN** performed reasonably well but struggled with noisy and high-dimensional data.
- **LSTM** required more computational resources and showed variable performance, as the dataset lacks temporal features.

## Installation

### Prerequisites
Ensure you have Python installed along with the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `tensorflow`
- `termcolor`

### Installing Dependencies
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
