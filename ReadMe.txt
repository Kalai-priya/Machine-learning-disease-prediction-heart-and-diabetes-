# Disease Prediction using Machine Learning

## Overview
This project focuses on predicting the likelihood of heart disease and diabetes using machine learning techniques. By analyzing patient health data, the models assist in early detection and risk assessment, supporting better clinical decision-making.

---

## Features
- Predicts heart disease and diabetes using patient medical data  
- Implements supervised machine learning models  
- Includes data preprocessing, feature scaling, and visualization  
- Provides model evaluation using accuracy and classification metrics  
- Generates probability-based predictions for individual patients  

---

## Technologies Used
- Python  
- NumPy  
- Matplotlib and Seaborn  
- Scikit-learn  
- XGBoost  

---

## Datasets
- Cleveland Heart Disease Dataset  
- Diabetes Dataset (based on clinical health parameters such as glucose, BMI, etc.)  

---

## Models
- XGBoost Classifier (primary model for both tasks)  
- Random Forest (optional comparison)  
- PCA for dimensionality reduction and visualization  

---

## Results
- Heart Disease Prediction Accuracy: 92%  
- Diabetes Prediction Accuracy: 81%  

The models effectively classify patients based on medical attributes and provide reliable predictions for both conditions.

---

## Workflow
1. Load datasets  
2. Data preprocessing and cleaning  
3. Feature scaling using StandardScaler  
4. Data visualization (correlation heatmap, histograms, boxplots)  
5. Dimensionality reduction using PCA  
6. Train-test split  
7. Model training with hyperparameter tuning (RandomizedSearchCV)  
8. Evaluation using accuracy and classification report  
9. Prediction of disease probability  

---

## Output
- Accuracy score  
- Classification report  
- Feature importance visualization  
- Probability predictions for heart disease and diabetes  

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git

2. Navigate to the project folder:
cd Heart-Disease-Prediction

3. Install dependencies:
pip install -r requirements.txt

4. Run the diabetes.py and heartdisease.py for the respective prediction results.
Happy coding!
