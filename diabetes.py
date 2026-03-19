import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore')

#  Load dataset
data = pd.read_csv('diabetes.csv')
print(" Dataset shape:", data.shape)

#  Handle zero-like missing values
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zero:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

print("\n Missing values handled successfully.")

#  Split into features & target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

#  Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Balance dataset using SMOTE
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_scaled, y)
print("\n Dataset balanced using SMOTE.")
print("Class distribution after balancing:", np.bincount(y_bal))

#  Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)

#  Define XGBoost model & hyperparameter grid
xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    use_label_encoder=False
)

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\n Performing Grid Search for best parameters...")
grid_search.fit(X_train, y_train)

#  Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

print("\n Best Parameters:", grid_search.best_params_)
print(" Best CV Accuracy:", round(grid_search.best_score_, 4))
print("\n Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

#  Show first 5 predicted probabilities
print("\n Predicted Diabetes Probabilities (first 5 patients):")
for i, p in enumerate(y_probs[:5], start=1):
    print(f"Patient {i} → Diabetes Probability: {p * 100:.2f}%")

# Save model & scaler for later use
joblib.dump(best_model, 'diabetes_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n Model and scaler saved successfully as 'diabetes_xgb_model.pkl' and 'scaler.pkl'.")


