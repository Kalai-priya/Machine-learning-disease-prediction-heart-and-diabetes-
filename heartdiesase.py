import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

dataset = np.genfromtxt('cleveland_data.csv', dtype=float, delimiter=',')
X = dataset[:, 0:12]
y = dataset[:, 13]

y = np.where(y != 0.0, 1, 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

feature_names = [
'age','sex','cp','trestbps','chol','fbs',
'restecg','thalach','exang','oldpeak','slope','ca'
]

corr = np.corrcoef(X_scaled.T)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
plt.title("Feature Correlation Heatmap")
plt.show()

plt.figure(figsize=(12,8))
for i in range(X.shape[1]):
    plt.subplot(3,4,i+1)
    plt.hist(X[:,i], bins=20)
    plt.title(feature_names[i])
plt.tight_layout()
plt.show()

plt.figure()
sns.boxplot(x=y, y=X[:,4])
plt.title("Cholesterol vs Heart Disease")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

def plot_2D(data, target):
    colors = ['r','b']
    plt.figure()
    for i,c in zip(range(2),colors):
        plt.scatter(data[target==i,0], data[target==i,1], c=c, label=str(i))
    plt.legend()
    plt.title("PCA Reduced Visualization")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig('Improved_PCA_Graph.png')
    plt.show()

plot_2D(X_pca, y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

param_dist = {
'n_estimators':[100,200,300],
'max_depth':[3,4,5,6],
'learning_rate':[0.01,0.05,0.1,0.2],
'subsample':[0.8,1.0],
'colsample_bytree':[0.8,1.0]
}

search = RandomizedSearchCV(
estimator=xgb,
param_distributions=param_dist,
n_iter=20,
scoring='accuracy',
cv=5,
random_state=42,
verbose=1,
n_jobs=-1
)

search.fit(X_train,y_train)

best_model = search.best_estimator_

print("\nBest Parameters:",search.best_params_)
print("Best Cross-Validation Accuracy:",search.best_score_)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

print("\nTest Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))

importance = best_model.feature_importances_

plt.figure(figsize=(8,6))
plt.barh(feature_names,importance)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.show()

for i in range(5):
    print(f"Person {i+1}: Disease Probability = {y_prob[i]*100:.2f}%")










