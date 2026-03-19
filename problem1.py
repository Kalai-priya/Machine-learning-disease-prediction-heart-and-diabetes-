import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold

# Load dataset
dataset = np.genfromtxt('cleveland_data.csv', dtype=float, delimiter=',')
X = dataset[:, 0:12]  # Feature Set
y = dataset[:, 13]    # Label Set

# Plot PCA-reduced 2D visualization
def plot_2d(data, target, target_names):
    colors = ['r', 'g', 'b', 'c', 'm']
    plt.figure()
    for i, c, label in zip(range(len(target_names)), colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
    plt.legend()
    plt.title("PCA Reduced Feature Visualization")
    plt.savefig('Reduced_PCA_Graph.png')
    plt.close()

# Apply PCA
pca = PCA(n_components=5, whiten=True).fit(X)
X_new = pca.transform(X)

target_names = ['0', '1', '2', '3', '4']
plot_2d(X_new, y, target_names)

# ---- Linear SVM ----
print("\nTesting Linear SVM using train/test split:")
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.4, random_state=0)
model_svm = LinearSVC(C=0.001, max_iter=5000)
model_svm.fit(X_train, y_train)
print("Accuracy (LinearSVC):", model_svm.score(X_test, y_test))

# Full training score
model_svm_full = LinearSVC(C=0.001, max_iter=5000)
model_svm_full.fit(X_new, y)
preds = model_svm_full.predict(X_new)
full_score = np.mean(preds == y)
print("Accuracy without split (LinearSVC):", full_score)

# Likelihood of disease class
unique, counts = np.unique(preds, return_counts=True)
print("\nLikelihood of belonging to each class:")
for label, count in zip(unique, counts):
    print(f"Class {int(label)}: {count / len(preds):.3f}")

# ---- RBF SVM ----
print("\nTesting RBF Kernel SVM using train/test split:")
model_rbf = SVC(C=0.001, kernel='rbf')
model_rbf.fit(X_train, y_train)
print("Accuracy (RBF, split):", model_rbf.score(X_test, y_test))

model_rbf_full = SVC(C=0.001, kernel='rbf')
model_rbf_full.fit(X_new, y)
preds_rbf = model_rbf_full.predict(X_new)
print("Accuracy without split (RBF):", np.mean(preds_rbf == y))

# ---- Stratified K-Fold Validation ----
print("\nTesting with Stratified K-Fold cross-validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = []

for train_index, test_index in skf.split(X_new, y):
    X_train_k, X_test_k = X_new[train_index], X_new[test_index]
    y_train_k, y_test_k = y[train_index], y[test_index]
    model = SVC(C=0.001, kernel='rbf')
    model.fit(X_train_k, y_train_k)
    scores.append(model.score(X_test_k, y_test_k))

print("Cross-validation accuracies:", np.round(scores, 3))
print("Mean CV accuracy:", np.mean(scores))


