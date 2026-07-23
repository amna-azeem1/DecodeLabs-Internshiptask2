import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map(dict(enumerate(iris.target_names)))

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['species_name'].value_counts())
print("\nBasic stats:")
print(df.describe())

sns.pairplot(
    df,
    hue="species_name",
    diag_kind="hist"
)
plt.savefig("pairplot.png")
plt.show()


plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop("species_name", axis=1).corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

X = df[iris.feature_names]      # features (sepal/petal measurements)
y = df['species']               # labels (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       
    random_state=42,    
    stratify=y          
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled = scaler.transform(X_test)         # ONLY transform on test

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, pred_k))

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker='o', linestyle='--')
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("k_elbow_plot.png")
plt.show()

best_k = k_range[error_rates.index(min(error_rates))]
print(f"\nBest K found: {best_k} (lowest error rate: {min(error_rates):.4f})")


model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)


print("\n=== Accuracy ===")
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2%}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, predictions)
print(cm)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Iris Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\n=== Classification Report (Precision, Recall, F1) ===")
print(classification_report(y_test, predictions, target_names=iris.target_names))


def predict_new_flower(sepal_length, sepal_width, petal_length, petal_width):
    sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    sample_scaled = scaler.transform(sample)
    pred_class = model.predict(sample_scaled)[0]
    return iris.target_names[pred_class]


new_prediction = predict_new_flower(5.1, 3.5, 1.4, 0.2)
print(f"\nPredicted species for sample flower: {new_prediction}")

joblib.dump(model, "iris_knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")