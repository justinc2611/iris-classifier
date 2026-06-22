import argparse
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)

args = parser.parse_args()

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.random_state
)

# Train model
clf = DecisionTreeClassifier(random_state=args.random_state)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "outputs/iris_model.joblib")

# Make predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Model saved to outputs/iris_model.joblib")
print("Confusion matrix saved to outputs/confusion_matrix.png")