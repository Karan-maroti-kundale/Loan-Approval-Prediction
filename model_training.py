import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("dataset/train.csv")

# Drop ID column
df.drop("Loan_ID", axis=1, inplace=True)

# ===============================
# Handle Missing Values
# ===============================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ===============================
# One-Hot Encoding
# ===============================
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

# ===============================
# Feature Scaling
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Models
# ===============================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced'
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42
    )
}

best_model = None
best_accuracy = 0

# ===============================
# Train & Evaluate
# ===============================
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {acc:.2f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

# ===============================
# Save Everything (IMPORTANT)
# ===============================
pickle.dump(best_model, open("model/loan_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(X.columns, open("model/features.pkl", "wb"))

print("\nBest Model Saved Successfully!")
print("Best Accuracy:", best_accuracy)
