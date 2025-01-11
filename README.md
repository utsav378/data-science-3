import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Load dataset
data = pd.read_csv("creditcard_fraud.csv")  # Replace with actual path
X = data.drop('Class', axis=1)
y = data['Class']

# Train model
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X)

# Predict
predictions = model.predict(X)
# Convert predictions (-1 for fraud) to 0/1
predictions = [1 if p == -1 else 0 for p in predictions]

# Evaluate
print("AUC-ROC:", roc_auc_score(y, predictions))
