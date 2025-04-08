# logistic_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset with correct separator
df = pd.read_csv('LEC_Winter_Season_2025.csv', sep=';')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Define early game features manually based
early_game_features = ['GD@15', 'CSD@15', 'XPD@15', 'LVLD@15']

# Define and encode the target variable
target = 'Outcome'
df[target] = df[target].map({'Win': 1, 'Loss': 0})

# Features and labels
X = df[early_game_features]
y = df[target]

# Handle missing values (if any)
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Predictions
y_pred = logreg.predict(X_test_scaled)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualize feature importance (coefficients)
coefficients = pd.Series(logreg.coef_[0], index=early_game_features)
coefficients.sort_values().plot(kind='barh', title='Logistic Regression Coefficients')
plt.tight_layout()
plt.show()
