import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

MODEL_FILE = 'final_model_data.csv'
try:
    df = pd.read_csv(MODEL_FILE)
except FileNotFoundError:
    print(f"Error: Model data file '{MODEL_FILE}' not found. Please run aggregate_features.py first.")
    

# Handle potential missing values (e.g., in '_std' columns for weeks with 1 article)
# We will fill these with 0 for simplicity.
df = df.fillna(0)

# --- 2. Feature Engineering & Selection ---
# Our goal is to predict the 'Outcome' based on the weekly features.

# Convert the 'Outcome' column to a binary format (0 or 1)
df['Outcome_binary'] = df['Outcome'].apply(lambda x: 1 if x == 'Conflict' else 0)

# The features for our model are all the numerical columns we created
# We exclude identifiers like Event_ID and the original Outcome string
features = [col for col in df.columns if col not in ['Event_ID', 'Outcome', 'Outcome_binary']]

X = df[features]
y = df['Outcome_binary']

print(f"Data prepared. Using {len(X.columns)} features.")
print(f"Dataset has {len(df)} total weekly observations.")
print(f"Conflict cases: {y.sum()} | No Conflict cases: {len(y) - y.sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete.")
print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")
print("\nConfusion Matrix:")
# A confusion matrix shows us what the model got right and what it got wrong.
# [[True Negative, False Positive], [False Negative, True Positive]]
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Conflict', 'Conflict']))
import numpy as np
# --- 7. Interpret the Model - Find the "Tipping Point" ---
print("\n--- Model Interpretation: Most Important Features ---")

# Get the coefficients (the "importance" score) the model learned for each feature
coefficients = model.coef_[0]

# Create a DataFrame to view the features and their learned importance
feature_importance = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

# Sort by the absolute value of the coefficient to see the most impactful features
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

print("Top 10 features predicting 'Conflict' (Positive Coefficients):")
print(feature_importance[feature_importance['Coefficient'] > 0].head(10))

print("\nTop 10 features predicting 'No Conflict' (Negative Coefficients):")
print(feature_importance[feature_importance['Coefficient'] < 0].head(10))
