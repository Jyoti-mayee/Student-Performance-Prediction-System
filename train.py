import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 1. Data Understanding & Explanation
print("Loading dataset...")
data_path = 'data/student_performance_dataset.xlsx'
df = pd.read_excel(data_path)

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Drop unnecessary columns if they exist
columns_to_drop = ['Name', 'Roll_Number', 'University_Number', 'Date_of_Birth']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Identify Target Column
target_col = 'Secured_Percentage'
if target_col not in df.columns:
    for col in df.columns:
        if 'Secured' in col or 'Percentage' in col:
            target_col = col
            break

print(f"\nTarget Column identified as: {target_col}")

# Drop rows with missing target
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# Identify Numerical and Categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical Features: {numeric_features}")
print(f"Categorical Features: {categorical_features}")

# 2. EDA

# Target Distribution
plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True, bins=30)
plt.title('Distribution of Secured Percentage')
plt.xlabel('Secured Percentage')
plt.ylabel('Frequency')
plt.savefig('visualizations/target_distribution.png')
plt.close()

# Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_features + [target_col]].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')
plt.close()

# 3. Data Preprocessing Pipeline
print("Setting up Preprocessing Pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model Building (Linear Regression Only)
print("Training Linear Regression Model...")

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation Results (Linear Regression) ---")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Actual Secured Percentage')
plt.ylabel('Predicted Secured Percentage')
plt.savefig('visualizations/actual_vs_predicted.png')
plt.close()

# 5. Model Saving
model_path = 'models/linear_regression_model.pkl'
joblib.dump(pipeline, model_path)

print(f"Model saved to {model_path}")
print("Training completed successfully.")