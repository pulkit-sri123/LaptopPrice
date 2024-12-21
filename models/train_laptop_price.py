import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
import pickle

# Load the dataset
file_path = 'C:/Users/DELL/Desktop/LAB/Proj1_LocalHost/files_for_training_model/laptopPrice.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Data Cleaning and Preprocessing
data_cleaned = data.copy()

# Remove units (e.g., "GB", "stars") and convert to numeric
columns_to_clean = ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 'weight']
for col in columns_to_clean:
    data_cleaned[col] = data_cleaned[col].str.extract('(\d+)').astype(float)

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 
                       'ram_type', 'os', 'os_bit', 'warranty', 'Touchscreen', 'msoffice', 'rating']

for col in categorical_columns:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
    label_encoders[col] = le

# Define features and target
X = data_cleaned.drop(columns=['Price'])
y = data_cleaned['Price']

from sklearn.impute import SimpleImputer

# Impute missing values with column mean
imputer = SimpleImputer(strategy='mean')
X_imputed_array = imputer.fit_transform(X)

# Ensure column alignment
X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns[:X_imputed_array.shape[1]])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Function to evaluate model with different feature combinations
def evaluate_combinations(X_train, X_test, y_train, y_test, feature_combinations):
    results = []
    for combination in feature_combinations:
        # Select feature subset
        X_train_subset = X_train[list(combination)]
        X_test_subset = X_test[list(combination)]

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train_subset, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test_subset)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Store results
        results.append({
            "Features": combination,
            "RMSE": rmse,
            "R2": r2
        })
    return results

# Generate feature combinations (up to 3 features for simplicity)
all_features = X_train.columns
feature_combinations = [combo for r in range(1, 4) for combo in combinations(all_features, r)]

# Evaluate models
results = evaluate_combinations(X_train, X_test, y_train, y_test, feature_combinations)

# Convert results to DataFrame and sort by R2
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

# Display top 5 results
print(results_df.head())


# Select the best feature combination based on R2
best_features = ['ram_gb', 'ssd', 'graphic_card_gb']  # Replace with the best feature combination from results_df

# Train the final model using the best features
X_train_best = X_train[best_features]
X_test_best = X_test[best_features]

final_model = LinearRegression()
final_model.fit(X_train_best, y_train)

# Save the model using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(final_model, file)

print("Model saved successfully!")

# Get the coefficients and intercept
coefficients = final_model.coef_
intercept = final_model.intercept_

# Display the equation
equation = f"Price = {intercept:.2f}"
for coef, feature in zip(coefficients, best_features):
    equation += f" + ({coef:.2f} * {feature})"
print("Linear Regression Equation for Price Prediction:")
print(equation)

# Example Prediction
new_data = {'ram_gb': 16, 'ssd': 512, 'graphic_card_gb': 4}  # Replace with actual values
new_data_df = pd.DataFrame([new_data])
predicted_price = final_model.predict(new_data_df)
print(f"Predicted Price for laptop with {new_data}: {predicted_price[0]:.2f}")

"""import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(final_model, file)"""