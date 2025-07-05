'''Project Name: Golden State Real Estate: A Price Prediction Model
+This project is created using the Regression of Supervised Machine Learning concepts from "Machine Learning Specialization- 
offerd by Standford Online". I wanna thank the instructor of the course- Andrew for this amazing course and
this project is the reflection of his teachings. Note- this project is created using the resources 
offed by Standford Online and Deep Learning IO thorugh Coursera@.... 
'''
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing # To load the real-world dataset
import matplotlib.pyplot as plt
import pandas as pd 

print("--- Simple Linear Regression Project (Real-World Data) ---")
print("\nLoading California Housing Dataset...")
housing = fetch_california_housing(as_frame=True) 
X = housing.data  
y = housing.target 

print("Dataset loaded successfully.")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print("\nFirst 5 rows of Features (X):")
print(X.head())
print("\nFirst 5 values of Target (y):")
print(y.head())
print(f"\nFeature names: {housing.feature_names}")


feature_for_plot = 'MedInc'
X_plot = X[[feature_for_plot]] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData Split:")
print(f"Training X shape: {X_train.shape}")
print(f"Testing X shape: {X_test.shape}")
print(f"Training y shape: {y_train.shape}")
print(f"Testing y shape: {y_test.shape}")

model = LinearRegression()


print("\nTraining the Linear Regression Model...")
model.fit(X_train, y_train) 
print("Model training complete.")


y_pred = model.predict(X_test)

print("\nPredictions on Test Data (first 10 samples):")
for i in range(min(10, len(X_test))):
    
    original_index = X_test.index[i]
    
    med_inc = X_test.loc[original_index, 'MedInc']
    print(f"District {original_index}: MedInc: ${med_inc:.2f}k, Actual Value: ${y_test.iloc[i]:.2f}k, Predicted Value: ${y_pred[i]:.2f}k")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_plot, y, color='blue', alpha=0.5, label='Actual Data Points (MedInc vs. Value)')


simple_model_for_plot = LinearRegression()
simple_model_for_plot.fit(X_plot, y)
plt.plot(X_plot, simple_model_for_plot.predict(X_plot), color='red', linewidth=2, label='Simple Regression Line (based on MedInc)')

plt.xlabel('Median Income (tens of thousands of USD)')
plt.ylabel('Median House Value (hundreds of thousands of USD)')
plt.title('California Housing: Median House Value Prediction')
plt.legend()
plt.grid(True)
plt.show()

new_district_features = pd.DataFrame([[8.0, 25.0, 6.0, 1.0, 1500.0, 3.0, 34.0, -118.0]],
                                     columns=housing.feature_names) 

predicted_value = model.predict(new_district_features)
print(f"\nPrediction for a new hypothetical district:")
print(f"Features: {new_district_features.iloc[0].to_dict()}")
print(f"Predicted Median House Value: ${predicted_value[0]:.2f}k")

