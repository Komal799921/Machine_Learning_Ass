# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Retail_Sales_Dataset.csv")   # Change to your file name

# Display first 5 rows
print(df.head())

# Features (Independent Variables)
X = df[['Age', 'Quantity', 'Price per Unit']]

# Target (Dependent Variable)
y = df['Total Amount']

# Split dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model Parameters
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Evaluation
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Compare Actual vs Predicted
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print("\nActual vs Predicted")
print(results.head(10))

# Scatter Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Total Amount")
plt.ylabel("Predicted Total Amount")
plt.title("Actual vs Predicted Total Amount")
plt.grid(True)
plt.show()



"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Retail_Sales_Dataset.csv")

print(df.head())
print(df.info())
print(df.tail())

# Input features
X = df[["Age", "Quantity", "Price per Unit"]]

# Target
y = df["Total Amount"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict new value
print("Prediction:", model.predict([[25, 3, 100]]))"""