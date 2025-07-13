import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
# Assuming the dataset has columns: 'date' and 'price'
data = pd.read_csv('Gold_data (1).csv')

# Step 3: Preprocess Data
# Convert 'date' to datetime then to ordinal number
data['date'] = pd.to_datetime(data['date'])
data['date_ordinal'] = data['date'].map(lambda x: x.toordinal())

# Step 4: Train-Test Split
X = data[['date_ordinal']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate Model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

score = r2_score(y_test, y_pred)
print("The accuracy of our model is {}%".format(round(score, 2) * 100))

# Step 7: Predict
# Predict price for a new date (July 19, 2024)
new_date = pd.to_datetime('2024-07-19')
new_date_ordinal = new_date.toordinal()
predicted_price = model.predict(pd.DataFrame({'date_ordinal': [new_date_ordinal]}))
print(f"Predicted Gold Price for July 2024: {predicted_price[0]}")

# Step 8: Visualize Results
plt.figure(figsize=(10, 6))

# Plot actual prices (using original dates)
plt.scatter(data['date'], y, color='blue', label='Actual Prices')

# Plot predicted prices on test set (convert ordinal back to datetime)
X_test_dates = pd.to_datetime(X_test['date_ordinal'], origin='unix', unit='D', errors='ignore')  # safer way
# But better is to just convert ordinal to datetime directly:
X_test_dates = X_test['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(x))

plt.plot(X_test_dates, y_pred, color='red', linewidth=2, label='Predicted Prices')

plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('Gold Price Prediction')
plt.legend()
plt.show()
