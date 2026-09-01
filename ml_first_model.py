import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

employees = pd.read_csv("employee_data.csv")

X = employees[["experience"]]
y = employees["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = mean_squared_error(y_test, predictions) ** 0.5

r2 = r2_score(y_test, predictions)

print("\nMean Absolute Error:")
print(mae)

print("\nPredictions:")
print(predictions)

print("\nActual salaries:")
print(y_test)

print("\nMean Squared Error:")
print(mse)

print("\nRoot Mean Squared Error:")
print(rmse)

print("\nR² Score:")
print(r2)

print("\n=== MODEL PARAMETERS ===")

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)


plt.scatter(y_test, predictions)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")

plt.show()