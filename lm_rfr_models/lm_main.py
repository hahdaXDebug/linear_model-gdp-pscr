import returnto_main as rt_main  # Importing the file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def linear_model_main():
    df_clean = pd.read_csv("df_clean.csv")

    while True: 
        print("Linear Regression Model Dashboard")
        print("Select an option:")
        print("1. Linear Regression Statistical Summary")
        print("2. View Transformed Data")
        print("3. View Supervised Machine Learning Training and Testing Sets")
        print("4. Return to Main Menu")
        choice = input("Enter your choice: ")
        try:
            choice = int(choice)
            if choice == 1:
                print("Linear Regression Model Summary:")
                df_clean["log_GDP per Capita"] = np.log(df_clean["GDP per Capita"])
                X = df_clean[['log_GDP per Capita']] #predictor
                y = df_clean['PSCR']

                model = LinearRegression()
                model.fit(X, y)

                y_pred = model.predict(X)

                slope = model.coef_[0]
                intercept = model.intercept_
                r2 = r2_score(y, y_pred)

                print("\nModel Results ")
                print(f"Slope (Coefficient): {slope:.4f}")
                print(f"Intercept: {intercept:.4f}")
                print(f"R-squared: {r2:.4f}")
                print("Interpretation: A 1% increase in GDP is associated with a " f"{slope/100:.4f} point increase in completion rate.")
            
            elif choice == 2:
                df_clean["log_GDP per Capita"] = np.log(df_clean["GDP per Capita"])
                print("Transformed Data:")
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df_clean, x='PSCR', y='log_GDP per Capita')
                plt.title('Primary School Completion Rate vs Log of GDP per Capita')
                plt.xlabel('Primary School Completion Rate (%)')
                plt.ylabel('Log of GDP per Capita (PPP)')
                plt.show()
            
            elif choice == 3:
                print("View Training and Testing Sets (uses 80/20 by default):")
                df_clean["log_GDP per Capita"] = np.log(df_clean["GDP per Capita"])
                X = df_clean[['log_GDP per Capita']] #predictor
                y = df_clean['PSCR']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                lm = LinearRegression()
                lm.fit(X_train, y_train)
                y_pred_test = lm.predict(X_test)
                mse = mean_squared_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                r2_train = lm.score(X_train, y_train)
                r2_test = r2_score(y_test, y_pred_test)

                print("--- Machine Learning Evaluation ---")
                print(f"Model Intercept: {lm.intercept_:.2f}")
                print(f"Model Slope:     {lm.coef_[0]:.2f}")
                print("-" * 30)
                print(f"Training R²:     {r2_train:.4f} (How well it learned known data)")
                print(f"Testing R²:      {r2_test:.4f} (How well it generalizes to new data)")
                print(f"RMSE:            {rmse:.2f} (Average error in percentage points)")

                plt.figure(figsize=(10, 6))

                # Plot Training Data (Blue)
                plt.scatter(X_train, y_train, color='blue', alpha=0.3, label='Training Data (80%)')

                # Plot Testing Data (Green)
                plt.scatter(X_test, y_test, color='green', alpha=0.7, edgecolors='black', label='Testing Data (20%)')

                # Plot the Regression Line (based on model)
                line_x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                line_y = lm.predict(line_x)
                plt.plot(line_x, line_y, color='red', linewidth=2, label='ML Prediction Line')

                plt.title('Supervised Learning: Training vs. Testing Sets')
                plt.xlabel('Log(GDP per Capita)')
                plt.ylabel('Primary Completion Rate (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            elif choice == 4:
                print("Returning to main menu.")
                return 
            else:
                print("Invalid choice. Please select from the choices.")
        except ValueError:
            print("Invalid input. Please select from the choices.")