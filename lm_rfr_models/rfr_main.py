import returnto_main as rt_main  # Importing the file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rfr_model_main():
    df_clean = pd.read_csv("df_clean.csv")

    while True: 
        print("Random Forest Regression Model Dashboard")
        print("Select an option:")
        print("1. View Supervised Machine Learning Training and Testing Sets")
        print("2. View Transformed Data")
        print("3. Return to Main Menu")
        choice = input("Enter your choice: ")
        try:
            choice = int(choice)
            if choice == 1:
                df_clean["log_GDP per Capita"] = np.log(df_clean["GDP per Capita"])
                X = df_clean[['log_GDP per Capita']] #predictor
                y = df_clean['PSCR']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred_rf = rf_model.predict(X_test)
                rf_r2 = r2_score(y_test, y_pred_rf)
                rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
                mean_squared_error(y_test, y_pred_rf)
                print(" Statistics ")
                print(f"Random Forest R²:      {rf_r2:.4f}")
                print(f"Random Forest RMSE:    {rf_rmse:.2f}")

                plt.figure(figsize=(12, 6))

                # Plot actual data
                plt.scatter(df_clean['log_GDP per Capita'], df_clean['PSCR'], color='blue', alpha=0.3, label='Actual Data')
                X_range = np.linspace(X['log_GDP per Capita'].min(), X['log_GDP per Capita'].max(), 200).reshape(-1, 1)

                # Random Forest Prediction Line
                y_rf_range = rf_model.predict(X_range)
                plt.plot(X_range, y_rf_range, color='green', linewidth=2, label='Random Forest')

                plt.title('Random Forest Regressor Model')
                plt.xlabel('Log(GDP per Capita)')
                plt.ylabel('Completion Rate (%)')
                plt.legend()
                plt.grid(True, alpha=0.5)
                plt.show()

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
                print("Returning to main menu.")
                return 
            else:
                print("Invalid choice. Please select from the choices.")
        except ValueError:
            print("Invalid input. Please select from the choices.")