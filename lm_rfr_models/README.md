# PSCR x GDP per Capita Analysis Dashboard

## Overview
This project analyzes the relationship between **GDP per Capita** and **Primary School Completion Rates (PSCR)** using global dataset indicators. It provides an interactive Command Line Interface (CLI) dashboard to visualize data, perform statistical analysis, and compare machine learning models.

The core analysis uses a **Log-Linear Regression Model** to demonstrate the diminishing marginal utility of wealth on education outcomes.

## Features
* **Data Transformation:** Applies a logarithmic transformation (`Log(GDP)`) to linearize the relationship between wealth and social outcomes.
* **Statistical Analysis:** Calculates Slope ($\beta_1$), Intercept ($\beta_0$), and $R^2$ to quantify the economic impact.
* **Machine Learning Workflow:**
    * **Linear Regression:** Trained on 80% of data, tested on 20% to evaluate predictive power (RMSE).
    * **Random Forest:** A non-linear comparison model to capture complex patterns.
* **Visualization:** Generates scatter plots with best-fit lines and model comparisons.

## Project Structure
```text
├── main.py              # Entry point of the application (Main Menu)
├── lm_main.py           # Logic for Linear Regression Model (Choice A)
├── rfr_main.py          # Logic for Random Forest Regressor (Choice B)
├── returnto_main.py     # Helper utility for navigation/exit logic
├── df_clean.csv         # The cleaned dataset used for analysis
└── README.md            # Project documentation