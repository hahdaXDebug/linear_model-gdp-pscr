import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class EducationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PSCR Prediction Dashboard")
        self.geometry("1000x700")
        
        # 1 Load Data
        self.load_data()
        
        # 2 Train Models immediately so they are ready for predictions
        self.train_models()
        
        # 3 Setup GUI Layout
        self.create_widgets()

    def load_data(self):
        try:
            self.df = pd.read_csv("df_clean.csv")
            if "log_GDP per Capita" not in self.df.columns:
                if "GDP per Capita" in self.df.columns:
                    self.df["log_GDP per Capita"] = np.log(self.df["GDP per Capita"])
                elif "GDP_Per_Capita" in self.df.columns:
                     self.df["GDP per Capita"] = self.df["GDP_Per_Capita"]
                     self.df["log_GDP per Capita"] = np.log(self.df["GDP per Capita"])
        except FileNotFoundError:
            messagebox.showerror("Error", "df_clean.csv not found! Generating dummy data.")
            self.df = pd.DataFrame({
                'GDP per Capita': np.linspace(500, 50000, 100),
                'PSCR': np.linspace(50, 100, 100) + np.random.normal(0, 5, 100)
            })
            self.df["log_GDP per Capita"] = np.log(self.df["GDP per Capita"])

    def train_models(self):
        X = self.df[['log_GDP per Capita']]
        y = self.df['PSCR']
        self.lm = LinearRegression()
        self.lm.fit(X, y)
        self.lm_score = r2_score(y, self.lm.predict(X))
        
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf.fit(X, y)
        self.rf_score = r2_score(y, self.rf.predict(X))

    def create_widgets(self):
        nav_frame = tk.Frame(self, bg="#eee", height=50)
        nav_frame.pack(side="top", fill="x")
        
        tk.Button(nav_frame, text="Linear Regression", command=self.show_linear_view).pack(side="left", padx=10, pady=10)
        tk.Button(nav_frame, text="Random Forest", command=self.show_rf_view).pack(side="left", padx=10, pady=10)
        tk.Button(nav_frame, text="Exit", command=self.quit, bg="#ffcccc").pack(side="right", padx=10, pady=10)

        self.content_frame = tk.Frame(self)
        self.content_frame.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        
        self.show_linear_view()

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_linear_view(self):
        self.clear_content()
        self.current_model = "Linear"
        
        # Header
        tk.Label(self.content_frame, text="Linear Regression Model", font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(self.content_frame, text=f"R² Score: {self.lm_score:.4f} (Explains {self.lm_score:.1%} of variance)", font=("Arial", 12)).pack(pady=5)

        # Prediction Control Frame
        control_frame = tk.LabelFrame(self.content_frame, text="Predictor", padx=10, pady=10)
        control_frame.pack(fill="x", pady=10)

        # Slider
        tk.Label(control_frame, text="Adjust GDP per Capita ($):").pack(side="left")
        self.slider_gdp = tk.Scale(control_frame, from_=500, to=100000, orient="horizontal", length=400, command=self.update_prediction)
        self.slider_gdp.set(10000) # Default value
        self.slider_gdp.pack(side="left", padx=20)
        
        # Result Label
        self.result_label = tk.Label(control_frame, text="Predicted PSCR: --%", font=("Arial", 14, "bold"), fg="blue")
        self.result_label.pack(side="left", padx=20)

        # Plot Area
        self.plot_frame = tk.Frame(self.content_frame)
        self.plot_frame.pack(fill="both", expand=True)
        
        self.plot_model(self.lm, "Linear Regression Fit")

    def show_rf_view(self):
        self.clear_content()
        self.current_model = "RF"
        
        # Header
        tk.Label(self.content_frame, text="Random Forest Model", font=("Arial", 18, "bold")).pack(pady=10)
        tk.Label(self.content_frame, text=f"R² Score: {self.rf_score:.4f} (Higher accuracy on complex data)", font=("Arial", 12)).pack(pady=5)

        # Prediction Control Frame
        control_frame = tk.LabelFrame(self.content_frame, text="Predictor", padx=10, pady=10)
        control_frame.pack(fill="x", pady=10)

        # Slider
        tk.Label(control_frame, text="Adjust GDP per Capita ($):").pack(side="left")
        self.slider_gdp = tk.Scale(control_frame, from_=500, to=100000, orient="horizontal", length=400, command=self.update_prediction)
        self.slider_gdp.set(10000)
        self.slider_gdp.pack(side="left", padx=20)
        
        # Result Label
        self.result_label = tk.Label(control_frame, text="Predicted PSCR: --%", font=("Arial", 14, "bold"), fg="green")
        self.result_label.pack(side="left", padx=20)

        # Plot Area
        self.plot_frame = tk.Frame(self.content_frame)
        self.plot_frame.pack(fill="both", expand=True)
        
        self.plot_model(self.rf, "Random Forest Fit")

    def update_prediction(self, val):
        gdp_val = float(val)
        # Transform input because models were trained on LOG GDP
        log_gdp = np.log(gdp_val)
        input_data = pd.DataFrame([[log_gdp]], columns=['log_GDP per Capita'])
        
        if self.current_model == "Linear":
            pred = self.lm.predict(input_data)[0]
        else:
            pred = self.rf.predict(input_data)[0]
            
        self.result_label.config(text=f"Predicted PSCR: {pred:.2f}%")

    def plot_model(self, model, title):
        # Create Figure
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        
        # Plot Scatter (Actual Data)
        ax.scatter(self.df['GDP per Capita'], self.df['PSCR'], alpha=0.3, color='gray', label='Actual Data')
        
        # Create Smooth Line for Model
        x_range = np.linspace(self.df['GDP per Capita'].min(), self.df['GDP per Capita'].max(), 200)
        x_log = np.log(x_range)
        
        # reshape for sklearn
        x_input = pd.DataFrame(x_log, columns=['log_GDP per Capita'])
        y_pred = model.predict(x_input)
        
        color = 'red' if self.current_model == "Linear" else 'green'
        ax.plot(x_range, y_pred, color=color, linewidth=2, label='Model Prediction')
        
        ax.set_title(title)
        ax.set_xlabel("GDP per Capita ($)")
        ax.set_ylabel("PSCR (%)")
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = EducationApp()
    app.mainloop()