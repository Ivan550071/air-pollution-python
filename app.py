import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget
from a6_ex4 import PM_Model

#run with shiny run --reload --launch-browser --port=0 app.py

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("file", "Upload cleaned CSV", accept=[".csv"]),
        ui.input_selectize("pollutant", "Select pollutant(s)", POLLUTANTS, selected="PM2.5", multiple=True),
        ui.input_slider("window", "Rolling average window (days)", 1, 30, 1),
        ui.input_checkbox("show_pred", "Show PM2.5 prediction", False),
        ui.input_file("scaler_file", "Upload scaler (.pkl)", accept=[".pkl"], multiple=False),
        ui.input_file("model_file", "Upload model parameters (.pt)", accept=[".pt"], multiple=False),
    ),
    ui.output_plot("pollutant_plot", height="800px", width="1200px"),
)

def server(input, output, session):
    @reactive.Calc
    def data():
        file = input.file()

        if not file:
            return None
        
        df = pd.read_csv(file[0]["datapath"])
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
        return df

    #function to plot the predicted PM2.5 values
    @reactive.Calc
    def model_pred():
        
        if not input.show_pred():
            return None
        
        scaler_file = input.scaler_file()
        model_file = input.model_file()
        df = data()

        if not (scaler_file and model_file and df is not None):
            return None
        
        # Load scaler
        with open(scaler_file[0]["datapath"], "rb") as f:
            scaler = pickle.load(f)

        # Prepare features (numeric columns except PM2.5)
        feature_cols = [col for col in df.select_dtypes(include='number').columns if col != "PM2.5"]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        # Load model
        input_size = X_scaled.shape[1]
        model = PM_Model(input_size=input_size)
        model.load_state_dict(torch.load(model_file[0]["datapath"], map_location="cpu"))
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            pred = model(X_tensor).squeeze().numpy()
        return pred

    @output
    @render.plot
    def pollutant_plot():
        df = data()

        if df is None:
            return
        
        window = input.window()
        pollutants = input.pollutant()
        if isinstance(pollutants, str):
            pollutants = [pollutants]
        plt.figure(figsize=(12, 6))
        for pol in pollutants:
            if pol in df.columns:
                plt.plot(df.index, df[pol].rolling(window).mean(), label=pol)

        # Plot prediction if u selected it
        if input.show_pred() and "PM2.5" in pollutants:
            pred = model_pred()
            if pred is not None:
                plt.plot(df.index, pd.Series(pred, index=df.index).rolling(window).mean(), label="PM2.5 Prediction", linestyle="--")

        plt.legend()
        plt.title("Pollutant Levels (Rolling Average)")
        plt.xlabel("Date")
        plt.ylabel("Concentration")
        plt.tight_layout()

app = App(app_ui, server)