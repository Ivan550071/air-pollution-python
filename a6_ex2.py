import matplotlib.pyplot as plt
import pandas as pd
from a6_ex1 import preprocess_data

def plot_pm25_trend(df: pd.DataFrame):

    if 'PM2.5' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['PM2.5'], label='PM2.5', color='blue')
        plt.title('PM2.5 Levels Over Time')
        plt.xlabel('Date')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.legend()
        plt.grid()
        plt.savefig("eda_pm25_trend.pdf")
    else:
        print("PM2.5 column not found in the DataFrame.")

def plot_correlation(df: pd.DataFrame): #i plot the correlation matrix
    
    numeric_df = df.select_dtypes(include='number') #i only take numerical data, cuz otherwise the heatmap gets broken
    correlation = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title('Correlation Matrix')

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            value = correlation.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color='black')
    
    plt.savefig("eda_correlation_heatmap.pdf")

def plot_histogram_pm25(df: pd.DataFrame):
    
    if 'PM2.5' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['PM2.5'].dropna(), bins=30, color='blue', alpha=0.7)
        plt.title('Histogram of PM2.5 Levels')
        plt.xlabel('PM2.5 (µg/m³)')
        plt.ylabel('Frequency')
        plt.grid()
        plt.savefig('eda_pm25_histogram.pdf')
    else:
        print("PM2.5 column not found in the DataFrame.")

if __name__ == "__main__":
    zip_path = "PRSA2017_Data_20130301-20170228.zip"
    station = "Aotizhongxin"
    df = preprocess_data(zip_path, station)

    plot_pm25_trend(df)

    plot_correlation(df)

    plot_histogram_pm25(df)