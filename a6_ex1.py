import pandas as pd
import zipfile
import os

def preprocess_data(zip_path: str, station: str) -> pd.DataFrame:
    # i get the proper filename based on the station and the zip file path
    filename = f"PRSA_Data_20130301-20170228/PRSA_Data_{station}_20130301-20170228.csv"

    # Extract the relevant file
    with zipfile.ZipFile(zip_path, 'r') as z:
        print("Files in zip:", z.namelist())
        if filename not in z.namelist():
            raise FileNotFoundError(f"{filename} not found in zip archive.")
        z.extract(filename, "air_quality_raw")

    # Load CSV
    file_path = os.path.join("air_quality_raw", filename)
    df = pd.read_csv(file_path, encoding="utf-8")

    # here i fill the NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # here i add a new column: datetime, which consists of year, month, day, and hour
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df.set_index("datetime", inplace=True)

    # Drop redundant columns (not needed for analysis since datetime is now the index)
    for col in ["No", "year", "month", "day", "hour", "station"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.to_csv("air_quality_cleaned.csv")

    return df

if __name__ == "__main__":
    zip_path = "PRSA2017_Data_20130301-20170228.zip"
    station = "Aotizhongxin"

    if not os.path.exists(zip_path):
        print(f"Zip file '{zip_path}' not found.")
    else:
        print(f"begining of preprocessing for station: {station}")
        df_cleaned = preprocess_data(zip_path, station)
        print("final cleaned data:")
        print(df_cleaned.head())