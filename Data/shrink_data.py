import pandas as pd
import os

def shrink_file(filename, columns):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in this folder.")
        return
    df = pd.read_csv(filename)
    existing_cols = [c for c in columns if c in df.columns]
    df_slim = df[existing_cols]

    df_slim.to_csv(filename, index=False)
    print(f"Successfully shrunk{filename}. New columns: {existing_cols}")

shrink_file('output.csv', ['date', 'state_location', 'total_cases', 'discharged', 'deaths'])
shrink_file('IndiaCovidVaccination2023.csv', ['total_doses', 'dose1', 'dose_2'])
shrink_file('COVID-19_Sentiments.csv', ['sentiments/public opinion', 'state_location'])