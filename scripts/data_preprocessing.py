import pandas as pd

def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Convert Date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Extract year, month, and week
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    
    return df
