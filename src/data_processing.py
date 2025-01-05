import pandas as pd

def load_data(file_path):
    """Load the raw data from a CSV file into a DataFrame."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean the DataFrame by handling missing values and duplicates."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (you can customize this part as needed)
    df = df.fillna(0)  # Replace NaNs with 0s; you might want to use other strategies
    
    return df

def transform_data(df):
    """Transform the DataFrame by creating new features or dropping unnecessary columns."""
    # Example: Create a new feature for goal difference
    df['GoalDifference'] = df['HomeScore'] - df['AwayScore']
    
    # Drop any columns that are not needed for analysis
    df = df.drop(columns=['n', 'game_id'])
    
    return df

def process_data(raw_file_path, processed_file_path):
    """Load, clean, transform, and save the processed data."""
    # Step 1: Load the data
    df = load_data(raw_file_path)

    # Step 2: Clean the data
    df = clean_data(df)

    # Step 3: Transform the data
    df = transform_data(df)

    # Step 4: Save the processed data
    df.to_csv(processed_file_path, index=False)

if __name__ == "__main__":
    # Example usage
    raw_file_path = 'data/raw/season.csv'
    processed_file_path = 'data/processed/processed_data.csv'
    process_data(raw_file_path, processed_file_path)