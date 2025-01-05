import pandas as pd

# Load the processed data
data_path = "data/processed/processed_data.csv"
df = pd.read_csv(data_path)

def get_match_features(row):
    """
    Get features for a single match based on the match data.
    :param row: Row of the DataFrame representing a match
    :return: Dictionary of match features
    """
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    
    # Get the match score and expected goals for the current match
    home_score = row["HomeScore"]
    away_score = row["AwayScore"]
    home_xg = row["Home_xG"]
    away_xg = row["Away_xG"]
    
    # Determine the match result
    if home_score > away_score:
        result = 1  # Home win
    elif away_score > home_score:
        result = -1  # Away win
    else:
        result = 0  # Draw

    # Get additional match statistics
    home_shots = row["Home_shots"]
    away_shots = row["Away_shots"]
    home_corners = row["Home_corner"]
    away_corners = row["Away_corner"]
    home_pk_goal = row["Home_PK_Goal"]
    away_pk_goal = row["Away_PK_Goal"]
    home_pk_shots = row["Home_PK_shots"]
    away_pk_shots = row["Away_PK_shots"]
    home_top = row["Home_ToP"]
    goal_difference = row["GoalDifference"]

    # Create feature dictionary for the match
    match_features = {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "result": result,  # Include the match result
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_corners": home_corners,
        "away_corners": away_corners,
        "home_pk_goal": home_pk_goal,
        "away_pk_goal": away_pk_goal,
        "home_pk_shots": home_pk_shots,
        "away_pk_shots": away_pk_shots,
        "home_top": home_top,
        "goal_difference": goal_difference,
    }

    return match_features

def generate_all_match_features(df):
    """
    Generate features for all matches in the dataset.
    :param df: Processed DataFrame with match data
    :return: DataFrame with additional features
    """
    match_features_list = []

    for _, row in df.iterrows():
        match_features = get_match_features(row)
        match_features_list.append(match_features)

    # Convert list of features to a DataFrame
    features_df = pd.DataFrame(match_features_list)
    return features_df

# Generate features and save to a new CSV
features_df = generate_all_match_features(df)
features_df.to_csv("data/features_data.csv", index=False)

print("Feature generation complete! Saved to data/features_data.csv")
