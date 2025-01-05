import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the processed data and the trained model
data_path = "data/features_data.csv"
df = pd.read_csv(data_path)
model = load_model("football_match_predictor.h5")  # Update with the path to your model

def get_team_features(home_team, away_team, df):
    """
    Prepare the average features for a match based on the home and away team.
    :param home_team: Name of the home team
    :param away_team: Name of the away team
    :param df: DataFrame containing match data
    :return: Dictionary of averaged team features
    """
    home_matches = df[df["home_team"] == home_team]
    away_matches = df[df["away_team"] == away_team]

    # Calculate averages for the home team
    home_avg_xG = home_matches["home_xg"].mean() if not home_matches.empty else 0
    home_avg_shots = home_matches["home_shots"].mean() if not home_matches.empty else 0
    home_avg_corners = home_matches["home_corners"].mean() if not home_matches.empty else 0
    home_avg_PK_goals = home_matches["home_pk_goal"].mean() if not home_matches.empty else 0
    home_avg_PK_shots = home_matches["home_pk_shots"].mean() if not home_matches.empty else 0
    home_avg_ToP = home_matches["home_top"].mean() if not home_matches.empty else 0
    home_avg_goal_difference = home_matches["goal_difference"].mean() if not home_matches.empty else 0
    home_avg_score = home_matches["home_score"].mean() if not home_matches.empty else 0

    # Calculate averages for the away team
    away_avg_xG = away_matches["away_xg"].mean() if not away_matches.empty else 0
    away_avg_shots = away_matches["away_shots"].mean() if not away_matches.empty else 0
    away_avg_corners = away_matches["away_corners"].mean() if not away_matches.empty else 0
    away_avg_PK_goals = away_matches["away_pk_goal"].mean() if not away_matches.empty else 0
    away_avg_PK_shots = away_matches["away_pk_shots"].mean() if not away_matches.empty else 0
    away_avg_goal_difference = away_matches["goal_difference"].mean() if not away_matches.empty else 0
    away_avg_score = away_matches["away_score"].mean() if not away_matches.empty else 0

    # Create a dictionary with averaged features
    features = {
        "home_team": home_team,
        "away_team": away_team,
        "home_xG": home_avg_xG,
        "away_xG": away_avg_xG,
        "home_shots": home_avg_shots,
        "away_shots": away_avg_shots,
        "home_corners": home_avg_corners,
        "away_corners": away_avg_corners,
        "home_PK_goals": home_avg_PK_goals,
        "away_PK_goals": away_avg_PK_goals,
        "home_PK_shots": home_avg_PK_shots,
        "away_PK_shots": home_avg_PK_shots,
        "home_ToP": home_avg_ToP,
        "goal_difference": home_avg_goal_difference - away_avg_goal_difference,  # Home - Away
        "home_score": home_avg_score,
        "away_score": away_avg_score
    }

    return features

def predict_match_result(home_team, away_team, num_simulations=100):
    """
    Predict the match result based on team names and return probabilities.
    :param home_team: Name of the home team
    :param away_team: Name of the away team
    :param num_simulations: Number of simulations to run
    :return: Dictionary with predicted results and probabilities
    """
    home_wins = 0
    away_wins = 0
    draws = 0

    for _ in range(num_simulations):
        # Get team features
        features = get_team_features(home_team, away_team, df)

        # Prepare the input DataFrame for the model
        input_data = pd.DataFrame([{
            "home_score": features["home_score"],
            "away_score": features["away_score"],
            "home_xG": features["home_xG"],
            "away_xG": features["away_xG"],
            "home_shots": features["home_shots"],
            "away_shots": features["away_shots"],
            "home_corners": features["home_corners"],
            "away_corners": features["away_corners"],
            "home_PK_goals": features["home_PK_goals"],
            "away_PK_goals": features["away_PK_goals"],
            "home_PK_shots": features["home_PK_shots"],
            "away_PK_shots": features["away_PK_shots"],
            "home_ToP": features["home_ToP"],
            "goal_difference": features["goal_difference"]
        }])

        # Make predictions
        prediction = model.predict(input_data)[0][0]  # Assuming single output for match result
        result = 1 if prediction > 0.5 else (-1 if prediction < -0.5 else 0)  # Interpreting the output
        
        # Count the results
        if result == 1:
            home_wins += 1
        elif result == -1:
            away_wins += 1
        else:
            draws += 1

    # Calculate probabilities
    total_simulations = home_wins + away_wins + draws
    return {
        "home_win_prob": home_wins / total_simulations,
        "away_win_prob": away_wins / total_simulations,
        "draw_prob": draws / total_simulations
    }

if __name__ == "__main__":
    home_team = input("Enter home team: ")
    away_team = input("Enter away team: ")
    
    result_data = predict_match_result(home_team, away_team, num_simulations=100)
    print(f"{home_team} win probability: {result_data['home_win_prob']:.2%}")
    print(f"{away_team} win probability: {result_data['away_win_prob']:.2%}")
    print(f"Draw probability: {result_data['draw_prob']:.2%}")
