import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
deliveries = pd.read_csv(r"C:\IPL_Match_Win_Predictor\data\deliveries.csv")
matches = pd.read_csv(r"C:\IPL_Match_Win_Predictor\data\matches.csv")

# Merge
df = deliveries.merge(matches, left_on='match_id', right_on='id')

# Only 2nd innings
df = df[df['inning'] == 2]

# Feature Engineering
df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
df['runs_left'] = df['target_runs'] - df['current_score']
df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])
df['wickets'] = df.groupby('match_id')['player_dismissed'].transform(lambda x: x.notnull().cumsum())
df['wickets_left'] = 10 - df['wickets']

# Final dataset
final_df = df[['batting_team','bowling_team','city',
               'runs_left','balls_left','wickets_left',
               'target_runs','winner']].dropna()

final_df['result'] = (final_df['batting_team'] == final_df['winner']).astype(int)
final_df.drop(columns=['winner'], inplace=True)

X = final_df.drop(columns=['result'])
y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline
trf = ColumnTransformer([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'),
     ['batting_team','bowling_team','city'])
], remainder='passthrough')

pipe = Pipeline([
    ('preprocessing', trf),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)

# Save model
pickle.dump(pipe, open("ipl_win_predictor.pkl", "wb"))

print("✅ Model saved!")