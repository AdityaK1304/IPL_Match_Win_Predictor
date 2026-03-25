# IPL Dashboard - Streamlit App

import streamlit as st
import pandas as pd
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="IPL Dashboard",
    page_icon="🏏",
    layout="wide"
)

# ================= CSS STYLING =================
st.markdown("""
<style>
.stSelectbox select, .stNumberInput input {
    background-color: #f0f2f6;
    color: #111;
    font-weight: bold;
    border: 1px solid #ccc;
    border-radius: 5px;
    padding: 5px;
}
.stButton button {
    background-color: #1f77b4;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 16px;
}
.stMetric-value { font-size: 32px !important; color: #ff6600; }
.stTable thead th { font-weight: bold; background-color: #f4f6f9; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
option = st.sidebar.radio(
    "Select Dashboard Option",
    ["Pre-Match / Previous Matches", "Live Match Win Prediction"]
)

# ================= LOAD DATA =================
# --- Use relative paths ---
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

# Precompute innings runs
innings_runs = deliveries.groupby(['match_id', 'inning', 'batting_team'])['total_runs'].sum().reset_index()
match_runs = {}
for idx, row in matches.iterrows():
    match_id = row['id']
    team1 = row['team1']
    team2 = row['team2']
    t1_runs = innings_runs[(innings_runs['match_id']==match_id) & (innings_runs['batting_team']==team1)]['total_runs']
    t2_runs = innings_runs[(innings_runs['match_id']==match_id) & (innings_runs['batting_team']==team2)]['total_runs']
    match_runs[match_id] = {
        'team1_runs': int(t1_runs.values[0]) if not t1_runs.empty else 0,
        'team2_runs': int(t2_runs.values[0]) if not t2_runs.empty else 0
    }

# ================= OPTION 1: PRE-MATCH =================
if option == "Pre-Match / Previous Matches":
    st.title("🏏 IPL Pre-Match Dashboard")
    teams = sorted(matches['team1'].unique())
    cities = ['Mumbai','Delhi','Chennai','Kolkata','Bangalore','Hyderabad','Jaipur','Chandigarh']

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
    with col2:
        team2 = st.selectbox("Select Team 2", teams)
    city = st.selectbox("Select City (Venue)", cities)

    if st.button("Show Match Info"):
        if team1 == team2:
            st.error("Please select two different teams")
        else:
            head2head = matches[
                (((matches['team1'] == team1) & (matches['team2'] == team2)) |
                 ((matches['team1'] == team2) & (matches['team2'] == team1)))
                & (matches['city'] == city)
            ]
            total_matches = head2head.shape[0]
            team1_wins = head2head[head2head['winner']==team1].shape[0]
            team2_wins = head2head[head2head['winner']==team2].shape[0]

            team1_scores, team2_scores = [], []
            for idx, row in head2head.iterrows():
                mid = row['id']
                if row['team1']==team1:
                    team1_scores.append(match_runs[mid]['team1_runs'])
                    team2_scores.append(match_runs[mid]['team2_runs'])
                else:
                    team1_scores.append(match_runs[mid]['team2_runs'])
                    team2_scores.append(match_runs[mid]['team1_runs'])
            team1_avg_score = round(pd.Series(team1_scores).mean(),2) if team1_scores else 0
            team2_avg_score = round(pd.Series(team2_scores).mean(),2) if team2_scores else 0

            toss_team1 = head2head[head2head['toss_winner']==team1].shape[0]
            toss_team2 = head2head[head2head['toss_winner']==team2].shape[0]

            def recent_form(team):
                recent = matches[((matches['team1']==team) | (matches['team2']==team))].sort_values('date',ascending=False).head(5)
                wins = recent[recent['winner']==team].shape[0]
                return wins

            team1_recent = recent_form(team1)
            team2_recent = recent_form(team2)

            st.markdown(f"## {team1} vs {team2} in {city}")
            st.markdown("### Head-to-Head Stats")
            st.table(pd.DataFrame({
                "Metric":["Total Matches","Wins","Avg Score","Toss Wins","Recent Form (last 5)"],
                team1:[total_matches, team1_wins, team1_avg_score, toss_team1, f"{team1_recent}/5"],
                team2:["-", team2_wins, team2_avg_score, toss_team2, f"{team2_recent}/5"]
            }).set_index("Metric"))

# ================= OPTION 2: LIVE MATCH PREDICTION =================
else:
    st.title("🏏 IPL Live Match Win Probability Predictor")
    # --- Use relative path ---
    with open("ipl_win_predictor.pkl","rb") as f:
        model = pickle.load(f)

    teams = ['Chennai Super Kings','Delhi Capitals','Kings XI Punjab','Kolkata Knight Riders',
             'Mumbai Indians','Rajasthan Royals','Royal Challengers Bangalore','Sunrisers Hyderabad']
    cities = ['Mumbai','Delhi','Chennai','Kolkata','Bangalore','Hyderabad','Jaipur','Chandigarh']

    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox("Batting Team", teams)
    with col2:
        bowling_team = st.selectbox("Bowling Team", teams)
    with col3:
        city = st.selectbox("City", cities)

    target_runs = st.number_input("Target Runs",0)
    runs_left = st.number_input("Runs Left",0)
    balls_left = st.number_input("Balls Left",0)
    wickets_left = st.number_input("Wickets Left",0,10)

    if st.button("Predict Probability"):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams must be different")
        elif balls_left == 0:
            st.error("Balls left must be greater than 0")
        else:
            input_df = pd.DataFrame([{
                'batting_team':batting_team,
                'bowling_team':bowling_team,
                'city':city,
                'runs_left':runs_left,
                'balls_left':balls_left,
                'wickets_left':wickets_left,
                'target_runs':target_runs
            }])
            probs = model.predict_proba(input_df)[0]
            batting_prob = round(probs[1]*100,2)
            bowling_prob = round(probs[0]*100,2)

            st.markdown("### Match Winning Probability")
            col1, col2 = st.columns(2)
            col1.metric(f"{batting_team}", f"{batting_prob}%")
            col2.metric(f"{bowling_team}", f"{bowling_prob}%")