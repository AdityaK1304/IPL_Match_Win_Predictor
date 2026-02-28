import streamlit as st
import pandas as pd
import pickle

# Load model
with open("research/ipl_win_predictor.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏏 IPL Win Probability Predictor")

teams = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

cities = [
    'Mumbai','Delhi','Chennai','Kolkata',
    'Bangalore','Hyderabad','Jaipur','Chandigarh'
]

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
city = st.selectbox("City", cities)

target_runs = st.number_input("Target Runs", min_value=0)
runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)

if st.button("Predict Probability"):

    if batting_team == bowling_team:
        st.error("Batting and Bowling teams must be different")
    else:
        input_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'target_runs': target_runs
        }])

        probs = model.predict_proba(input_df)[0]

        batting_team_prob = round(probs[1] * 100, 2)
        bowling_team_prob = round(probs[0] * 100, 2)

        st.subheader("Match Winning Probability")

        st.success(f"{batting_team} 🏏 – {batting_team_prob}%")
        st.success(f"{bowling_team} 🎯 – {bowling_team_prob}%")