import streamlit as st
import pandas as pd
import pickle

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

cities = ['Mumbai','Delhi','Chennai','Kolkata','Bangalore','Hyderabad','Jaipur','Chandigarh']

batting_team = st.selectbox("Batting Team", teams)
bowling_team = st.selectbox("Bowling Team", teams)
city = st.selectbox("City", cities)

total_runs = st.number_input("Target", min_value=0)
runs_left = st.number_input("Runs Left", min_value=0)
balls_left = st.number_input("Balls Left", min_value=0)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)

if st.button("Predict"):
    if batting_team != bowling_team:
        input_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'city': city,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'total_runs': total_runs
        }])

        prob = pipe.predict_proba(input_df)[0][1]
        st.success(f"Win Probability: {round(prob*100,2)}%")
    else:
        st.error("Teams must be different")