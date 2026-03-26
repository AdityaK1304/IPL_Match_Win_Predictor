from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# ================= LOAD =================
model = pickle.load(open("ipl_win_predictor.pkl", "rb"))
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# ================= CLEAN DATA =================
team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants"
}

matches.replace(team_mapping, inplace=True)
deliveries.replace(team_mapping, inplace=True)

# CLEAN TEXT
for col in ['team1', 'team2', 'venue']:
    matches[col] = matches[col].astype(str).str.strip()

# REMOVE CITY FROM VENUE
matches['venue'] = matches['venue'].apply(lambda x: x.split(",")[0])

# STANDARDIZE VENUES
venue_mapping = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "Mohali Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Mohali Stadium"
}
matches['venue'] = matches['venue'].replace(venue_mapping)

# UNIQUE VALUES
teams = sorted(set(matches['team1']).union(set(matches['team2'])))
venues = sorted(set(matches['venue'].dropna()))

# ================= HOME =================
@app.route('/')
def home():
    return render_template("home.html")

# ================= LIVE =================
@app.route('/live')
def live():
    return render_template("live.html", teams=teams, venues=venues)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    if data['batting_team'] == data['bowling_team']:
        return render_template("live.html",
            teams=teams, venues=venues,
            error="Teams must be different"
        )

    input_df = pd.DataFrame([{
        'batting_team': data['batting_team'],
        'bowling_team': data['bowling_team'],
        'city': data['city'],
        'runs_left': int(data['runs_left']),
        'balls_left': int(data['balls_left']),
        'wickets_left': int(data['wickets_left']),
        'target_runs': int(data['target_runs'])
    }])

    probs = model.predict_proba(input_df)[0]

    return render_template("live.html",
        teams=teams,
        venues=venues,
        result=True,
        batting_team=data['batting_team'],
        bowling_team=data['bowling_team'],
        batting_prob=round(probs[1]*100,2),
        bowling_prob=round(probs[0]*100,2)
    )

# ================= PREVIOUS =================
@app.route('/previous', methods=['GET','POST'])
def previous():

    if request.method == "POST":

        team1 = request.form['team1']
        team2 = request.form['team2']
        venue = request.form['venue']

        if team1 == team2:
            return render_template("previous.html",
                teams=teams, venues=venues,
                error="Select different teams"
            )

        h2h = matches[
            (((matches['team1']==team1)&(matches['team2']==team2))|
             ((matches['team1']==team2)&(matches['team2']==team1))) &
            (matches['venue']==venue)
        ]

        total = len(h2h)
        t1_wins = len(h2h[h2h['winner']==team1])
        t2_wins = len(h2h[h2h['winner']==team2])

        t1_pct = round((t1_wins/total)*100,2) if total else 0
        t2_pct = round((t2_wins/total)*100,2) if total else 0

        # RUN STATS
        innings = deliveries.groupby(['match_id','batting_team'])['total_runs'].sum().reset_index()

        t1_scores, t2_scores = [], []

        for _, row in h2h.iterrows():
            mid = row['id']

            t1 = innings[(innings['match_id']==mid)&(innings['batting_team']==team1)]
            t2 = innings[(innings['match_id']==mid)&(innings['batting_team']==team2)]

            if not t1.empty:
                t1_scores.append(int(t1['total_runs'].values[0]))
            if not t2.empty:
                t2_scores.append(int(t2['total_runs'].values[0]))

        def avg(lst): return round(sum(lst)/len(lst),2) if lst else 0

        # ✅ CHASING & DEFENDING
        t1_chasing = len(h2h[
            (h2h['winner']==team1) &
            (h2h['toss_decision']=='field') &
            (h2h['toss_winner']==team1)
        ])

        t2_chasing = len(h2h[
            (h2h['winner']==team2) &
            (h2h['toss_decision']=='field') &
            (h2h['toss_winner']==team2)
        ])

        t1_defending = t1_wins - t1_chasing
        t2_defending = t2_wins - t2_chasing

        return render_template("previous.html",
            teams=teams, venues=venues, result=True,
            team1=team1, team2=team2,
            total=total,
            t1_wins=t1_wins, t2_wins=t2_wins,
            t1_pct=t1_pct, t2_pct=t2_pct,
            t1_avg=avg(t1_scores), t2_avg=avg(t2_scores),
            t1_chasing=t1_chasing,
            t2_chasing=t2_chasing,
            t1_defending=t1_defending,
            t2_defending=t2_defending
        )

    return render_template("previous.html", teams=teams, venues=venues)


if __name__ == "__main__":
    app.run(debug=True)