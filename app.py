import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Set Streamlit config
st.set_page_config(page_title="ODI Cricket Analysis & Prediction", layout="wide")
DEFAULT_FIGSIZE = (6, 4)

# Load data
@st.cache_data
def load_data():
    batting = pd.read_csv('odi_Batting_Card.csv')
    bowling = pd.read_csv('odi_Bowling_Card.csv')
    fow = pd.read_csv('odi_Fow_Card.csv')
    matches = pd.read_csv('odi_Matches_Data.csv')
    partnership = pd.read_csv('odi_Partnership_Card.csv')
    players = pd.read_csv('players_info.csv')
    return batting, bowling, fow, matches, partnership, players

batting, bowling, fow, matches, partnership, players = load_data()

# Preprocessing
@st.cache_data
def preprocess(batting, bowling, fow, matches, partnership, players):
    bowling = bowling.merge(players[['player_id', 'player_name']], how='left', left_on='bowler id', right_on='player_id')\
                     .rename(columns={'player_name':'bowler_name'}).drop('player_id', axis=1)

    partnership = partnership.merge(players[['player_id', 'player_name']], how='left', left_on='player1', right_on='player_id')\
                     .rename(columns={'player_name':'player1_name'}).drop('player_id', axis=1)
    partnership = partnership.merge(players[['player_id', 'player_name']], how='left', left_on='player2', right_on='player_id')\
                     .rename(columns={'player_name':'player2_name'}).drop('player_id', axis=1)

    fow['wicket'] = pd.to_numeric(fow['wicket'], errors='coerce').dropna().astype(int)
    players_dict = players.set_index('player_id')['player_name'].to_dict()
    matches['MOM Player'] = matches['MOM Player'].map(players_dict).fillna(matches['MOM Player'])
    matches['Match Date'] = pd.to_datetime(matches['Match Date'], errors='coerce')
    matches['Year'] = matches['Match Date'].dt.year
    return batting, bowling, fow, matches, partnership

batting, bowling, fow, matches, partnership = preprocess(batting, bowling, fow, matches, partnership, players)

def generate_plots():
    figs = {}

    # Outlier Detection
    fig, ax = plt.subplots(figsize=(5,3))
    sampled = batting.sample(n=1000, random_state=42)
    sns.stripplot(x=sampled['runs'], color='purple', size=2, jitter=True, ax=ax)
    ax.set_title("Outlier Detection")
    figs['strip'] = fig

    # Balls vs Runs
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.scatterplot(data=batting, x='balls', y='runs', hue='team', s=10, palette='tab10', ax=ax)
    ax.set_title("Balls vs Runs")
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 10:
        ax.legend(handles[:10], labels[:10])
    figs['scatter'] = fig

    # Runs Distribution
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.histplot(batting['runs'], bins=50, kde=True, color='orange', ax=ax)
    ax.set_title("Runs Distribution")
    figs['runs'] = fig

    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    corr = batting[['runs','balls','fours','sixes','strikeRate']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Batting Correlation")
    figs['corr'] = fig

    # Matches Per Year
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    yearly = matches['Year'].value_counts().sort_index()
    sns.lineplot(x=yearly.index, y=yearly.values, marker='o', color='green', ax=ax)
    ax.set_title("Matches Per Year")
    figs['year'] = fig

    # Fall of Wickets
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.countplot(x=fow['wicket'], color='salmon', ax=ax)
    ax.set_title("Fall of Wickets")
    figs['fow'] = fig

    # Strike Rate Distribution
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.histplot(batting['strikeRate'], bins=50, kde=True, color='skyblue', ax=ax)
    ax.set_title("Strike Rate Distribution")
    figs['strike'] = fig

    # Top Bowlers
    top_bowlers = bowling.groupby('bowler_name')['wickets'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.barplot(x=top_bowlers.values, y=top_bowlers.index, color='orange', ax=ax)
    ax.set_title("Top Bowlers by Wickets")
    figs['top_bowlers'] = fig

    # Extras
    extras = bowling[['wides','noballs']].sum()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    extras.plot(kind='bar', color=['tomato', 'gold'], ax=ax)
    ax.set_title("Extras (Wides & No Balls)")
    figs['extras'] = fig

    # Partnership Runs
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.histplot(partnership['partnership runs'], bins=50, kde=True, color='purple', ax=ax)
    ax.set_title("Partnership Runs Distribution")
    figs['partnership'] = fig

    return figs

figs = generate_plots()

@st.cache_data
def train_models(matches):
    # Prepare Toss Decision Model
    toss_data = matches[['Team1 Name','Team2 Name','Match Venue (Stadium)','Toss Winner','Toss Winner Choice']].dropna()
    toss_data = toss_data[
        (toss_data['Toss Winner'] == toss_data['Team1 Name']) | 
        (toss_data['Toss Winner'] == toss_data['Team2 Name'])]

    toss_y = toss_data['Toss Winner Choice']
    toss_X = pd.get_dummies(toss_data[['Team1 Name','Team2 Name','Match Venue (Stadium)','Toss Winner']])

    if len(toss_X) == 0:
        st.warning("⚠ Not enough data for Toss Decision Model after filtering.")
        return None

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(toss_X, toss_y, test_size=0.5, random_state=42)

    toss_model = RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42)
    toss_model.fit(Xt_train, yt_train)
    yt_pred = toss_model.predict(Xt_test)
    toss_acc = accuracy_score(yt_test, yt_pred)

    # Prepare Match Winner Model
    match_data = matches[['Team1 Name','Team2 Name','Match Venue (Stadium)','Toss Winner','Toss Winner Choice','Match Winner']].dropna()
    match_data = match_data[
        (match_data['Match Winner'] == match_data['Team1 Name']) | 
        (match_data['Match Winner'] == match_data['Team2 Name'])
    ]

    match_y = match_data['Match Winner']
    match_X = pd.get_dummies(match_data[['Team1 Name','Team2 Name','Match Venue (Stadium)','Toss Winner','Toss Winner Choice']])

    if len(match_X) == 0:
        st.warning("⚠ Not enough data for Match Winner Model after filtering.")
        return None

    Xm_train, Xm_test, ym_train, ym_test = train_test_split(match_X, match_y, test_size=0.5, random_state=42)

    match_model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42)
    match_model.fit(Xm_train, ym_train)
    ym_pred = match_model.predict(Xm_test)
    match_acc = accuracy_score(ym_test, ym_pred)

    return {
        'toss_model': toss_model,
        'toss_X_test': Xt_test,
        'toss_y_test': yt_test,
        'toss_pred': yt_pred,
        'toss_acc': toss_acc,
        'match_model': match_model,
        'match_X_test': Xm_test,
        'match_y_test': ym_test,
        'match_pred': ym_pred,
        'match_acc': match_acc,
        'toss_X_columns': toss_X.columns,
        'match_X_columns': match_X.columns
    }

model_results = train_models(matches)


# Sidebar Navigation
st.sidebar.title("🏏 ODI Cricket Match Outcome Prediction")
page = st.sidebar.radio(
    "Go to", 
    ["Introduction", "Data Overview", "EDA", "ML Model","Prediction", "Conclusion"]
)

# Page rendering logic
if page == "Introduction":
    st.title("🏏 ODI Cricket Match Outcome Prediction")

    st.markdown("""
    Welcome to the **Cricket Match Outcome Prediction App**!

    In this project, we leverage machine learning techniques to predict two key aspects of a cricket match:

    - **Toss Decision Prediction**: Predict whether the toss-winning team will choose to bat or field first.
    - **Match Winner Prediction**: Predict which team is likely to win the match based on pre-match conditions.

    The models are trained on historical match data, including information such as:
    - Team names
    - Match venues
    - Toss winner
    - Toss decision
    - Match result

    We have used **Random Forest Classifiers** to build robust predictive models, achieving an approximate accuracy of **70%** on both tasks.

    This app offers:
    - An interactive interface for **data exploration and visualization**.
    - **Model training and evaluation** with performance metrics.
    - A **prediction module** where you can input new match details to get real-time predictions.

    Explore the tabs to navigate through the full functionality of the application.
    """)

elif page == "Data Overview":
    st.title("ODI Cricket Dataset Overview")

    # ================= Batting Card =================
    with st.expander("1️⃣ Batting Card ", expanded=False):
        st.write("**Shape:**", batting.shape)
        st.dataframe(batting.dropna().head())

        st.subheader("Column Descriptions:")
        batting_dict = {
            'MatchID': 'Unique Match Identifier',
            'innings': 'Innings number (1 or 2)',
            'team': 'Team batting',
            'batsman': 'Batsman name',
            'runs': 'Runs scored by batsman',
            'balls': 'Balls faced',
            'fours': 'Number of 4s hit',
            'sixes': 'Number of 6s hit',
            'strikeRate': 'Strike rate of batsman',
            'isOut': 'Whether batsman got out',
            'wicketType': 'Type of dismissal',
            'fielders': 'Fielders involved in dismissal',
            'bowler': 'Bowler who dismissed the batsman'
        }
        for col, desc in batting_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(batting.describe())
    # ================= Bowling Card =================
    with st.expander("2️⃣ Bowling Card ", expanded=False):
        st.write("**Shape:**", bowling.shape)
        st.dataframe(bowling.dropna().head())

        st.subheader("Column Descriptions:")
        bowling_dict = {
            'Match ID': 'Unique Match Identifier',
            'innings': 'Innings number',
            'team': 'Bowling team',
            'opposition': 'Batting opposition team',
            'bowler id': 'Unique ID of bowler',
            'overs': 'Overs bowled',
            'balls': 'Balls bowled',
            'maidens': 'Maiden overs',
            'conceded': 'Total runs conceded',
            'wickets': 'Wickets taken',
            'economy': 'Economy rate',
            'dots': 'Dot balls bowled',
            'fours': 'Fours conceded',
            'sixes': 'Sixes conceded',
            'wides': 'Wide balls bowled',
            'noballs': 'No balls bowled'
        }
        for col, desc in bowling_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(bowling.describe())
    # ================= Fall of Wickets =================
    with st.expander("3️⃣ Fall of Wickets ", expanded=False):
        st.write("**Shape:**", fow.shape)
        st.dataframe(fow.dropna().head())

        st.subheader("Column Descriptions:")
        fow_dict = {
            'Match ID': 'Unique Match Identifier',
            'innings': 'Innings number',
            'team': 'Batting team',
            'player': 'Batsman dismissed',
            'wicket': 'Wicket number (1,2,...10)',
            'overruns': 'Team score at fall of wicket'
        }
        for col, desc in fow_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(fow.describe())
    # ================= Matches Data =================
    with st.expander("4️⃣ Matches Data ", expanded=False):
        st.write("**Shape:**", matches.shape)
        st.dataframe(matches.dropna().head())

        st.subheader("Column Descriptions:")
        matches_dict = {
            'ODI Match No': 'ODI Match Number',
            'Match ID': 'Unique Match Identifier',
            'Match Name': 'Name of the match',
            'Series ID': 'Series ID',
            'Series Name': 'Name of the series',
            'Match Date': 'Date of the match',
            'Match Format': 'Format of match',
            'Team1 ID': 'ID for Team 1',
            'Team1 Name': 'Name of Team 1',
            'Team1 Captain': 'Captain of Team 1',
            'Team1 Runs Scored': 'Runs scored by Team 1',
            'Team1 Wickets Fell': 'Wickets lost by Team 1',
            'Team1 Extras Rec': 'Extras conceded to Team 1',
            'Team2 ID': 'ID for Team 2',
            'Team2 Name': 'Name of Team 2',
            'Team2 Captain': 'Captain of Team 2',
            'Team2 Runs Scored': 'Runs scored by Team 2',
            'Team2 Wickets Fell': 'Wickets lost by Team 2',
            'Team2 Extras Rec': 'Extras conceded to Team 2',
            'Match Venue (Stadium)': 'Stadium name',
            'Match Venue (City)': 'City name',
            'Match Venue (Country)': 'Country name',
            'Umpire 1': 'First umpire',
            'Umpire 2': 'Second umpire',
            'Match Referee': 'Referee',
            'Toss Winner': 'Team who won toss',
            'Toss Winner Choice': 'Bat or Bowl decision',
            'Match Winner': 'Team who won the match',
            'Match Result Text': 'Match result description',
            'MOM Player': 'Man of the Match',
            'Team1 Playing 11': 'Team1 playing 11 players',
            'Team2 Playing 11': 'Team2 playing 11 players',
            'Debut Players': 'Debutants in match'
        }
        for col, desc in matches_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(matches.describe())
    # ================= Partnership Card =================
    with st.expander("5️⃣ Partnership Card ", expanded=False):
        st.write("**Shape:**", partnership.shape)
        st.dataframe(partnership.dropna().head())

        st.subheader("Column Descriptions:")
        partnership_dict = {
            'Match ID': 'Unique Match Identifier',
            'innings': 'Innings number',
            'for wicket': 'Wicket partnership was for',
            'team': 'Batting team',
            'opposition': 'Opposition team',
            'player1': 'First batsman',
            'player2': 'Second batsman',
            'player1 runs': 'Runs by player1',
            'player2 runs': 'Runs by player2',
            'player1 balls': 'Balls faced by player1',
            'player2 balls': 'Balls faced by player2',
            'partnership runs': 'Total partnership runs',
            'partnership balls': 'Total partnership balls'
        }
        for col, desc in partnership_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(partnership.describe())
    # ================= Players Info =================
    with st.expander("6️⃣ Players Info ", expanded=False):
        st.write("**Shape:**", players.shape)
        st.dataframe(players.dropna().head())

        st.subheader("Column Descriptions:")
        players_dict = {
            'player_id': 'Unique player ID',
            'player_object_id': 'Player object ID',
            'player_name': 'Full name of player',
            'dob': 'Date of birth',
            'dod': 'Date of death (if any)',
            'gender': 'Gender',
            'batting_style': 'Batting style',
            'bowling_style': 'Bowling style',
            'country_id': 'Country ID'
        }
        for col, desc in players_dict.items():
            st.write(f"**{col}**: {desc}")
        st.write(players.describe())
        missing_summary = pd.DataFrame({
        "Dataset": [
            "Batting Card",
            "Bowling Card",
            "FOW Card",
            "Matches Data",
            "Partnership Card",
            "Players Info"
        ],
        "Rows": [
            batting.shape[0],
            bowling.shape[0],
            fow.shape[0],
            matches.shape[0],
            partnership.shape[0],
            players.shape[0]
        ],
        "Columns": [
            batting.shape[1],
            bowling.shape[1],
            fow.shape[1],
            matches.shape[1],
            partnership.shape[1],
            players.shape[1]
        ],
        "Total Missing Values": [
            batting.isnull().sum().sum(),
            bowling.isnull().sum().sum(),
            fow.isnull().sum().sum(),
            matches.isnull().sum().sum(),
            partnership.isnull().sum().sum(),
            players.isnull().sum().sum()
        ]
    })

    # Add missing value % column
    missing_summary["Missing %"] = (
        missing_summary["Total Missing Values"] / (missing_summary["Rows"] * missing_summary["Columns"])
    ).round(2) * 100

    # Display in Streamlit
    st.header("📊 Overall Missing Data Summary")
    st.dataframe(missing_summary)
    # Combined Summary
    st.header("📊 Dataset Summary")
    st.write(f"✅ Total Batting Records: {len(batting):,}")
    st.write(f"✅ Total Bowling Records: {len(bowling):,}")
    st.write(f"✅ Total FOW Records: {len(fow):,}")
    st.write(f"✅ Total Match Records: {len(matches):,}")
    st.write(f"✅ Total Partnership Records: {len(partnership):,}")
    st.write(f"✅ Total Players Records: {len(players):,}")

elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,tab12 = st.tabs([
        "Outlier Detection",
        "Balls vs Runs",
        "Runs Distribution",
        "Batting Correlation",
        "Matches Per Year",
        "Fall of Wickets",
        "Strike Rate Distribution",
        "Top Bowlers",
        "Extras",
        "Partnership Runs",
        "Top MOM & Partnerships",
        "Pairwise Relation"
    ])

    with tab1:
        st.subheader("Outlier Detection (Runs Scored)")
        st.write("This plot helps us spot extreme performances in batting — for example, very high individual scores that deviate from the general trend.")
        st.pyplot(figs['strip'])

    with tab2:
        st.subheader("Balls vs Runs")
        st.write("This scatterplot shows the relationship between balls faced and runs scored across players and teams. Usually, higher balls lead to higher scores, but aggressive players may score faster.")
        st.pyplot(figs['scatter'])

    with tab3:
        st.subheader("Runs Distribution")
        st.write("The histogram displays how frequently different ranges of individual runs occur. Most innings usually fall in lower to mid-range scores.")
        st.pyplot(figs['runs'])

    with tab4:
        st.subheader("Batting Correlation Heatmap")
        st.write("This heatmap shows correlations between batting performance metrics such as runs, balls, strike rate, fours and sixes. High positive correlation between runs and strike rate is expected.")
        st.pyplot(figs['corr'])

    with tab5:
        st.subheader("Matches Played Per Year")
        st.write("This line chart shows the number of ODI matches played each year, helping to observe trends in match frequency over time.")
        st.pyplot(figs['year'])

    with tab6:
        st.subheader("Fall of Wickets")
        st.write("This count plot shows at which wicket numbers dismissals occur, giving a general pattern of team collapses or stability.")
        st.pyplot(figs['fow'])

    with tab7:
        st.subheader("Strike Rate Distribution")
        st.write("Distribution of strike rates across batsmen. This helps us understand general scoring speeds — whether ODIs are trending towards more aggressive batting.")
        st.pyplot(figs['strike'])

    with tab8:
        st.subheader("Top Bowlers by Wickets")
        st.write("The top wicket-taking bowlers across the dataset. This provides insight into dominant bowlers in ODI cricket history.")
        st.pyplot(figs['top_bowlers'])

    with tab9:
        st.subheader("Extras (Wides & No Balls)")
        st.write("This bar chart highlights how many extra runs bowlers are conceding via wides and no balls — an important factor in disciplined bowling.")
        st.pyplot(figs['extras'])

    with tab10:
        st.subheader("Partnership Runs Distribution")
        st.write("Distribution of total partnership runs. Larger partnerships often lead to match-winning scores, while lower partnerships indicate frequent wicket loss.")
        st.pyplot(figs['partnership'])

    with tab11:
        st.subheader("Top Man of the Match Players")
        st.write("Top players who have earned the most 'Man of the Match' awards, indicating consistent match-winning performances.")
        top_mom = matches['MOM Player'].value_counts().head(10)
        st.bar_chart(top_mom)

        st.subheader("Top Partnerships")
        st.write("This table displays the highest run-scoring partnerships in ODI history captured in the dataset.")
        top_partnerships = partnership.sort_values('partnership runs', ascending=False).head(10)
        st.dataframe(top_partnerships[['player1_name','player2_name','partnership runs']])
    
    with tab12:
        batting_numeric = batting[['runs', 'balls', 'fours', 'sixes', 'strikeRate']].dropna()
        fig_bat = sns.pairplot(batting_numeric, diag_kind='kde', corner=True)
        st.pyplot(fig_bat)
elif page == "ML Model":
    st.title("📊 Model Evaluation & Performance")

    # Divider for clarity
    st.divider()

    col1, col2 = st.columns(2)

    # Toss Decision Model Evaluation
    with col1:
        st.subheader("🎯 Toss Decision Model")
        
        st.markdown(f"**✅ Accuracy:** `{model_results['toss_acc']*100:.2f}%`")
        
        # Confusion Matrix
        cm_toss = confusion_matrix(model_results['toss_y_test'], model_results['toss_pred'])
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm_toss, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, square=True)
        ax.set_title("Confusion Matrix", fontsize=12)
        st.pyplot(fig)

        # Classification Report
        st.markdown("**📄 Classification Report:**")
        st.code(classification_report(model_results['toss_y_test'], model_results['toss_pred']), language='text')

    # Match Winner Model Evaluation
    with col2:
        st.subheader("🏆 Match Winner Model")
        
        st.markdown(f"**✅ Accuracy:** `{model_results['match_acc']*100:.2f}%`")
        
        cm_match = confusion_matrix(model_results['match_y_test'], model_results['match_pred'])
        fig2, ax2 = plt.subplots(figsize=(4,5))
        sns.heatmap(cm_match, annot=True, fmt='d', cmap='Oranges', ax=ax2, cbar=False, square=True)
        ax2.set_title("Confusion Matrix", fontsize=12)
        st.pyplot(fig2)

        st.markdown("**📄 Classification Report:**")
        st.code(classification_report(model_results['match_y_test'], model_results['match_pred']), language='text')

    # Another divider for nice closure
    st.divider()
elif page == "Prediction":
    st.title("Predict Toss Decision & Match Winner")

    teams_all = pd.concat([matches['Team1 Name'], matches['Team2 Name']]).dropna().unique()
    venue_list = sorted(matches['Match Venue (Stadium)'].dropna().unique())

    st.subheader("Input Match Details")

    team1 = st.selectbox("Select Team 1", sorted(teams_all))
    team2 = st.selectbox("Select Team 2", sorted([t for t in teams_all if t != team1]))
    venue = st.selectbox("Select Venue", venue_list)
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("If predicting Match Winner, Toss Decision?", ["bat", "bowl"])

    st.markdown("---")

    # Predict Toss Decision
    toss_input = pd.DataFrame([{
        'Team1 Name': team1,
        'Team2 Name': team2,
        'Match Venue (Stadium)': venue,
        'Toss Winner': toss_winner
    }])
    toss_input_encoded = pd.get_dummies(toss_input)
    toss_input_encoded = toss_input_encoded.reindex(columns=model_results['toss_X_columns'], fill_value=0)

    toss_pred_result = model_results['toss_model'].predict(toss_input_encoded)[0]
    st.success(f"Predicted Toss Decision: **{toss_pred_result}**")

    # Predict Match Winner
    match_input = pd.DataFrame([{
        'Team1 Name': team1,
        'Team2 Name': team2,
        'Match Venue (Stadium)': venue,
        'Toss Winner': toss_winner,
        'Toss Winner Choice': toss_decision
    }])
    match_input_encoded = pd.get_dummies(match_input)
    match_input_encoded = match_input_encoded.reindex(columns=model_results['match_X_columns'], fill_value=0)

    match_pred_result = model_results['match_model'].predict(match_input_encoded)[0]
    st.success(f"Predicted Match Winner: **{match_pred_result}**")

elif page == "Conclusion":
    st.title("📊 Conclusion")

    st.markdown("""
    ### 🔍 Summary

    In this project, we successfully applied machine learning techniques to predict:

    - The decision of the toss winner (bat or field).
    - The winner of the match itself.

    Using Random Forest classifiers, we achieved an accuracy of approximately **70%** on both tasks. The models were trained and evaluated on historical match data, taking into account factors such as teams, venues, and toss outcomes.

    ### 📈 Key Outcomes

    - Random Forest provided strong performance and handled categorical features effectively.
    - The system allows users to input new match details and get live predictions.
    - The modular design makes it easy to update with new data for improved performance in the future.

    ### 🚀 Future Improvements

    - Include additional features such as player statistics, team rankings, and weather conditions to further boost model accuracy.
    - Implement hyperparameter tuning and feature engineering for better generalization.
    - Deploy the model for real-time predictions during live matches.

    ---

    Thank you for exploring this application!
    """)
