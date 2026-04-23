import pandas as pd

df = pd.read_csv(r"C:\Users\niran\Downloads\matches.csv")
print(df.head())
df = df[['team1', 'team2', 'winner']]

# Drop missing values
df.dropna(inplace=True)

# Convert categorical data into numbers
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['team1'] = le.fit_transform(df['team1'])
df['team2'] = le.fit_transform(df['team2'])
df['winner'] = le.fit_transform(df['winner'])
from sklearn.model_selection import train_test_split

X = df[['team1', 'team2']]
y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
from sklearn.metrics import accuracy_score

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
import pickle

pickle.dump(rf, open("model.pkl", "wb"))
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("IPL Match Winner Prediction")

team1 = st.number_input("Team 1 (encoded value)")
team2 = st.number_input("Team 2 (encoded value)")

if st.button("Predict"):
    result = model.predict([[team1, team2]])
    st.write("Predicted Winner:", result)