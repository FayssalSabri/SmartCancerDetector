import streamlit as st
import pandas as pd
import pickle 
import numpy as np

def get_clean_data():
    df = pd.read_csv('data/data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    # Afficher les colonnes de l'input_data
    st.write("Colonnes dans input_data:", input_data.keys())
    
    # Ensure input_data is in the correct format
    if isinstance(input_data, pd.DataFrame):
        input_array = input_data.values.reshape(1, -1)  # Reshape if it's a DataFrame
    else:
        input_array = np.array(list(input_data.values())).reshape(1, -1)  # Fallback for other formats

    scaled_input_array = scaler.transform(input_array)

    prediction = model.predict(scaled_input_array)
    
    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    
    st.subheader("RÉSULTATS :")
    st.write("Le résultat de diagnostic est :")
    
    if prediction[0] == 0:
        st.write("**Bénin 😄**")
    else:
        st.write("**Malin 🤒**")

    st.write("Probabilité d'être bénin :", model.predict_proba(scaled_input_array)[0][0] * 100, "%")
    st.write("Probabilité d'être malin :", model.predict_proba(scaled_input_array)[0][1] * 100, "%")

def main():
    input_data = get_clean_data()
    add_predictions(input_data)

if __name__ == "__main__":
    main()
