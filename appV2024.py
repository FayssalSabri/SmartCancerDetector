import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load model and predict results
def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = pd.DataFrame([list(input_data.values())], columns=[list(input_data.keys())])
    scaled_input_array = scaler.transform(input_array)


    prediction = model.predict(scaled_input_array)
    
    st.markdown("<br><br><br><br><br><br><br>", unsafe_allow_html=True)
    
    st.subheader("RÉSULTATS :")
    st.write("Le résultat de diagnostic est :")
    
    if prediction[0] == 0:
        st.write("**Bénin 😄**")
    else:
        st.write("**Malin 🤒**")

    st.write("Probabilité d'être bénin :", model.predict_proba(scaled_input_array)[0][0]*100, "%")
    st.write("Probabilité d'être malin :", model.predict_proba(scaled_input_array)[0][1]*100, "%")
    
# Function to add sidebar for input
def add_sidebar(data):
    st.sidebar.header("CARACTÉRISTIQUES CELLULAIRES")
    input_dict = {}
    column_names = data.columns[1:]  # Ignoring the diagnosis column
    sliders_labels = [(f"{column} (mean)", column) for column in column_names]

    for label, key in sliders_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

# Function to create radar chart
def get_radar_chart(input_data, scaler):
    input_data_scaled = get_scaled_values(input_data, scaler)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
                  'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 
                  'Fractal Dimension']

    values1 = [input_data_scaled['radius_mean'], input_data_scaled['texture_mean'], 
               input_data_scaled['perimeter_mean'], input_data_scaled['area_mean'], 
               input_data_scaled['smoothness_mean'], input_data_scaled['compactness_mean'], 
               input_data_scaled['concavity_mean'], input_data_scaled['concave points_mean'], 
               input_data_scaled['symmetry_mean'], input_data_scaled['fractal_dimension_mean']]

    values2 = [input_data_scaled['radius_se'], input_data_scaled['texture_se'], 
               input_data_scaled['perimeter_se'], input_data_scaled['area_se'], 
               input_data_scaled['smoothness_se'], input_data_scaled['compactness_se'], 
               input_data_scaled['concavity_se'], input_data_scaled['concave points_se'], 
               input_data_scaled['symmetry_se'], input_data_scaled['fractal_dimension_se']]

    values3 = [input_data_scaled['radius_worst'], input_data_scaled['texture_worst'], 
               input_data_scaled['perimeter_worst'], input_data_scaled['area_worst'], 
               input_data_scaled['smoothness_worst'], input_data_scaled['compactness_worst'], 
               input_data_scaled['concavity_worst'], input_data_scaled['concave points_worst'], 
               input_data_scaled['symmetry_worst'], input_data_scaled['fractal_dimension_worst']]

    values1 += values1[:1]
    values2 += values2[:1]
    values3 += values3[:1]
    categories += categories[:1]

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=categories,
        fill='toself',
        name='Valeur Moyenne'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories,
        fill='toself',
        name='Erreur Standard'
    ))
    fig.add_trace(go.Scatterpolar(
        r=values3,
        theta=categories,
        fill='toself',
        name='Pire Valeur'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True,
        title='Diagramme Radar des Caractéristiques du Cancer'
    )

    return fig

# Function to scale input values
def get_scaled_values(input_data, scaler):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_input_array = scaler.transform(input_array)
    
    return dict(zip(input_data.keys(), scaled_input_array.flatten()))

# Function to clean data
def get_clean_data():
    df = pd.read_csv('data/data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

# Function to select cancer type
def type_selection():
    selected_cancer = st.sidebar.selectbox("Sélectionnez le type de cancer", ["Breast Cancer", "Skin Cancer", "Lung Cancer"])
    return selected_cancer

# Function to display the description
def description():
    st.markdown("<h1 style='color: #ff5733; text-align: center;'>CancerPredictor</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #3366cc;'>Bienvenue dans l'application CancerPredictor!</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>Cette application vise à prédire le type de cancer (Breast Cancer, Skin Cancer, Lung Cancer) en se basant sur les caractéristiques des cellules.</p>
        <p>Sélectionnez le type de cancer dans le menu à gauche, ajustez les caractéristiques des cellules à l'aide des sliders, puis cliquez sur "Commencer" pour obtenir des prédictions.</p>
        <p>Les données utilisées dans cette application sont collectées à partir d'une base de données sur KAGGLE.</p>
        <p>Il est important de noter que les prédictions fournies par cette application sont basées sur des modèles statistiques et ne doivent pas remplacer un diagnostic professionnel.</p>
        <p>Consultez toujours un professionnel de la santé qualifié pour un diagnostic précis.</p>
        """
    , unsafe_allow_html=True)

    # Suggestions et conseils
    st.markdown("<h3 style='color: #3366cc;'>Conseils & Ressources</h3>", unsafe_allow_html=True)
    st.markdown("""
        - **Restez informé :** Suivez les dernières recherches sur le cancer et les traitements.
        - **Écoutez votre corps :** Si vous ressentez des symptômes inquiétants, consultez un médecin.
        - **Ressources utiles :**
            - [Cancer.org](https://www.cancer.org/)
            - [National Cancer Institute](https://www.cancer.gov/)
            - [Kaggle Datasets](https://www.kaggle.com/datasets)
    """, unsafe_allow_html=True)
    
    st.markdown("Cette application reste un outil d'aide à la décision qui peut aider les professionnels de la santé dans le processus de diagnostic, mais ne doit pas être utilisée comme substitut à un diagnostic professionnel.", unsafe_allow_html=True)

# Function to display cancer data based on selection
def display_cancer_data(cancer_type):
    if cancer_type == 'Breast Cancer':
        data = get_clean_data()
        scaler = pickle.load(open("model/scaler.pkl", "rb"))
        input_data = add_sidebar(data)
        st.title("Breast Cancer Predictor")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = get_radar_chart(input_data, scaler)
            st.plotly_chart(radar_chart)
        with col2:
            add_predictions(input_data)
    elif cancer_type == 'Skin Cancer' or cancer_type == 'Lung Cancer':
        st.title(f"{cancer_type} Predictor")
        st.subheader(f"Les données pour {cancer_type} ne sont pas prêtes pour le moment.")
    
    # Button to go back to the home page
    if st.button('Retour'):
        st.session_state.selected_cancer = None
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# Function to visualize model performance
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Supposons que get_clean_data() soit déjà définie

def model_visualization():
    st.title("Visualisation des performances du modèle")
    
    # Charger les données nettoyées
    df = get_clean_data()

    # Séparation des données en features et target
    X = df.drop('diagnosis', axis=1)  # Les features
    y = df['diagnosis']  # La target (diagnostic)

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner un modèle de régression logistique
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Visualisation de la matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Vérités')
    ax.set_title('Matrice de Confusion')
    st.pyplot(fig)

    # Prédictions probabilistes pour la courbe ROC
    y_score = model.predict_proba(X_test)[:, 1]

    # Calcul de la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Visualisation de la courbe ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('Taux de Faux Positifs')
    ax.set_ylabel('Taux de Vrais Positifs')
    ax.set_title('Courbe ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)


# Main function to handle navigation and display pages
def main():
    st.sidebar.title("Navigation")
    
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Description"

    # Buttons for navigation
    if st.sidebar.button("Description"):
        st.session_state.selected_page = "Description"
    
    if st.sidebar.button("Démarrer la prédiction"):
        st.session_state.selected_page = "Prediction"
    
    if st.sidebar.button("Visualisation du modèle"):
        st.session_state.selected_page = "Visualisation du modèle"

    # Handle page selection
    if st.session_state.selected_page == "Description":
        description()
    elif st.session_state.selected_page == "Prediction":
        selected_cancer = type_selection()
        display_cancer_data(selected_cancer)
    elif st.session_state.selected_page == "Visualisation du modèle":
        model_visualization()

if __name__ == "__main__":
    main()
