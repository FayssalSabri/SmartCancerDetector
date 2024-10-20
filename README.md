# CancerPredictor

## Description
**CancerPredictor** est une application web interactive développée avec **Streamlit**, conçue pour prédire le type de cancer (cancer du sein, cancer de la peau, cancer du poumon) à partir de caractéristiques cellulaires. L'application exploite des modèles de machine learning pour fournir des prédictions précises basées sur les données saisies par l'utilisateur.

![Capture d'écran de l'application](images/Home_page_1.png) 

## Fonctionnalités
- **Sélection de type de cancer** : Choisissez parmi cancer du sein, cancer de la peau, ou cancer du poumon.
- **Prédiction** : Entrez les caractéristiques cellulaires via des curseurs pour obtenir des prédictions sur la nature bénigne ou maligne des cellules.
![Graphique radar des résultats](images/Prédictions.png) 
![Graphique radar des résultats](images/Prédictions_2.png) 

- **Visualisation** : Affichez les résultats sous forme de graphique radar, et visualisez les performances du modèle à travers des courbes ROC et des matrices de confusion.

![matrice de confusion](images/Visualisation.png) 
![courbe ROC](images/Visualisation_2.png) 



## Technologies Utilisées
- **Langages** : Python
- **Frameworks** : Streamlit
- **Bibliothèques** : 
  - Scikit-learn
  - Plotly
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/FayssalSabri/SmartCancerDetector.git
   cd SmartCancerDetector
