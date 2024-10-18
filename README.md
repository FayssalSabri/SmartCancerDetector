<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CancerPredictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        a {
            color: #007bff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>CancerPredictor</h1>

    <h2>Description</h2>
    <p>
        CancerPredictor est une application web interactive développée avec Streamlit, conçue pour prédire le type de cancer (Cancer du Sein, Cancer de la Peau, Cancer du Poumon) à partir de caractéristiques cellulaires. L'application utilise des modèles de machine learning pour fournir des prédictions basées sur les données saisies par l'utilisateur.
    </p>

    <h2>Fonctionnalités</h2>
    <ul>
        <li><strong>Sélection de type de cancer</strong> : Choisissez parmi Cancer du Sein, Cancer de la Peau, ou Cancer du Poumon.</li>
        <li><strong>Prédiction</strong> : Entrez les caractéristiques cellulaires via des curseurs pour obtenir des prédictions sur la nature bénigne ou maligne.</li>
        <li><strong>Visualisation</strong> : Affichez les résultats sous forme de graphique radar et visualisez les performances du modèle avec des courbes ROC et des matrices de confusion.</li>
        <li><strong>Données d'entrée</strong> : Basé sur des données collectées à partir de la base de données KAGGLE.</li>
    </ul>

    <h2>Technologies Utilisées</h2>
    <ul>
        <li>Python</li>
        <li>Streamlit</li>
        <li>Scikit-learn</li>
        <li>Plotly</li>
        <li>Pandas</li>
        <li>NumPy</li>
        <li>Matplotlib</li>
        <li>Seaborn</li>
    </ul>

    <h2>Installation</h2>
    <ol>
        <li>Clonez le dépôt :
            <pre><code>git clone https://github.com/votre-utilisateur/CancerPredictor.git
cd CancerPredictor</code></pre>
        </li>
        <li>Installez les dépendances :
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Lancez l'application :
            <pre><code>streamlit run app.py</code></pre>
        </li>
    </ol>

    <h2>Utilisation</h2>
    <ol>
        <li>Sélectionnez le type de cancer dans la barre latérale.</li>
        <li>Ajustez les curseurs pour entrer les caractéristiques cellulaires.</li>
        <li>Cliquez sur "Commencer" pour obtenir la prédiction.</li>
        <li>Visualisez les résultats et les performances du modèle.</li>
    </ol>

    <h2>Avertissement</h2>
    <p>
        Les résultats fournis par cette application sont basés sur des modèles statistiques et ne doivent pas remplacer un diagnostic professionnel. Consultez toujours un professionnel de la santé qualifié pour un diagnostic précis.
    </p>

    <h2>Contribuer</h2>
    <p>
        Les contributions sont les bienvenues ! Pour contribuer, veuillez suivre ces étapes :
    </p>
    <ol>
        <li>Forkez le projet.</li>
        <li>Créez votre branche (<code>git checkout -b feature-nouvelle-fonctionnalité</code>).</li>
        <li>Commitez vos modifications (<code>git commit -m 'Ajout d'une nouvelle fonctionnalité'</code>).</li>
        <li>Poussez votre branche (<code>git push origin feature-nouvelle-fonctionnalité</code>).</li>
        <li>Ouvrez une Pull Request.</li>
    </ol>

    <h2>Licence</h2>
    <p>
        Ce projet est sous licence MIT. Voir le fichier <a href="LICENSE">LICENSE</a> pour plus d'informations.
    </p>

    <h2>Contact</h2>
    <p>
        Pour toute question, vous pouvez me contacter à <a href="mailto:votre.email@example.com">votre.email@example.com</a>.
    </p>
</body>
</html>
