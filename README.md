# CANCERPREDICTOR

*Accurate Cancer Type Predictions with Interactive Visual Insights*

![Last Commit](https://img.shields.io/badge/last%20commit-today-blue)
![Python](https://img.shields.io/badge/python-100%25-blue)
![Languages](https://img.shields.io/badge/languages-1-gray)

*Built with the tools and technologies:*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Models and Evaluation](#models-and-evaluation)
- [Contributing](#contributing)

---

## Overview

**CancerPredictor** is an interactive web application developed with **Streamlit** to predict the type of cancer (breast cancer, skin cancer, or lung cancer) based on cellular characteristics.  
It uses machine learning models to deliver accurate predictions based on user input data.

![Application Screenshot](images/Home_page_1.png)

---

## Features

- **Cancer Type Selection**: Choose between breast, skin, or lung cancer for prediction.  
- **Prediction Interface**: Use sliders to enter cellular characteristics and obtain predictions (benign or malignant).  
  ![Radar Chart of Predictions](images/Prédictions.png)  
  ![Radar Chart of Predictions](images/Prédictions_2.png)  

- **Visualization Tools**:  
  - Radar charts  
  - ROC curves  
  - Confusion matrices  
  ![Confusion Matrix](images/Visualisation.png)  
  ![ROC Curve](images/Visualisation_2.png)

---

## Technologies Used

- **Programming Language**: Python  
- **Framework**: Streamlit  
- **Libraries**:
  - Scikit-learn
  - Plotly
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn

---

## Installation

```bash
git clone https://github.com/FayssalSabri/SmartCancerDetector.git
cd SmartCancerDetector
pip install -r requirements.txt
```

To launch the app:

```bash
streamlit run app.py
```

---

## How It Works

Users select the cancer type and input features. The ML model then predicts if the cancer is benign or malignant.  
Visual outputs like radar charts, ROC curves, and confusion matrices help interpret model behavior.

---

## Models and Evaluation

CancerPredictor uses validated ML models trained on specialized datasets.  
Metrics used:
- Accuracy
- Precision
- Recall
- ROC-AUC

Visual tools help users evaluate classification quality for each cancer type.

---

## Contributing

Feel free to fork the repo and submit a pull request to add features, fix bugs, or enhance performance.