import streamlit as st
import numpy as np
import joblib

# Chargement du modèle
model = joblib.load('best_model_random_forest.pkl')

# Titre de l'application
st.title("🔍 Prédiction d'achat via publicité sur un réseau social")
st.write("Entrez les informations utilisateur pour prédire s’il achètera après avoir vu une pub.")

# Champs de saisie utilisateur
age = st.slider("Âge", min_value=18, max_value=70, value=30)
salaire = st.number_input("Salaire estimé (€)", min_value=1000, max_value=200000, step=1000, value=50000)

# Bouton de prédiction
if st.button("Prédire l'achat"):
    # Préparer les données sous forme de tableau 2D
    features = np.array([[age, salaire]])
    
    # Prédiction
    prediction = model.predict(features)[0]

    # Affichage du résultat
    if prediction == 1:
        st.success("✅ L'utilisateur est susceptible d'acheter le produit.")
    else:
        st.warning("❌ L'utilisateur n'est probablement pas intéressé par le produit.")
