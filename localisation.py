import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Pour charger le modèle entraîné
from sklearn.preprocessing import StandardScaler

# Charger le modèle et le scaler préalablement entraînés
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # Charge le scaler utilisé pendant l'entraînement
label_encoder = joblib.load("label_encoder.pkl")  # Pour décoder la prédiction
# Charger la liste des features utilisées à l'entraînement
feature_names = joblib.load("feature_names.pkl")

# Vérifier les colonnes attendues par le modèle
#expected_features = model.feature_names_in_
#print("Colonnes attendues :", expected_features)

# Interface utilisateur avec Streamlit
st.title("Prédiction des Zones de Tension pour Kinésithérapeutes")
st.write("Remplissez les informations du patient pour estimer la zone de tension.")

# Saisie des informations patient
age = st.number_input("Âge du patient", min_value=10, max_value=100, value=30)
taille = st.number_input("Taille (cm)", min_value=100, max_value=220, value=175)
poids = st.number_input("Poids (kg)", min_value=30, max_value=150, value=70)

# Sélection du sexe (remplace le 0/1)
sexe = st.selectbox("Sexe", ["Homme", "Femme"])
sexe_homme = 1 if sexe == "Homme" else 0

# Sélection du type de pied
type_pied = st.selectbox("Type de pied", ["Plat", "Creux", "Normal"])
type_pied_plat = 1 if type_pied == "Plat" else 0
type_pied_creux = 1 if type_pied == "Creux" else 0

# Sélection des déséquilibres posturaux (ajout de Genoux valgum)
desequilibre = st.multiselect("Déséquilibres posturaux", 
    ["Hyperlordose lombaire", "Cyphose dorsale", "Épaule antépulsée", "Genoux valgum", "Aucun"])

# Encodage des déséquilibres sélectionnés
des_bal = {
    "Hyperlordose lombaire": 1 if "Hyperlordose lombaire" in desequilibre else 0,
    "Cyphose dorsale": 1 if "Cyphose dorsale" in desequilibre else 0,
    "Épaule antépulsée": 1 if "Épaule antépulsée" in desequilibre else 0,
    "Genoux valgum": 1 if "Genoux valgum" in desequilibre else 0
}


# Saisie des distances anatomiques
dist_acromion_g = st.number_input("Distance Acromion Gauche", step=0.1)
dist_acromion_d = st.number_input("Distance Acromion Droit", step=0.1)
dist_eips_g = st.number_input("Distance EIPS Gauche", step=0.1)
dist_eips_d = st.number_input("Distance EIPS Droit", step=0.1)
dist_t4 = st.number_input("Distance T4", step=0.1)
dist_l1 = st.number_input("Distance L1", step=0.1)

# Création du DataFrame pour la prédiction
patient_data = pd.DataFrame({
    "Âge": [age],
    "Taille (cm)": [taille],
    "Poids (kg)": [poids],
    "Distance_Acromion_G": [dist_acromion_g],
    "Distance_Acromion_D": [dist_acromion_d],
    "Distance_EIPS_G": [dist_eips_g],
    "Distance_EIPS_D": [dist_eips_d],
    "Distance_T4": [dist_t4],
    "Distance_L1": [dist_l1],
    "Sexe_Homme": [sexe_homme],
    "Type de pied_Plat": [type_pied_plat],
    "Type de pied_Creux": [type_pied_creux],
    "Déséquilibres posturaux_Hyperlordose lombaire": [des_bal["Hyperlordose lombaire"]],
    "Déséquilibres posturaux_Cyphose dorsale": [des_bal["Cyphose dorsale"]],
    "Déséquilibres posturaux_Épaule antépulsée": [des_bal["Épaule antépulsée"]]
})

# Recréer les colonnes dans le bon ordre et remplir celles manquantes
patient_data = patient_data.reindex(columns=feature_names, fill_value=0)

# Normalisation des données
patient_data_scaled = scaler.transform(patient_data)

## Prédiction
if st.button("Prédire la zone de tension"):
        # Faire la prédiction avec les probabilités
    probas = model.predict_proba(patient_data_scaled)
    prediction = model.predict(patient_data_scaled)

    # Récupérer la localisation prédite et son score de confiance
    zone_predite = label_encoder.inverse_transform(prediction)[0]
    confiance = np.max(probas) * 100  # Convertir en pourcentage

    # Affichage du résultat principal
    st.success(f"Zone de tension probable : **{zone_predite}**")
    st.info(f"Précision de la prédiction : **{confiance:.2f}%**")

    # Affichage des scores pour toutes les classes
    st.subheader("Probabilités pour chaque zone de tension :")
    proba_dict = {label_encoder.inverse_transform([i])[0]: probas[0][i] * 100 for i in range(len(probas[0]))}

    # Afficher sous forme de barres de progression
    for zone, score in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
        st.progress(int(score))
        st.write(f"**{zone}** : {score:.2f}%")
