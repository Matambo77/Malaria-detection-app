import streamlit as st
import pandas as pd
import numpy as np
import joblib



# CONFIGURATION

st.set_page_config(
    page_title="Détection du paludisme sévère",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLE

st.markdown(
    """
    <style>

    .main {
        background-color: #f0f4f8;
    }

    .block-container {
        padding-top: 1.5rem;
    }

    .hero {
        padding: 2rem 2.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #0f3460, #16537e, #1f8a8a);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }

    .hero h1 {
        margin-bottom: .3rem;
    }

    /* Cartes résultat */
    .result-card {
        padding: 0.7rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1rem;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    .result-positive {
        background: linear-gradient(135deg, #ff4b4b, #c81e1e);
        color: white;
    }

    .result-negative {
        background: linear-gradient(135deg, #21c07a, #0d8a4f);
        color: white;
    }

    /* Boutons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161244;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161244;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

.sidebar-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100px;
    padding: 1rem;
    background-color: #161244;
    z-index: 999;
    box-sizing: border-box;
}
section[data-testid="stSidebar"] {
    position: relative;
}

    </style>
    """,
    unsafe_allow_html=True
)

# FICHIERS


MODEL_PATH = "pipeline_lg.pkl"
DATASET_PATH = "Dataset_Nigeria.csv"

# CHARGEMENT DU MODELE

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model

# CHARGEMENT DU DATASET

@st.cache_data
def load_data():

    try:

        return pd.read_csv(DATASET_PATH)

    except FileNotFoundError:

        return None


# SIDEBAR

st.sidebar.title("🩺Detection du Paludisme")

st.sidebar.caption(
    "Application Web pour la détection du paludisme."
)


# CHARGEMENT DU MODELE


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Impossible de charger pipeline_lg.pkl : {e}"
    )

    st.stop()


# CHARGEMENT DES DONNEES


df = load_data()

if df is None:

    st.markdown(
        """
        <div class="hero">

            <h1>🩺 Détection du paludisme sévère</h1>

            <p>
                Application de prédiction basée sur
                une régression logistique.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "Dataset_Nigeria.csv est introuvable. "
        "Placez-le dans le même dossier que app.py "
        "ou importez-le depuis la barre latérale."
    )

    st.stop()


# TITRE

st.warning(
    "Cette application est un prototype académique "
    "de machine learning. Elle ne remplace pas "
    "un diagnostic médical professionnel."
)
# RECUPERATION DES VARIABLES DU MODELE

def get_model_features(model):

    """
    Récupère les variables utilisées lors
    de l'entraînement du pipeline.
    """

    # Cas 1 : le pipeline possède directement
    # feature_names_in_
    if hasattr(model, "feature_names_in_"):

        return list(model.feature_names_in_)


    # Cas 2 : recherche dans les étapes du pipeline
    if hasattr(model, "named_steps"):

        for name, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):

                return list(step.feature_names_in_)


    # Si on ne trouve pas les variables
    return None


features = get_model_features(model)

# VERIFICATION DES VARIABLES

if features is None:

    st.error(
        "Impossible de récupérer automatiquement "
        "les noms des variables du modèle."
    )

    st.stop()

# VERIFICATION DATASET

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]


if missing_features:

    st.error(
        "Certaines variables utilisées par le modèle "
        "ne sont pas présentes dans le dataset :"
    )

    st.write(missing_features)

    st.stop()

# FORMULAIRE PATIENT

st.subheader("Informations du patient")

st.write(
    "Saisissez les informations du patient "
)


input_data = {}


columns = st.columns(2)


for i, feature in enumerate(features):

    with columns[i % 2]:

        series = df[feature].dropna()

        # VARIABLE NUMERIQUE
        if pd.api.types.is_numeric_dtype(series):

            unique_values = sorted(
                series.unique().tolist()
            )


            # Variable binaire
            if len(unique_values) <= 2:

                input_data[feature] = st.selectbox(
                    feature,
                    unique_values
                )


            # Age
            elif feature.lower() == "age":

                input_data[feature] = st.number_input(
                    "Âge",
                    min_value=0.0,
                    max_value=120.0,
                    value=float(series.median()),
                    step=1.0
                )


            # Autres variables numériques
            else:

                input_data[feature] = st.number_input(
                    feature,
                    value=float(series.median())
                )


        # VARIABLE CATEGORIELLE

        else:

            values = sorted(
                series.astype(str).unique().tolist()
            )


            if len(values) <= 30:

                input_data[feature] = st.selectbox(
                    feature,
                    values
                )

            else:

                mode = series.astype(str).mode()

                default_value = (
                    mode.iloc[0]
                    if not mode.empty
                    else ""
                )

                input_data[feature] = st.text_input(
                    feature,
                    value=default_value
                )


# BOUTON PREDICTION
st.divider()
seuil = st.slider(
    "Seuil de décision (probabilité à partir de laquelle le cas est considéré positif)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

if st.button(
    "🔍 Lancer la prédiction",
    type="primary"
):

    # Création du DataFrame
    # avec exactement les variables
    # attendues par le modèle

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )


    try:
   # PREDICTION et PROBABILITE


        probability = None


        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(input_df)[0][1]
            prediction = 1 if probability >= seuil else 0
        else:
            prediction = model.predict(input_df)[0]
            st.write(f"Valeur brute : {probability}")
  # RESULTAT
        st.subheader(
            "📊 Résultat de la prédiction"
        )


        col1, col2 = st.columns(2)


        with col1:
          if int(prediction) == 1:
            st.markdown(
            '<div class="result-card result-positive">⚠️ CAS POSITIF</div>',
            unsafe_allow_html=True
          )
          else:
           st.markdown(
            '<div class="result-card result-negative">✅ CAS NÉGATIF</div>',
            unsafe_allow_html=True
          )


        with col2:

            if probability is not None:

                st.metric(
                    "Probabilité estimée",
                    f"{probability * 100:.1f}%"
                )

 # BARRE DE PROBABILITE
        if probability is not None:

            st.progress(
                float(probability)
            )


    except Exception as e:

        st.error(
            f"Erreur lors de la prédiction : {e}"
        )

# FOOTER

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <hr style="border-color: rgba(255,255,255,0.2); margin: 0.5rem 0;">
        <p style="font-size: 0.8rem; margin: 0.2rem 0;">Projet académique — Machine Learning / Streamlit</p>
        <p style="font-size: 0.8rem; margin: 0.2rem 0;">Modèle utilisé : Logistic Regression</p>
    </div>
    """,
    unsafe_allow_html=True
)
