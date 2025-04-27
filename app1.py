import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io

# Configuration de la page
st.set_page_config(page_title="Prévision Multi-Séries avec Filtres", layout="wide")

st.title("📈 Application de Prévision Multi-Séries avec Filtres (Prophet)")

# 1. Upload du fichier CSV
uploaded_file = st.file_uploader("Téléversez votre fichier CSV :", type=['csv'])

if uploaded_file is not None:
    # Lire les données
    data = pd.read_csv(uploaded_file, delimiter=';', encoding='utf-8')

    # Vérification
    required_columns = ['Rez Class', 'Seg Arr Port', 'Seg Dep Port', 'Demande', 'ds']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Le fichier doit contenir les colonnes suivantes : {', '.join(required_columns)}")
    else:
        # Conversion de la colonne 'ds' en datetime
        data['ds'] = pd.to_datetime(data['ds'], format='%Y-%m', errors='coerce')

        # Suppression des lignes avec des dates invalides
        data = data.dropna(subset=['ds'])

        # Sélectionner les filtres
        st.sidebar.header("Filtres de sélection")

        rez_class = st.sidebar.selectbox("Sélectionnez la Rez Class :", sorted(data['Rez Class'].dropna().unique()))
        seg_arr_port = st.sidebar.selectbox("Sélectionnez le Seg Arr Port :", sorted(data['Seg Arr Port'].dropna().unique()))
        seg_dep_port = st.sidebar.selectbox("Sélectionnez le Seg Dep Port :", sorted(data['Seg Dep Port'].dropna().unique()))

        # Filtrer les données
        filtered_data = data[
            (data['Rez Class'] == rez_class) &
            (data['Seg Arr Port'] == seg_arr_port) &
            (data['Seg Dep Port'] == seg_dep_port)
        ]

        if filtered_data.empty:
            st.warning("Aucune donnée trouvée pour cette combinaison. Essayez d'autres sélections.")
        else:
            st.subheader(f"Prévision pour {rez_class} | {seg_dep_port} ➔ {seg_arr_port}")

            # Regrouper par date (au cas où il y aurait plusieurs enregistrements par mois)
            ts = filtered_data.groupby('ds')['Demande'].sum().reset_index()

            # Renommer les colonnes pour Prophet
            ts = ts.rename(columns={'ds': 'ds', 'Demande': 'y'})

            # Période de prévision
            periods = st.slider("Nombre de mois à prévoir :", 1, 24, 12)

            # Créer et entraîner le modèle
            model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
            model.fit(ts)

            # Créer un futur DataFrame
            future = model.make_future_dataframe(periods=periods, freq='M')

            # Faire la prévision
            forecast = model.predict(future)

            # Afficher le graphe
            fig = model.plot(forecast)
            plt.title(f'Prévision de la Demande', fontsize=16)
            plt.xlabel('Date')
            plt.ylabel('Demande')

            # Convertir le graphique en image pour téléchargement
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)

            # Ajouter un bouton pour télécharger l'image
            st.download_button(
                label="Télécharger le graphique",
                data=buf,
                file_name=f"prevision_{rez_class}_{seg_dep_port}_to_{seg_arr_port}.png",
                mime="image/png"
            )

            # Afficher les prévisions futures
            st.write("Prévisions futures :")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

else:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")
