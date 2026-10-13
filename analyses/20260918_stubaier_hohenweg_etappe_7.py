# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 7: hartslag, HRV en readiness
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 7 (18 september 2026)
# MAGIC
# MAGIC **Vraag:** Hoe verhoudt de belasting van deze etappe zich tot mijn herstel (HRV en readiness) over de dagen?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De hartslag en het hoogteprofiel van de etappe komen uit het FIT-bestand.
# MAGIC 2. De dagelijkse HRV (RMSSD, in ms) en readiness-score (0 tot 100) komen uit `data/herstel_hrv.csv`. Die waarden zijn handmatig overgenomen uit de horloge-app.
# MAGIC 3. De dag van de etappe wordt in de HRV- en readiness-grafiek gemarkeerd.
# MAGIC
# MAGIC **Resultaat:** een afbeelding in de map `output/`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installatie van de benodigde pakketten
# MAGIC
# MAGIC De pakketten staan in `requirements.txt`. Deze cel hoeft alleen de eerste keer na het opstarten van het cluster te draaien.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_hrv_readiness
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, lees_herstelgegevens, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 7
FIT_BESTAND = PADEN.fit_map / "20260918_stubaier_hohenweg_etappe_7.fit"
HERSTEL_BESTAND = PADEN.data_map / "herstel_hrv.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data inlezen

# COMMAND ----------

ruwe_data = lees_fit_bestand(FIT_BESTAND)
df = schoon_fit_data(ruwe_data, cadans_factor=ANALYSE.wandel_cadans_factor)
df = bereken_verstreken_tijd(df)
herstel = lees_herstelgegevens(HERSTEL_BESTAND)
print(f"Herstelgegevens voor dag {herstel['dag'].min()} t/m {herstel['dag'].max()} ingelezen.")
print(f"{len(df)} metingen ingelezen: {df['tijd'].min():%d-%m-%Y %H:%M} t/m {df['tijd'].max():%H:%M}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_hrv_readiness(df, herstel, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260918_hrv_readiness_impact", formaat="liggend")