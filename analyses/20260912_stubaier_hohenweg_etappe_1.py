# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 1: klim-intensiteit (VAM) versus inspanning
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 1 (12 september 2026)
# MAGIC
# MAGIC **Vraag:** Hoe snel klim ik, en welke hartslag hoort daarbij?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De VAM (Velocità Ascensionale Media) is de stijging in meters per uur: stijging / tijd x 3600.
# MAGIC 2. De VAM wordt berekend over een rollend tijdvenster van 5 minuten. Alleen positieve hoogteverschillen tellen als stijging.
# MAGIC 3. De VAM wordt alleen getoond waar de gemiddelde helling in dat venster minstens 1 % is. Op afdalingen en vlakke stukken blijft de lijn leeg.
# MAGIC 4. De hartslag staat als voortschrijdend gemiddelde over 60 metingen in dezelfde grafiek.
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

from utils.berekeningen import bereken_vam
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_vam_dashboard
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 1
FIT_BESTAND = PADEN.fit_map / "20260912_stubaier_hohenweg_etappe_1.fit"
VAM_VENSTER_S = 300  # Tijdvenster in seconden voor de VAM

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data inlezen

# COMMAND ----------

ruwe_data = lees_fit_bestand(FIT_BESTAND)
df = schoon_fit_data(ruwe_data, cadans_factor=ANALYSE.wandel_cadans_factor)
df = bereken_verstreken_tijd(df)

print(f"{len(df)} metingen ingelezen: {df['tijd'].min():%d-%m-%Y %H:%M} t/m {df['tijd'].max():%H:%M}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening: VAM

# COMMAND ----------

df_vam = bereken_vam(df, venster_s=VAM_VENSTER_S)

print(f"Totale stijging:  {df_vam['totale_stijging_m'].iloc[-1]:.0f} m")
print(f"Gemiddelde VAM:   {df_vam['vam_klim'].mean():.0f} m/u")
print(f"Maximale VAM:     {df_vam['vam_klim'].max():.0f} m/u")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_vam_dashboard(df_vam, venster_s=VAM_VENSTER_S, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260912_vam_etappe_1", formaat="liggend")