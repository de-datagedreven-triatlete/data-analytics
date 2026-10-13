# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 2: terreinimpact op de cadans
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 2 (13 september 2026)
# MAGIC
# MAGIC **Vraag:** Hoeveel daalt mijn cadans (stappen per minuut) naarmate het steiler wordt?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De helling is het hoogteverschil gedeeld door de afgelegde afstand tussen twee metingen, afgekapt tussen -30 % en +45 % (meetruis).
# MAGIC 2. De cadans die Garmin per been registreert, wordt verdubbeld tot stappen per minuut (`ANALYSE.wandel_cadans_factor`).
# MAGIC 3. Alleen klimmen vanaf 1 % met een geldige cadans (40 tot 220 spm) tellen mee.
# MAGIC 4. Door die punten loopt een rechte lijn (lineaire regressie). De richtingscoëfficiënt geeft de verandering in cadans per +10 % helling.
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

from utils.berekeningen import analyseer_helling_cadans
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_helling_cadans
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 2
FIT_BESTAND = PADEN.fit_map / "20260913_stubaier_hohenweg_etappe_2.fit"
MIN_HELLING_PCT = ANALYSE.klim_min_helling_pct  # Alleen klimmen vanaf deze helling

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
# MAGIC ## Berekening: helling en cadans

# COMMAND ----------

klim, resultaat = analyseer_helling_cadans(df, min_helling_pct=MIN_HELLING_PCT)

print(f"Je cadans verandert met {resultaat['verandering_per_10_pct']:+.1f} spm per +10% helling "
      f"(gebaseerd op {resultaat['aantal_punten']} metingen).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_helling_cadans(klim, resultaat, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260913_slope_vs_cadence_etappe_2", formaat="liggend")