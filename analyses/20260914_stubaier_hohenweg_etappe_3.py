# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 3: herstelhartslag (HRR) in rustpauzes
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 3 (14 september 2026)
# MAGIC
# MAGIC **Vraag:** Hoe snel zakt mijn hartslag tijdens een rustpauze?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. Een meting geldt als stilstand wanneer de snelheid onder 1,62 km/u (0,45 m/s) ligt.
# MAGIC 2. Aaneengesloten stilstand van minstens 30 seconden geldt als rustpauze.
# MAGIC 3. De herstelhartslag is de hartslag aan het begin van de pauze minus de hartslag 60 seconden later. Is de pauze korter dan 60 seconden, dan geldt de laagste hartslag in de pauze.
# MAGIC 4. Herstel van minder dan 5 slagen per minuut wordt niet getoond.
# MAGIC 5. In de grafiek blijft de hartslag tijdens stilstand weg, evenals reeksen van meer dan 30 identieke waarden (sensorstoring).
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

from utils.berekeningen import (
    bereken_beweegtijden,
    bereken_herstelhartslag,
    maak_hartslagreeks_voor_weergave,
    markeer_pauzes,
)
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_herstelhartslag
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 3
FIT_BESTAND = PADEN.fit_map / "20260914_stubaier_hohenweg_etappe_3.fit"
MIN_PAUZE_S = 30                              
HERSTEL_VENSTER_S = ANALYSE.hrr_venster_s     

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
# MAGIC ## Berekening: pauzes en herstelhartslag

# COMMAND ----------

df = markeer_pauzes(df, drempel_kmh=ANALYSE.hrr_stilstand_kmh, min_duur_s=MIN_PAUZE_S)
df["hartslag_glad"] = maak_hartslagreeks_voor_weergave(df)

pauzes = bereken_herstelhartslag(df, min_pauze_s=MIN_PAUZE_S, venster_s=HERSTEL_VENSTER_S)
tijden = bereken_beweegtijden(df)

print(f"{len(pauzes)} rustpauzes met herstel gevonden.")
print(pauzes[["start_minuut", "duur_min", "hartslag_start", "herstel_bpm"]].round(1).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_herstelhartslag(df, pauzes, tijden, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260914_heart_rate_recovery", formaat="liggend")