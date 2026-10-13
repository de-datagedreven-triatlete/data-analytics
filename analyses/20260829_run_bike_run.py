# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Run-bike-run: opbouw van de wedstrijd
# MAGIC
# MAGIC **Activiteit:** run-bike-run van 29 augustus 2026
# MAGIC
# MAGIC **Vragen:**
# MAGIC 1. Hoe verloopt de hartslag over de wedstrijd, en waar liggen de wissels?
# MAGIC 2. Kost dezelfde snelheid in de tweede run meer hartslag dan in de eerste run?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De fases worden afgeleid uit de snelheid: fietsen is sneller dan 20 km/u, een wissel is stilstand (< 3 km/u) rond de fiets.
# MAGIC 2. Per run worden alleen metingen in een vergelijkbaar tempo- en hartslagbereik gebruikt (zie `ANALYSE.run_*` in `utils/config.py`).
# MAGIC 3. Door de punten van beide runs loopt een trendlijn van hartslag tegen snelheid. Ligt de lijn van run 2 hoger, dan kostte hetzelfde tempo meer hartslag.
# MAGIC
# MAGIC **Resultaat:** twee afbeeldingen in de map `output/`.

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

from utils.berekeningen import detecteer_triatlonfasen, selecteer_runs_voor_vergelijking
from utils.config import PADEN
from utils.grafieken import teken_race_opbouw, teken_run_vergelijking
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

FIT_BESTAND = PADEN.fit_map / "20260829_run-bike-run.fit"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data inlezen
# MAGIC
# MAGIC Het FIT-bestand wordt ingelezen en omgezet naar het vaste formaat (zie `utils/inlezen.py`).

# COMMAND ----------

ruwe_data = lees_fit_bestand(FIT_BESTAND)
df = schoon_fit_data(ruwe_data)
df = bereken_verstreken_tijd(df)
print(f"{len(df)} metingen ingelezen: {df['tijd'].min():%d-%m-%Y %H:%M} t/m {df['tijd'].max():%H:%M}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening: fases van de wedstrijd

# COMMAND ----------

fasen = detecteer_triatlonfasen(df)

for fase in fasen["fases"]:
    print(f"{fase['naam']:<6} {fase['start']:6.1f} - {fase['eind']:6.1f} min  ({fase['eind'] - fase['start']:5.1f} min)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek 1: opbouw van de wedstrijd

# COMMAND ----------

fig_opbouw = teken_race_opbouw(df, fasen["fases"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren van grafiek 1

# COMMAND ----------

exporteer_afbeelding(fig_opbouw, bestandsnaam="20260829_race_opbouw", formaat="liggend")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening en grafiek 2: run 1 versus run 2

# COMMAND ----------

run1, run2 = selecteer_runs_voor_vergelijking(df, fasen["fiets_start"], fasen["fiets_eind"])
print(f"Run 1: {len(run1)} metingen, gem. hartslag {run1['hartslag'].mean():.0f} bpm")
print(f"Run 2: {len(run2)} metingen, gem. hartslag {run2['hartslag'].mean():.0f} bpm")

fig_vergelijking = teken_run_vergelijking(run1, run2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren van grafiek 2

# COMMAND ----------

exporteer_afbeelding(fig_vergelijking, bestandsnaam="20260829_run1_vs_run2", formaat="staand")