# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 4: beweegratio (actief versus stilstand)
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 4 (15 september 2026)
# MAGIC
# MAGIC **Vraag:** Welk deel van de dag was ik daadwerkelijk in beweging?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De metingen worden omgezet naar een raster van één meting per seconde. Gaten in de opname (automatische pauze) tellen als stilstand.
# MAGIC 2. Een seconde geldt als stilstand wanneer de snelheid onder 2,0 km/u ligt.
# MAGIC 3. Alleen aaneengesloten stilstand van minstens 45 seconden geldt als rustpauze.
# MAGIC 4. De beweegratio is de tijd buiten rustpauzes gedeeld door de totale tijd.
# MAGIC 5. De snelheid in de grafiek is gladgestreken met een Savitzky-Golay-filter.
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

from utils.berekeningen import bereken_beweegratio
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_beweegratio
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, maak_tijdraster, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 4
FIT_BESTAND = PADEN.fit_map / "20260915_stubaier_hohenweg_etappe_4.fit"
STILSTAND_KMH = 2.0   
MIN_PAUZE_S = 45      

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
# MAGIC ## Berekening: tijdraster en beweegratio

# COMMAND ----------

raster = maak_tijdraster(df, max_gat_s=ANALYSE.max_gat_s)
raster, resultaat = bereken_beweegratio(raster, drempel_kmh=STILSTAND_KMH, min_pauze_s=MIN_PAUZE_S)

print(f"Totale tijd:  {resultaat['totale_tijd_s'] / 60:.0f} min")
print(f"In beweging:  {resultaat['beweging_s'] / 60:.0f} min ({resultaat['beweegratio_pct']:.1f}%)")
print(f"Stilstand:    {resultaat['stilstand_s'] / 60:.0f} min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_beweegratio(raster, resultaat, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260915_movingratio", formaat="liggend")