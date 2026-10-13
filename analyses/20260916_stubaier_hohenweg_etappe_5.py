# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 5: cardiac drift en hoogte-impact
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 5 (16 september 2026)
# MAGIC
# MAGIC **Vraag:** Loopt mijn hartslag op bij gelijke inspanning naarmate de dag vordert?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. Rustpauzes (stilstand onder 0,5 km/u gedurende minstens 150 seconden) worden verwijderd, zodat alleen actieve tijd overblijft.
# MAGIC 2. De snelheid wordt gecorrigeerd voor hoogtewinst: de stijgsnelheid wordt omgerekend naar km/u en met factor 10 meegeteld (aanname: 1 m stijgen weegt als 10 m horizontaal).
# MAGIC 3. De actieve tijd wordt in twee gelijke helften gesplitst.
# MAGIC 4. Per helft is de efficiëntiefactor (EF) de gemiddelde hartslag gedeeld door de gemiddelde gecorrigeerde snelheid.
# MAGIC 5. Cardiac drift = (EF helft 2 - EF helft 1) / EF helft 1 x 100.
# MAGIC
# MAGIC **Let op:** de hoogtecorrectie is een aanname en niet wetenschappelijk gevalideerd. De uitkomst is bedoeld voor vergelijking tussen etappes, niet als absolute maat. Klimt een etappe vooral in één helft, dan beïnvloedt dat de uitkomst sterk.
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

from utils.berekeningen import bereken_cardiac_drift
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_cardiac_drift
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, maak_tijdraster, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 5
FIT_BESTAND = PADEN.fit_map / "20260916_stubaier_hohenweg_etappe_5.fit"
FILTER_VENSTER = 601   # Savitzky-Golay-venster in seconden (10 minuten)
FILTER_POLYORDE = 1    # Graad van het filter

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
# MAGIC ## Berekening: tijdraster en cardiac drift

# COMMAND ----------

raster = maak_tijdraster(df, max_gat_s=ANALYSE.max_gat_s)
actief, resultaat = bereken_cardiac_drift(raster, venster=FILTER_VENSTER, polyorder=FILTER_POLYORDE)

print(f"Gem. hartslag helft 1: {resultaat['gem_hartslag_helft1']:.0f} bpm")
print(f"Gem. hartslag helft 2: {resultaat['gem_hartslag_helft2']:.0f} bpm")
print(f"Gecorrigeerde drift:   {resultaat['drift_pct']:+.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_cardiac_drift(actief, resultaat, etappe_nr=ETAPPE_NR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren

# COMMAND ----------

exporteer_afbeelding(fig, bestandsnaam="20260916_cardiac_drift", formaat="liggend")