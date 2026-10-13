# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Etappe 8: opgebouwde vermoeidheid in de cadans
# MAGIC
# MAGIC **Activiteit:** Stubaier Höhenweg, etappe 8 (19 september 2026)
# MAGIC
# MAGIC **Vraag:** Is mijn cadans op steile stukken lager dan in de eerste week (baseline)?
# MAGIC
# MAGIC **Methode:**
# MAGIC 1. De baseline is de gemiddelde klimcadans (stappen per minuut, helling vanaf 5 %) over de etappes van 12 t/m 18 september.
# MAGIC 2. Hoogte en cadans van de te onderzoeken etappe worden geïnterpoleerd naar een raster van 10 meter. Daarmee tellen stukken met veel metingen op dezelfde plek niet zwaarder.
# MAGIC 3. Het terrein wordt ingedeeld op helling: klimmen (vanaf +5 %), dalen (vanaf -5 %) of vlak.
# MAGIC 4. De cadans van het gekozen terreintype wordt gladgestreken met een Savitzky-Golay-filter en vergeleken met de baseline.
# MAGIC
# MAGIC **Reproduceerbaarheid:** de baseline wordt uit de FIT-bestanden berekend en opgeslagen in `data/klimcadans_per_etappe.csv`. Zijn de FIT-bestanden niet aanwezig, dan wordt dat csv-bestand gebruikt.
# MAGIC
# MAGIC **Eenheid:** de cadans staat in stappen per minuut (spm). In de eerste versie stond de cadans per been (rpm); dat is de helft.
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

import pandas as pd

from utils.berekeningen import (
    bereken_baseline,
    bereken_cadanstrend,
    bereken_klimcadans_per_bestand,
    filter_terrein,
    maak_terreinprofiel,
)
from utils.config import ANALYSE, PADEN
from utils.grafieken import teken_cadanstrend
from utils.inlezen import bereken_verstreken_tijd, lees_fit_bestand, schoon_fit_data
from utils.visualisatie import exporteer_afbeelding

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instellingen

# COMMAND ----------

ETAPPE_NR = 8
FIT_BESTAND = PADEN.fit_map / "20260919_stubaier_hohenweg_etappe_8.fit"
TERREIN = "klim"                          # Kies 'klim', 'afdaling' of 'vlak'
BASELINE_VAN = "20260912"                 # Eerste dag van de baseline (JJJJMMDD)
BASELINE_TOT = "20260918"                 # Laatste dag van de baseline (JJJJMMDD)
SAMENVATTING_BESTAND = PADEN.data_map / "klimcadans_per_etappe.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening 1: baseline uit de eerdere etappes
# MAGIC
# MAGIC Zijn de FIT-bestanden beschikbaar, dan wordt de klimcadans per etappe opnieuw berekend en opgeslagen. Anders wordt de eerder opgeslagen samenvatting gebruikt.

# COMMAND ----------

if any(PADEN.fit_map.glob("*stubaier_hohenweg*.fit")):
    samenvatting = bereken_klimcadans_per_bestand(PADEN.fit_map, van_datum=BASELINE_VAN, tot_datum=BASELINE_TOT)
    PADEN.data_map.mkdir(exist_ok=True)
    samenvatting.to_csv(SAMENVATTING_BESTAND, index=False)
else:
    samenvatting = pd.read_csv(SAMENVATTING_BESTAND, dtype={"datum": str})

baseline_spm = bereken_baseline(samenvatting)
print(samenvatting.round(1).to_string(index=False))
print(f"\nBaseline klimcadans ({BASELINE_VAN} t/m {BASELINE_TOT}): {baseline_spm:.1f} spm")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data inlezen: de te onderzoeken etappe

# COMMAND ----------

ruwe_data = lees_fit_bestand(FIT_BESTAND)
df = schoon_fit_data(ruwe_data, cadans_factor=ANALYSE.wandel_cadans_factor)
df = bereken_verstreken_tijd(df)

print(f"{len(df)} metingen ingelezen: {df['tijd'].min():%d-%m-%Y %H:%M} t/m {df['tijd'].max():%H:%M}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Berekening 2: terreinprofiel en cadanstrend

# COMMAND ----------

profiel = maak_terreinprofiel(df)
gefilterd = bereken_cadanstrend(filter_terrein(profiel, terrein=TERREIN))

print(f"{len(gefilterd)} rasterpunten van {ANALYSE.terrein_rooster_m:.0f} m op terreintype '{TERREIN}'.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grafiek

# COMMAND ----------

fig = teken_cadanstrend(gefilterd, profiel, baseline_spm=baseline_spm, etappe_nr=ETAPPE_NR, terrein=TERREIN)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exporteren
# MAGIC
# MAGIC Voor klimmen blijft de oorspronkelijke bestandsnaam behouden. Voor de andere terreintypen komt het terreintype achter de naam.

# COMMAND ----------

suffix = "" if TERREIN == "klim" else f"_{TERREIN}"
exporteer_afbeelding(fig, bestandsnaam=f"20260919_accumulated_fatigue_cadence{suffix}", formaat="liggend")